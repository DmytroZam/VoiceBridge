"""Глубокий перенос тембра голоса (Zero-Shot Diffusion Voice Conversion — Seed-VC)."""
from __future__ import annotations

import gc
import os
import warnings

import numpy as np
import yaml

from .utils import free_gpu, normalize_audio, setup_cuda_dlls, trim_silence

setup_cuda_dlls()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402

SEEDVC_REPO = "Plachta/Seed-VC"
SEEDVC_CKPT = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
SEEDVC_CFG = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
CAMPPLUS_REPO = "funasr/campplus"
CAMPPLUS_FILE = "campplus_cn_common.bin"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"


def _crossfade(prev_tail: np.ndarray, next_chunk: np.ndarray, overlap: int) -> np.ndarray:
    n = min(overlap, prev_tail.size, next_chunk.size)
    if n <= 0:
        return next_chunk
    out = next_chunk.copy()
    fade_out = (np.cos(np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)) ** 2).astype(np.float32)
    fade_in = 1.0 - fade_out
    out[:n] = out[:n] * fade_in + prev_tail[-n:] * fade_out
    return out


class VoiceRefiner:
    """Переносит тембр, текстуру голосовых связок и регистр оригинального спикера на синтезированную речь."""

    def __init__(self, device: str = "cuda"):
        from huggingface_hub import hf_hub_download
        from transformers import AutoFeatureExtractor, WhisperModel

        from .seedvc.audio import mel_spectrogram
        from .seedvc.bigvgan import bigvgan
        from .seedvc.campplus.DTDNN import CAMPPlus
        from .seedvc.commons import build_model, load_checkpoint, recursive_munch

        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.use_fp16 = self.device.type == "cuda"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dit_ckpt = hf_hub_download(repo_id=SEEDVC_REPO, filename=SEEDVC_CKPT)
            dit_cfg = hf_hub_download(repo_id=SEEDVC_REPO, filename=SEEDVC_CFG)
            camp_ckpt = hf_hub_download(repo_id=CAMPPLUS_REPO, filename=CAMPPLUS_FILE)

            with open(dit_cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            params = recursive_munch(cfg["model_params"])
            params.dit_type = "DiT"
        self.hop_length = int(cfg["preprocess_params"]["spect_params"]["hop_length"])
        self.sr = int(cfg["preprocess_params"]["sr"])
        self.overlap_frame_len = 16
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length
        self.max_context_window = (self.sr // self.hop_length) * 30

        # Загружаем подмодели поочерёдно со сборкой мусора после каждой, чтобы не упереться в лимит файла подкачки Windows
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_name = getattr(params.speech_tokenizer, "name", "openai/whisper-small")
            w_dtype = torch.float16 if self.use_fp16 else torch.float32
            self.whisper = WhisperModel.from_pretrained(w_name, dtype=w_dtype)
            del self.whisper.decoder
            self.whisper.to(self.device).eval()
            self.w_feat = AutoFeatureExtractor.from_pretrained(w_name)
            gc.collect()

            self.model = build_model(params, stage="DiT")
            self.model, _, _, _ = load_checkpoint(
                self.model, None, dit_ckpt, load_only_params=True, ignore_modules=[], is_distributed=False
            )
            for k in self.model:
                self.model[k].eval().to(self.device)
            self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            gc.collect()

            self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus.load_state_dict(torch.load(camp_ckpt, map_location="cpu"))
            self.campplus.eval().to(self.device)
            gc.collect()

            self.bigvgan = bigvgan.BigVGAN.from_pretrained(BIGVGAN_REPO, use_cuda_kernel=False)
            self.bigvgan.remove_weight_norm()
            self.bigvgan.eval().to(self.device)
        free_gpu()

        spect = cfg["preprocess_params"]["spect_params"]
        fmax = None if spect.get("fmax", "None") in ("None", None) else int(spect["fmax"])
        mel_args = {
            "n_fft": int(spect["n_fft"]),
            "win_size": int(spect["win_length"]),
            "hop_size": self.hop_length,
            "num_mels": int(spect["n_mels"]),
            "sampling_rate": self.sr,
            "fmin": int(spect.get("fmin", 0)),
            "fmax": fmax,
            "center": False,
        }
        self._mel_fn = lambda x: mel_spectrogram(x, **mel_args)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    def _semantic_features(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Извлекает языконезависимые фонетические признаки Whisper-encoder (поддерживает фрагменты >30 с)."""
        total = wav_16k.size(-1)
        win_samples = 16000 * 30
        if total <= win_samples:
            return self._semantic_window(wav_16k)

        overlap_sec = 5
        step_samples = 16000 * (30 - overlap_sec)
        chunks = []
        pos = 0
        while pos < total:
            end = min(pos + win_samples, total)
            feats = self._semantic_window(wav_16k[:, pos:end])
            if pos == 0:
                chunks.append(feats)
            else:
                chunks.append(feats[:, 50 * overlap_sec :])
            if end >= total:
                break
            pos += step_samples
        return torch.cat(chunks, dim=1)

    @torch.inference_mode()
    def _semantic_window(self, wav_16k: torch.Tensor) -> torch.Tensor:
        inputs = self.w_feat(
            [wav_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        masked = self.whisper._mask_input_features(
            inputs.input_features, attention_mask=inputs.attention_mask
        ).to(self.device)
        out = self.whisper.encoder(
            masked.to(self.whisper.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state.to(torch.float32)[:, : wav_16k.size(-1) // 320 + 1]

    @torch.inference_mode()
    def _campplus_emb(self, wav_16k: torch.Tensor) -> torch.Tensor:
        feat = torchaudio.compliance.kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus(feat.unsqueeze(0))

    @torch.inference_mode()
    def speaker_embedding(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Возвращает нормализованный 192-мерный вектор спикера (CAMP++) для кластеризации голосов."""
        import soxr

        w = wav.astype(np.float32, copy=False)
        if sr != 16000:
            w = soxr.resample(w, sr, 16000, quality="HQ").astype(np.float32)
        t = torch.from_numpy(w).unsqueeze(0).to(self.device)
        emb = F.normalize(self._campplus_emb(t), dim=-1)
        return emb.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def prepare_reference(self, ref_wav: np.ndarray, ref_sr: int, max_sec: float = 18.0,
                          prompt_sec: float = 3.0, cross_lingual: bool = True) -> dict:
        """Предвычисляет вектор тембра (CAMP++ по полным до 18 с) и контекстный промпт DiT.

        При кросс-язычном дубляже (cross_lingual=True) в mel_r подаётся только короткая тишина (0.15 с),
        чтобы DiT брал 100% тембра из языконезависимого CAMP++ и 0% иностранной фонетики (убирает английское 'r').
        """
        import soxr

        w = trim_silence(ref_wav, ref_sr, threshold=0.01, pad=0.04)
        if w.size == 0:
            w = ref_wav
        w = normalize_audio(w, target_peak=0.70, target_rms=0.09)
        if ref_sr != self.sr:
            w = soxr.resample(w, ref_sr, self.sr, quality="HQ").astype(np.float32)
        w = w[: int(max_sec * self.sr)]
        if w.size < int(0.5 * self.sr):
            w = np.pad(w, (0, int(0.5 * self.sr) - w.size))

        # Извлекаем независимый от языка и фонем вектор тембра CAMP++ из всех чистых фрагментов
        full_t = torch.from_numpy(w).unsqueeze(0).float().to(self.device)
        style2 = self._campplus_emb(torchaudio.functional.resample(full_t, self.sr, 16000))

        if cross_lingual:
            prompt_w = np.zeros(int(0.15 * self.sr), dtype=np.float32)
        else:
            max_p = int(prompt_sec * self.sr)
            if w.size > max_p:
                lo = min(int(1.6 * self.sr), max_p)
                hi = min(int(2.8 * self.sr), w.size)
                win = max(int(0.02 * self.sr), 1)
                cut = max_p
                if hi - lo > win:
                    seg = w[lo:hi]
                    csum = np.concatenate([[0.0], np.cumsum(seg * seg, dtype=np.float64)])
                    energies = csum[win:] - csum[:-win]
                    cut = lo + int(np.argmin(energies)) + win // 2
                prompt_w = _fade_edges_np(w[:cut], self.sr, fade_ms=12.0)
            else:
                prompt_w = _fade_edges_np(w, self.sr, fade_ms=12.0)
            prompt_w = np.concatenate([prompt_w, np.zeros(int(0.25 * self.sr), dtype=np.float32)])

        ref_t = torch.from_numpy(prompt_w).unsqueeze(0).float().to(self.device)
        ref_16k = torchaudio.functional.resample(ref_t, self.sr, 16000)
        s_ori = self._semantic_features(ref_16k)
        mel_r = self._mel_fn(ref_t)
        r_len = torch.LongTensor([mel_r.size(2)]).to(self.device)
        prompt_cond, *_ = self.model.length_regulator(s_ori, ylens=r_len, n_quantizers=3, f0=None)
        return {"mel_r": mel_r, "style2": style2, "prompt_cond": prompt_cond}

    @torch.inference_mode()
    def convert(self, src_wav: np.ndarray, src_sr: int, target_ref: dict,
                out_sr: int = 24000, steps: int = 35, cfg_rate: float = 0.50) -> np.ndarray:
        """Конвертирует фразу синтезатора в голос и тембр целевого спикера."""
        import soxr

        w = trim_silence(src_wav, src_sr, threshold=0.008, pad=0.04)
        if w.size < int(0.15 * src_sr):
            return src_wav
        w = normalize_audio(w, target_peak=0.70, target_rms=0.09)
        if src_sr != self.sr:
            w = soxr.resample(w, src_sr, self.sr, quality="HQ").astype(np.float32)

        # Небольшие нулевые поля исключают срез последних согласных или призвуки на границах BigVGAN
        pad_pre = np.zeros(int(0.08 * self.sr), dtype=np.float32)
        pad_post = np.zeros(int(0.10 * self.sr), dtype=np.float32)
        w_padded = np.concatenate([pad_pre, w, pad_post])

        src_t = torch.from_numpy(w_padded).unsqueeze(0).float().to(self.device)
        src_16k = torchaudio.functional.resample(src_t, self.sr, 16000)
        s_alt = self._semantic_features(src_16k)
        mel_s = self._mel_fn(src_t)
        t_len = torch.LongTensor([mel_s.size(2)]).to(self.device)
        cond, *_ = self.model.length_regulator(s_alt, ylens=t_len, n_quantizers=3, f0=None)

        mel_r = target_ref["mel_r"]
        style2 = target_ref["style2"]
        prompt_cond = target_ref["prompt_cond"]
        max_source_window = max(self.max_context_window - mel_r.size(2), self.overlap_frame_len * 4)

        processed = 0
        wave_chunks: list[np.ndarray] = []
        prev_tail: np.ndarray | None = None

        while processed < cond.size(1):
            chunk_cond = cond[:, processed : processed + max_source_window]
            is_last = processed + max_source_window >= cond.size(1)
            cat_cond = torch.cat([prompt_cond, chunk_cond], dim=1)
            cat_lens = torch.LongTensor([cat_cond.size(1)]).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.use_fp16 else torch.float32):
                vc_mel = self.model.cfm.inference(
                    cat_cond, cat_lens, mel_r, style2, None, steps, inference_cfg_rate=cfg_rate
                )
                vc_mel = vc_mel[:, :, mel_r.size(-1) :]
            vc_wave = self.bigvgan(vc_mel.float()).squeeze().detach().cpu().numpy().astype(np.float32)
            if vc_wave.ndim > 1:
                vc_wave = vc_wave.reshape(-1)

            if processed == 0:
                if is_last:
                    wave_chunks.append(vc_wave)
                    break
                wave_chunks.append(vc_wave[: -self.overlap_wave_len])
                prev_tail = vc_wave[-self.overlap_wave_len :]
            elif is_last:
                assert prev_tail is not None
                wave_chunks.append(_crossfade(prev_tail, vc_wave, self.overlap_wave_len))
                break
            else:
                assert prev_tail is not None
                chunk_out = _crossfade(prev_tail, vc_wave[: -self.overlap_wave_len], self.overlap_wave_len)
                wave_chunks.append(chunk_out)
                prev_tail = vc_wave[-self.overlap_wave_len :]

            step_frames = max(vc_mel.size(2) - self.overlap_frame_len, 1)
            processed += step_frames

        out = np.concatenate(wave_chunks) if wave_chunks else w
        out = trim_silence(out, self.sr, threshold=0.008, pad=0.03)
        if out.size == 0:
            return src_wav
        if out_sr != self.sr:
            out = soxr.resample(out, self.sr, out_sr, quality="VHQ").astype(np.float32)
        return out

    def close(self):
        for attr in ("model", "campplus", "bigvgan", "whisper", "w_feat"):
            if hasattr(self, attr):
                delattr(self, attr)
        free_gpu()


def _fade_edges_np(wav: np.ndarray, sr: int, fade_ms: float = 8.0) -> np.ndarray:
    n = min(int(sr * fade_ms / 1000.0), wav.size // 4)
    if n < 4:
        return wav
    out = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def build_vc_references(refiner: VoiceRefiner, audio: np.ndarray, sr: int, segs: list[dict],
                        per_segment: bool, fallback_clip: np.ndarray, workdir: str,
                        cross_lingual: bool = True) -> tuple[dict, dict[int, dict]]:
    """Подбирает и кэширует чистые референсы для Seed-VC с кластеризацией дикторов через CAMP++."""
    from .utils import compact_silence, save_wav

    items: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for s in segs:
        idx = int(s["idx"])
        a, b = int(s["start"] * sr), int(s["end"] * sr)
        clean = compact_silence(trim_silence(audio[a:b], sr, threshold=0.008), sr, max_pause_sec=0.08)
        dur = clean.size / sr
        if dur < 0.7:
            continue
        v_ratio = float(s.get("voiced_ratio", 0.5))
        clarity = v_ratio + min(dur, 8.0) / 16.0
        norm_clip = _fade_edges_np(normalize_audio(clean), sr, fade_ms=8.0)
        emb = refiner.speaker_embedding(norm_clip, sr)
        items.append((idx, norm_clip, emb, clarity))

    if not items:
        return refiner.prepare_reference(fallback_clip, sr, cross_lingual=cross_lingual), {}

    clusters: list[dict] = []
    seg_to_cluster: dict[int, int] = {}
    for seg_idx, clip, emb, clarity in sorted(items, key=lambda x: x[3], reverse=True):
        best_cid, best_sim = -1, -1.0
        for cid, cl in enumerate(clusters):
            sim = float(np.dot(emb, cl["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        if best_cid >= 0 and best_sim >= 0.68:
            cl = clusters[best_cid]
            cl["members"].append((seg_idx, clip, emb, clarity))
            mean_e = np.mean([m[2] for m in cl["members"]], axis=0)
            cl["centroid"] = mean_e / max(float(np.linalg.norm(mean_e)), 1e-6)
            cl["total_sec"] += clip.size / sr
            seg_to_cluster[seg_idx] = best_cid
        else:
            cid = len(clusters)
            clusters.append({
                "members": [(seg_idx, clip, emb, clarity)],
                "centroid": emb.copy(),
                "total_sec": clip.size / sr,
            })
            seg_to_cluster[seg_idx] = cid

    def _compile_cluster_ref(members: list[tuple[int, np.ndarray, np.ndarray, float]]) -> dict:
        pause = np.zeros(int(0.05 * sr), dtype=np.float32)
        parts: list[np.ndarray] = []
        total = 0.0
        for _, m_clip, _, _ in sorted(members, key=lambda x: x[3], reverse=True):
            parts.append(m_clip)
            parts.append(pause)
            total += m_clip.size / sr
            if total >= 18.0:
                break
        wav_ref = np.concatenate(parts)[: int(18.0 * sr)] if parts else fallback_clip
        return refiner.prepare_reference(wav_ref, sr, cross_lingual=cross_lingual)

    dom_cid = max(range(len(clusters)), key=lambda c: clusters[c]["total_sec"])
    dom_ref = _compile_cluster_ref(clusters[dom_cid]["members"])
    cluster_cache: dict[int, dict] = {dom_cid: dom_ref}
    seg_refs: dict[int, dict] = {}
    if per_segment:
        for seg_idx, cid in seg_to_cluster.items():
            if cid not in cluster_cache:
                cluster_cache[cid] = _compile_cluster_ref(clusters[cid]["members"])
            seg_refs[seg_idx] = cluster_cache[cid]

    dom_parts = [m[1] for m in sorted(clusters[dom_cid]["members"], key=lambda x: x[3], reverse=True)[:5]]
    if dom_parts:
        save_wav(os.path.join(workdir, "voice_ref_vc.wav"), np.concatenate(dom_parts)[: int(18.0 * sr)], sr)
    return dom_ref, seg_refs


def _vc_worker_main(manifest_path: str, out_path: str) -> None:
    """Точка входа изолированного подпроцесса Seed-VC: переносит тембр без удержания CUDA в основном процессе."""
    import faulthandler
    import json
    from .utils import load_wav, save_wav, setup_cuda_dlls

    try:
        faulthandler.enable()
    except Exception:
        pass
    setup_cuda_dlls()
    with open(manifest_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = cfg["device"]
    workdir = cfg["workdir"]
    seg_dir = os.path.join(workdir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    sr = int(cfg.get("sr", 24000))
    per_segment = bool(cfg.get("per_segment_voice", False))
    cross_lingual = bool(cfg.get("cross_lingual", True))
    steps = int(cfg.get("steps", 35))
    cfg_rate = float(cfg.get("cfg_rate", 0.50 if cross_lingual else 0.45))

    voice_audio = load_wav(cfg["vocals"], sr)
    best_ref_clip = load_wav(cfg["best_ref_path"], sr)
    segs = list(cfg["segments"])
    items = list(cfg["items"])

    print("__PROG__:0.02:Loading Seed-VC voice conversion…", flush=True)
    refiner = VoiceRefiner(device)
    dom_ref, seg_refs = build_vc_references(
        refiner, voice_audio, sr, segs, per_segment, best_ref_clip, workdir, cross_lingual=cross_lingual
    )

    converted_map: dict[str, str] = {}
    warnings_out: list[str] = []
    n_items = len(items)
    for k, item in enumerate(items):
        print(f"__PROG__:{0.06 + 0.94 * k / max(n_items, 1):.4f}:Cloning voice timbre {k + 1}/{n_items}", flush=True)
        idx = int(item["idx"])
        wav = load_wav(item["wav_path"], sr)
        tgt_ref = seg_refs.get(idx, dom_ref)
        try:
            conv = refiner.convert(wav, sr, tgt_ref, out_sr=sr, steps=steps, cfg_rate=cfg_rate)
            if conv.size > int(0.15 * sr):
                vc_path = save_wav(os.path.join(seg_dir, f"vc_{idx:04d}.wav"), conv, sr)
                converted_map[str(idx)] = vc_path
        except Exception as e:
            warnings_out.append(f"Warning: Seed-VC skipped segment #{idx + 1} ({e})")

    refiner.close()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"converted": converted_map, "warnings": warnings_out}, f, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    _vc_worker_main(args.manifest, args.out)
