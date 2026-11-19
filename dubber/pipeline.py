"""Главный конвейер: видео -> речь -> текст -> перевод -> голос -> видео."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import soundfile as sf

from . import asr, prosody
from .languages import SOURCE_NLLB, TARGET_LANGUAGES
from .utils import (compact_silence, configure_utf8_console, extract_audio, free_gpu, get_device,
                    has_audio, load_wav, media_duration, mux_video, normalize_audio, rms, run,
                    run_worker, save_wav, setup_cuda_dlls, time_stretch, trim_silence,
                    video_dimensions, write_ass, write_srt)

configure_utf8_console()
setup_cuda_dlls()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(ROOT, "work")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

Progress = Callable[[float, str], None]


@dataclass
class Job:
    video: str
    workdir: str
    duration: float = 0.0
    audio: str = ""          # исходная дорожка 44.1 кГц стерео
    vocals: str = ""         # голос (после Demucs) или исходная дорожка
    background: str = ""     # фон без голоса (Demucs) или исходная дорожка
    separated: bool = False
    src_lang: str = ""
    segments: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def say(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            try:
                enc = getattr(sys.stdout, "encoding", None) or "utf-8"
                print(line.encode(enc, errors="replace").decode(enc), flush=True)
            except Exception:
                pass
        except Exception:
            pass
        self.log.append(line)


def _sub(progress: Progress | None, lo: float, hi: float) -> Progress:
    def f(p: float, msg: str = ""):
        if progress:
            progress(lo + (hi - lo) * max(0.0, min(1.0, p)), msg)
    return f


def _new_workdir() -> str:
    os.makedirs(WORK_DIR, exist_ok=True)
    wd = os.path.join(WORK_DIR, time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    os.makedirs(wd)
    return wd


def new_job(video_path: str) -> Job:
    wd = _new_workdir()
    ext = os.path.splitext(video_path)[1] or ".mp4"
    local = os.path.join(wd, "input" + ext)
    shutil.copy(video_path, local)
    return Job(video=local, workdir=wd)


def new_job_from_url(url: str, progress: Progress | None = None) -> Job:
    """Скачивает видео по ссылке (YouTube, TikTok и др. — всё, что умеет yt-dlp)."""
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed: run  pip install -U yt-dlp") from e

    wd = _new_workdir()
    job = Job(video="", workdir=wd)
    job.say(f"Downloading video: {url}")

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total, "Downloading video…")

    opts = {
        "outtmpl": os.path.join(wd, "input.%(ext)s"),
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "")
    except Exception as e:
        raise RuntimeError(f"Failed to download video from URL: {e}") from e

    files = [f for f in os.listdir(wd) if f.startswith("input.")]
    if not files:
        raise RuntimeError("Video downloaded, but file was not found.")
    job.video = os.path.join(wd, files[0])
    job.say(f"Downloaded: {title}")
    return job


def separate_vocals(job: Job, progress: Progress | None = None):
    """Demucs отделяет голос от музыки/шумов: голос нужен для клонирования, фон — для итогового микса."""
    out_dir = os.path.join(job.workdir, "demucs")
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
           "-d", get_device(), "-o", out_dir, job.audio]
    if progress:
        progress(0.1, "Demucs: separating vocals from background…")
    run(cmd)
    stem = os.path.splitext(os.path.basename(job.audio))[0]
    base = os.path.join(out_dir, "htdemucs", stem)
    job.vocals = os.path.join(base, "vocals.wav")
    job.background = os.path.join(base, "no_vocals.wav")
    job.separated = True
    free_gpu()


def analyze(job: Job, source_lang: str, target_name: str, whisper_model: str = "large-v3",
            mt_model: str = "facebook/nllb-200-distilled-1.3B", use_demucs: bool = True,
            progress: Progress | None = None) -> Job:
    """Шаг 1: извлечение звука, (опц.) Demucs, распознавание, перевод."""
    device = get_device()
    job.say(f"Device: {device.upper()}")
    if not has_audio(job.video):
        raise RuntimeError("The video has no audio track.")
    job.duration = media_duration(job.video)
    job.audio = extract_audio(job.video, os.path.join(job.workdir, "audio.wav"))
    job.vocals = job.background = job.audio
    job.say(f"Audio extracted, duration {job.duration:.1f} s")

    if use_demucs:
        job.say("Separating voice from music (Demucs)…")
        separate_vocals(job, _sub(progress, 0.0, 0.25))
        job.say("Vocals and background separated")

    job.say(f"Transcribing speech (Whisper {whisper_model})…")
    asr_json = os.path.join(job.workdir, "asr.json")
    run_worker(
        [sys.executable, "-u", "-m", "dubber.asr", "--audio", job.vocals,
         "--model", whisper_model, "--lang", source_lang, "--device", device, "--out", asr_json],
        cwd=ROOT,
        progress=_sub(progress, 0.25, 0.65),
    )
    with open(asr_json, "r", encoding="utf-8") as f:
        asr_data = json.load(f)
    segs, lang = asr_data["segments"], asr_data["language"]
    job.src_lang = lang
    segs = asr.merge_segments(segs)
    job.say(f"Detected language: {lang}; segments: {len(segs)}")
    if not segs:
        raise RuntimeError("No speech detected in the video.")

    src_nllb = SOURCE_NLLB.get(lang)
    if src_nllb is None:
        raise RuntimeError(f"Language '{lang}' is not yet supported for translation.")
    tgt_nllb = TARGET_LANGUAGES[target_name][1]
    if src_nllb == tgt_nllb:
        raise RuntimeError(f"Source language ({lang}) matches target language ({target_name}). "
                           f"Please choose a different language in 'Translate to'.")
    job.say(f"Translating {src_nllb} -> {tgt_nllb} ({mt_model.split('/')[-1]})...")
    mt_in = os.path.join(job.workdir, "mt_in.json")
    mt_out = os.path.join(job.workdir, "mt_out.json")
    with open(mt_in, "w", encoding="utf-8") as f:
        json.dump([s["text"] for s in segs], f, ensure_ascii=False)
    run_worker(
        [sys.executable, "-u", "-m", "dubber.translate", "--inp", mt_in, "--out", mt_out,
         "--src", src_nllb, "--tgt", tgt_nllb, "--model", mt_model, "--device", device],
        cwd=ROOT,
        progress=_sub(progress, 0.65, 1.0),
    )
    with open(mt_out, "r", encoding="utf-8") as f:
        tr = json.load(f)
    for s, t in zip(segs, tr):
        s["translation"] = t
    job.segments = segs
    job.say("Translation ready — you can edit the text before dubbing")
    return job


def _reference_clip(audio: np.ndarray, sr: int, segments: list[dict], target_sec: float = 24.0) -> np.ndarray:
    """Собирает чистый нормализованный образец голоса из самых длинных фраз."""
    order = sorted(segments, key=lambda s: s["end"] - s["start"], reverse=True)
    parts, total = [], 0.0
    pause = np.zeros(int(0.06 * sr), dtype=np.float32)
    for s in order:
        a, b = int(s["start"] * sr), int(s["end"] * sr)
        chunk = compact_silence(trim_silence(audio[a:b], sr), sr, max_pause_sec=0.10)
        if chunk.size < int(0.8 * sr):
            continue
        parts.append(_fade_edges(normalize_audio(chunk), sr, fade_ms=10.0))
        parts.append(pause)
        total += chunk.size / sr
        if total >= target_sec:
            break
    if not parts:
        parts = [normalize_audio(audio[: int(target_sec * sr)])]
    return np.concatenate(parts)


def _select_reference_clips(audio: np.ndarray, sr: int, segments: list[dict],
                            pros: list[prosody.Prosody] | None, workdir: str) -> tuple[list[str], np.ndarray]:
    """Отбирает наиболее чистые нормализованные речевые сегменты для эталонного отпечатка голоса."""
    scored: list[tuple[float, np.ndarray]] = []
    for i, s in enumerate(segments):
        a, b = int(s["start"] * sr), int(s["end"] * sr)
        raw = audio[a:b]
        clean = compact_silence(trim_silence(raw, sr, threshold=0.008), sr, max_pause_sec=0.10)
        dur = clean.size / sr
        if dur < 1.0:
            continue
        p = pros[i] if pros and i < len(pros) else None
        if p and p.db < -50.0:
            continue
        dur_score = min(dur, 10.0) / 10.0 if dur <= 14.0 else 0.8
        v_score = p.voiced_ratio if p else 0.5
        calm_bonus = 0.3 if (p and abs(p.arousal) < 0.95 and p.f0 is not None) else 0.0
        score = 1.5 * v_score + dur_score + calm_bonus
        norm_clean = _fade_edges(normalize_audio(clean), sr, fade_ms=10.0)
        scored.append((score, norm_clean))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        fallback = _reference_clip(audio, sr, segments, target_sec=20.0)
        p = save_wav(os.path.join(workdir, "voice_ref.wav"), fallback, sr)
        return [p], fallback[: int(18.0 * sr)]

    master_parts: list[np.ndarray] = []
    master_sec = 0.0
    pause = np.zeros(int(0.06 * sr), dtype=np.float32)
    for _, clip in scored:
        master_parts.append(clip)
        master_parts.append(pause)
        master_sec += clip.size / sr
        if master_sec >= 24.0:
            break
    master_clip = np.concatenate(master_parts)[: int(24.0 * sr)]
    save_wav(os.path.join(workdir, "voice_ref_master.wav"), master_clip, sr)

    # Передаём XTTS до 6 различных чистых фрагментов без дублирования мастер-склейки
    ref_paths: list[str] = []
    for idx, (_, clip) in enumerate(scored[:6]):
        clipped = clip[: int(12.0 * sr)]
        ref_paths.append(save_wav(os.path.join(workdir, f"voice_ref_{idx}.wav"), clipped, sr))

    best_clip = master_clip[: int(18.0 * sr)]
    return ref_paths, best_clip


def _fade_edges(wav: np.ndarray, sr: int, fade_ms: float = 12.0) -> np.ndarray:
    """Мягкий фейд краёв фразы для устранения щелчков при склейке."""
    n = min(int(sr * fade_ms / 1000.0), wav.size // 4)
    if n < 4:
        return wav
    out = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def synthesize(job: Job, target_name: str, bg_mode: str, bg_volume: float, max_speed: float,
               per_segment_voice: bool, temperature: float, keep_original: bool,
               expressiveness: float = 0.8, progress: Progress | None = None,
               deep_voice_clone: bool = True, subtitle_mode: str = "none") -> tuple[str, list[str]]:
    """Шаг 2: озвучка переведённого текста голосом оригинала и сборка видео.

    expressiveness (0..1) — насколько сильно переносить интонацию оригинала (громкость, тон, крик).
    deep_voice_clone — второй проход диффузионного переноса тембра (Seed-VC) поверх TTS для точной копии голоса.
    subtitle_mode — 'none' | 'burn_tgt' | 'burn_src' | 'soft'.
    """
    from .tts import XTTS_SR

    device = get_device()
    lang_code, tgt_nllb, xtts_lang = TARGET_LANGUAGES[target_name]
    segs = [s for s in job.segments if s.get("translation", "").strip()]
    if not segs:
        raise RuntimeError("No text to synthesize.")

    # Всегда отделяем чистый голос через Demucs для клонирования (даже если фон в миксе не нужен),
    # иначе музыка и интершум попадают в энкодер XTTS/Seed-VC и искажают голос.
    if not job.separated:
        job.say("Separating voice from music (Demucs)…")
        separate_vocals(job, _sub(progress, 0.0, 0.1))

    sr = XTTS_SR
    voice_audio = load_wav(job.vocals, sr)

    expr = float(np.clip(expressiveness, 0.0, 1.0))
    pros = prosody.analyze_segments(voice_audio, sr, segs)
    if pros:
        job.say(f"Prosody analyzed: loud/emphatic segments — {sum(p.shout for p in pros)}")
    f0_vals = [p.f0 for p in pros if p and p.f0]
    global_f0 = float(np.median(f0_vals)) if f0_vals else None

    ref_paths, best_ref_clip = _select_reference_clips(voice_audio, sr, segs, pros, job.workdir)
    best_ref_path = save_wav(os.path.join(job.workdir, "_best_ref.wav"), best_ref_clip, sr)

    seg_dir = os.path.join(job.workdir, "segments")
    os.makedirs(seg_dir, exist_ok=True)

    seg_meta: list[dict] = []
    for i, seg in enumerate(segs):
        info = pros[i] if pros and i < len(pros) else None
        seg_meta.append({
            "idx": i,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg.get("text", ""),
            "translation": seg["translation"],
            "f0": float(info.f0) if (info and info.f0 is not None) else None,
            "shout": bool(info.shout) if info else False,
            "arousal": float(info.arousal) if info else 0.0,
            "f0_slope": float(info.f0_slope) if info else 0.0,
            "voiced_ratio": float(info.voiced_ratio) if info else 0.5,
        })

    # Этап синтеза (Silero v5 / XTTS v2) выполняется в изолированном подпроцессе,
    # чтобы драйверы CUDA гарантированно освобождали VRAM (включая контекст cuBLAS/cuDNN) перед Seed-VC.
    tts_end = 0.55 if deep_voice_clone else 0.85
    tts_in = os.path.join(job.workdir, "tts_in.json")
    tts_out = os.path.join(job.workdir, "tts_out.json")
    with open(tts_in, "w", encoding="utf-8") as f:
        json.dump({
            "device": device,
            "workdir": job.workdir,
            "vocals": job.vocals,
            "sr": sr,
            "lang_code": lang_code,
            "xtts_lang": xtts_lang,
            "src_lang": job.src_lang,
            "deep_voice_clone": bool(deep_voice_clone),
            "global_f0": global_f0,
            "expressiveness": expr,
            "per_segment_voice": bool(per_segment_voice),
            "temperature": float(temperature),
            "max_speed": float(max_speed),
            "duration": float(job.duration),
            "ref_paths": ref_paths,
            "best_ref_path": best_ref_path,
            "segments": seg_meta,
        }, f, ensure_ascii=False)

    if progress:
        progress(0.10, "Synthesizing speech…")
    run_worker(
        [sys.executable, "-u", "-m", "dubber.tts", "--manifest", tts_in, "--out", tts_out],
        cwd=ROOT,
        progress=_sub(progress, 0.10, tts_end),
    )
    with open(tts_out, "r", encoding="utf-8") as f:
        tts_data = json.load(f)
    for msg in tts_data.get("logs", []):
        job.say(msg)

    tts_items = tts_data.get("items", [])

    # Этап переноса тембра (Seed-VC) также изолирован в подпроцессе — основной UI остаётся с 0 МБ VRAM
    vc_map: dict[str, str] = {}
    if deep_voice_clone and tts_items:
        job.say("Loading Seed-VC (deep voice timbre transfer)…")
        vc_in = os.path.join(job.workdir, "vc_in.json")
        vc_out = os.path.join(job.workdir, "vc_out.json")
        # Для неродственных языковых пар (например en -> ru/sk) отключаем речевой mel_r промпт,
        # чтобы иностранная 'r' не переносилась в дубляж; между родственными славянскими языками
        # живой mel_r сохраняет мягкость и естественную текстуру голоса без акцента.
        slavic = {"ru", "uk", "be", "sk", "cs", "pl", "bg", "sr", "hr"}
        strip_prompt = (not job.src_lang) or (
            job.src_lang != lang_code and not (job.src_lang in slavic and lang_code in slavic)
        )
        with open(vc_in, "w", encoding="utf-8") as f:
            json.dump({
                "device": device,
                "workdir": job.workdir,
                "vocals": job.vocals,
                "sr": sr,
                "per_segment_voice": bool(per_segment_voice),
                "cross_lingual": strip_prompt,
                "best_ref_path": best_ref_path,
                "steps": 35,
                "cfg_rate": 0.50 if strip_prompt else 0.45,
                "segments": seg_meta,
                "items": tts_items,
            }, f, ensure_ascii=False)
        run_worker(
            [sys.executable, "-u", "-m", "dubber.vc", "--manifest", vc_in, "--out", vc_out],
            cwd=ROOT,
            progress=_sub(progress, 0.56, 0.86),
        )
        with open(vc_out, "r", encoding="utf-8") as f:
            vc_data = json.load(f)
        for warn in vc_data.get("warnings", []):
            job.say(warn)
        vc_map = vc_data.get("converted", {})

    synthesized: list[tuple[int, dict, np.ndarray, prosody.Prosody | None, np.ndarray, float]] = []
    for item in tts_items:
        i = int(item["idx"])
        seg = segs[i]
        info = pros[i] if pros and i < len(pros) else None
        a, b = int(seg["start"] * sr), int(seg["end"] * sr)
        raw_seg = voice_audio[a:b]
        wav_path = vc_map.get(str(i), item["wav_path"])
        wav, _ = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        synthesized.append((i, seg, raw_seg, info, wav, float(item["native_speed"])))

    total_len = int((job.duration + 1.0) * sr)
    track = np.zeros(total_len, dtype=np.float32)
    cursor = 0.0
    pref_speed = max(float(max_speed), 1.05)
    dub_segs: list[dict] = []

    # Обратный расчёт предельных дедлайнов: дубляж никогда не отстаёт от сцен и не обрывается на конце видео
    end_cap = max(job.duration - 0.04, 0.30)
    latest_ends = [end_cap] * len(synthesized)
    for k in range(len(synthesized) - 2, -1, -1):
        _, _, _, _, w_next, _ = synthesized[k + 1]
        min_next = max((w_next.size / sr) / 1.9, 0.18)
        next_seg_start = synthesized[k + 1][1]["start"]
        latest_ends[k] = max(next_seg_start, min(latest_ends[k + 1] - min_next, next_seg_start + 0.28))

    for idx_out, (i, seg, raw_seg, info, wav, native_speed) in enumerate(synthesized):
        if progress:
            progress(0.86 + 0.05 * idx_out / max(len(synthesized), 1), f"Mastering segment {idx_out + 1}/{len(synthesized)}")

        # При активном Seed-VC высоту тона корректируем мягко и только при заметном сдвиге (>1.5 полутона),
        # чтобы не трогать естественно восстановленные форманты.
        if info and info.f0 and expr > 0.05:
            if deep_voice_clone:
                cur_f0 = prosody.median_f0(wav, sr)
                if cur_f0 and abs(12.0 * np.log2(info.f0 / cur_f0)) > 1.5:
                    wav = prosody.align_pitch_resample(wav, sr, info.f0, expr=expr, max_semitones=1.0)
            else:
                wav = prosody.align_pitch_resample(wav, sr, info.f0, expr=expr, max_semitones=1.5)

        start = max(seg["start"], cursor)
        is_last = idx_out == len(synthesized) - 1
        next_start = synthesized[idx_out + 1][1]["start"] if not is_last else end_cap
        available = max(next_start - start, 0.18)
        raw_length = wav.size / sr

        if raw_length > available:
            # Сначала убираем лишние внутренние паузы синтезатора без изменения скорости самой речи
            wav = compact_silence(wav, sr, max_pause_sec=0.05, threshold=0.012)
        length = wav.size / sr

        if length > available:
            drift = max(0.0, start - seg["start"])
            slack = max(0.0, latest_ends[idx_out] - next_start)
            max_spill = 0.0 if is_last else min(0.28, max(0.0, 0.35 - drift), slack)
            ceiling = available if is_last else min(available + max_spill, max(0.18, latest_ends[idx_out] - start))
            floor = min(available, ceiling)
            target_dur = max(floor, min(length / pref_speed, ceiling))
            needed_ratio = length / max(target_dur, 0.18)
            ratio = min(needed_ratio, 2.2)
            if ratio > 1.02:
                wav = time_stretch(wav, sr, ratio, seg_dir)

        if raw_seg.size > 0 and expr > 0.05:
            wav = prosody.apply_dynamic_contour(raw_seg, wav, sr, expr=expr)
        orig_rms, dub_rms = rms(raw_seg), rms(wav)
        if orig_rms > 1e-4 and dub_rms > 1e-4:
            scale = (orig_rms / dub_rms) ** float(np.clip(expr, 0.25, 1.0))
            max_scale = 4.5 if (info and info.shout) else 2.5
            wav = wav * float(np.clip(scale, 0.35, max_scale))

        wav = _fade_edges(wav, sr)

        pos = int(start * sr)
        needed = pos + wav.size
        if needed > track.size:
            track = np.pad(track, (0, max(needed - track.size, sr * 5)))
        track[pos : pos + wav.size] += wav
        cursor = start + wav.size / sr
        dub_segs.append({
            "start": float(start),
            "end": float(min(max(cursor, start + 0.2), job.duration)),
            "text": seg.get("text", ""),
            "translation": seg.get("translation", ""),
        })
        tag = " [shout]" if info and info.shout else ""
        job.say(f"#{i + 1}{tag}: \"{seg['translation'][:60]}\" ({length:.1f} s -> {wav.size / sr:.1f} s)")

    track = prosody.soft_limit(track)
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.98:
        track *= 0.98 / peak

    dub_path = save_wav(os.path.join(job.workdir, "dub.wav"), track, sr)
    if progress:
        progress(0.92, "Muxing final video…")

    bg_wav, vol = None, 0.0
    if bg_mode == "separate":
        bg_wav, vol = job.background, bg_volume
    elif bg_mode == "original":
        bg_wav, vol = job.audio, bg_volume

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = time.strftime("%Y%m%d_%H%M%S")
    out_video = os.path.join(OUTPUT_DIR, f"dub_{lang_code}_{stem}.mp4")

    # Исключаем перекрытия соседних титров при смещении реплик дубляжа
    for k in range(len(dub_segs) - 1):
        if dub_segs[k]["end"] > dub_segs[k + 1]["start"]:
            dub_segs[k]["end"] = max(dub_segs[k]["start"] + 0.1, dub_segs[k + 1]["start"] - 0.02)

    srt_src = write_srt(os.path.join(OUTPUT_DIR, f"dub_{stem}_{job.src_lang}.srt"), segs, "text")
    srt_tgt = write_srt(os.path.join(OUTPUT_DIR, f"dub_{stem}_{lang_code}.srt"), dub_segs or segs, "translation")

    burn_ass = None
    if subtitle_mode in ("burn_tgt", "burn_src"):
        vw, vh = video_dimensions(job.video)
        ass_segs, ass_key = (dub_segs or segs, "translation") if subtitle_mode == "burn_tgt" else (segs, "text")
        burn_ass = write_ass(os.path.join(job.workdir, "sub_burn.ass"), ass_segs, ass_key, vw, vh)

    src_iso3 = SOURCE_NLLB.get(job.src_lang, "und")[:3]
    tgt_iso3 = tgt_nllb[:3]
    soft_srts = [(srt_tgt, tgt_iso3, target_name), (srt_src, src_iso3, "Original")] if subtitle_mode == "soft" else None

    mux_video(
        job.video, dub_path, out_video, bg_wav, vol, keep_original,
        subtitle_mode=subtitle_mode, burn_ass=burn_ass, soft_srts=soft_srts, duration=job.duration,
    )

    dub_copy = os.path.join(OUTPUT_DIR, f"dub_{lang_code}_{stem}.wav")
    shutil.copy(dub_path, dub_copy)
    job.say(f"Done: {out_video}")
    if progress:
        progress(1.0, "Done")
    return out_video, [out_video, dub_copy, srt_tgt, srt_src]
