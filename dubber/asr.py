"""Распознавание речи — faster-whisper (OpenAI Whisper на CTranslate2)."""
from __future__ import annotations

from typing import Callable

from .utils import free_gpu, setup_cuda_dlls

SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")


def transcribe(audio_path: str, model_size: str = "large-v3", language: str = "auto",
               device: str = "cuda", progress: Callable[[float, str], None] | None = None):
    setup_cuda_dlls(update_path=True)
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception:
        if device != "cuda":
            raise
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        audio_path,
        language=None if language == "auto" else language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )
    total = max(info.duration, 1e-6)
    result = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            result.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        if progress:
            progress(min(seg.end / total, 1.0), f"Transcribed {seg.end:.0f}/{total:.0f} s")
    lang = info.language
    del model
    free_gpu()
    return result, lang


def merge_segments(segments: list[dict], max_duration: float = 14.0, max_gap: float = 0.6) -> list[dict]:
    """Склеивает обрывки фраз в цельные предложения — так перевод получается точнее."""
    merged: list[dict] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            unfinished = not prev["text"].rstrip().endswith(SENTENCE_END)
            if unfinished and gap <= max_gap and (seg["end"] - prev["start"]) <= max_duration:
                prev["end"] = seg["end"]
                prev["text"] = (prev["text"] + " " + seg["text"]).strip()
                continue
        merged.append(dict(seg))
    return merged


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--model", default="large-v3")
    p.add_argument("--lang", default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    def _emit(frac: float, msg: str = ""):
        print(f"__PROG__:{frac:.4f}:{msg}", flush=True)

    segs, detected = transcribe(args.audio, args.model, args.lang, args.device, _emit)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"segments": segs, "language": detected}, f, ensure_ascii=False)
