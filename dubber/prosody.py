"""Интонация: анализ оригинала (громкость, высота тона, крик) и перенос на дубляж."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Prosody:
    db: float           # средняя громкость фразы, дБ
    f0: float | None    # медианная высота тона, Гц
    arousal: float      # -1 (тихо/шёпот) .. 0 (обычно) .. 2 (крик)
    f0_semitones: float = 0.0  # отклонение высоты тона фразы от медианы спикера, полутоны
    f0_slope: float = 0.0      # наклон тона к концу фразы (вопрос / восклицание), полутоны
    voiced_ratio: float = 0.0  # доля чистого вокализованного сигнала во фразе (0..1)

    @property
    def shout(self) -> bool:
        return self.arousal >= 1.0


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(max(r, 1e-6))


def _f0_details(x: np.ndarray, sr: int) -> tuple[float | None, float, float]:
    """Возвращает (медианный F0, наклон F0 к концу фразы в полутонах, долю вокализованных кадров)."""
    if x.size < 4096:
        return None, 0.0, 0.0
    try:
        import librosa

        fmax = 450.0
        f0 = librosa.yin(x.astype(np.float32), fmin=70.0, fmax=fmax, sr=sr, frame_length=2048, hop_length=512)
        e = librosa.feature.rms(y=x.astype(np.float32), frame_length=2048, hop_length=512)[0]
    except Exception:
        return None, 0.0, 0.0
    n = min(len(f0), len(e))
    if n == 0:
        return None, 0.0, 0.0
    f0, e = f0[:n], e[:n]
    voiced = (e > 0.25 * e.max()) & (f0 < 0.96 * fmax) & (f0 > 72.0)
    v_ratio = float(voiced.sum()) / float(n)
    if voiced.sum() < 5:
        return None, 0.0, v_ratio
    vf0 = f0[voiced]
    # Отсекаем удвоение гармоник (октавные ошибки YIN) по нижней квартили основного тона
    q25 = float(np.percentile(vf0, 25))
    fund = vf0[vf0 <= q25 * 1.75]
    clean_f0 = fund if fund.size >= 3 else vf0
    med = float(np.median(clean_f0))
    # Оцениваем интонационный контур (начало/середина против концовки фразы)
    k = max(len(clean_f0) // 3, 2)
    head_f0 = float(np.median(clean_f0[: 2 * k]))
    tail_f0 = float(np.median(clean_f0[-k:]))
    slope_st = float(np.clip(12.0 * np.log2(max(tail_f0, 1.0) / max(head_f0, 1.0)), -6.0, 6.0))
    return med, slope_st, v_ratio


def median_f0(x: np.ndarray, sr: int) -> float | None:
    """Медианная частота основного тона по озвученным кадрам (None, если определить не удалось)."""
    med, _, _ = _f0_details(x, sr)
    return med


def analyze_segments(audio: np.ndarray, sr: int, segments: list[dict]) -> list[Prosody]:
    """Для каждой фразы определяет громкость, высоту тона и «возбуждённость» относительно среднего по ролику."""
    raw = []
    for s in segments:
        clip = audio[int(s["start"] * sr): int(s["end"] * sr)]
        db = rms_db(clip) if clip.size else -60.0
        f0, slope, v_ratio = _f0_details(clip, sr)
        raw.append((db, f0, slope, v_ratio))
    dbs = [d for d, _, _, _ in raw if d > -50]
    f0s = [f for _, f, _, _ in raw if f]
    med_db = float(np.median(dbs)) if dbs else -25.0
    med_f0 = float(np.median(f0s)) if f0s else None
    out = []
    for db, f0, slope, v_ratio in raw:
        a_db = np.clip((db - med_db) / 6.0, -1.0, 2.0)               # +6 дБ ≈ 1.0 (повышенный голос)
        st = 0.0
        if f0 and med_f0:
            st = float(np.clip(12.0 * np.log2(f0 / med_f0), -12.0, 12.0))
            a_f0 = np.clip(st / 4.0, -1.0, 2.0)                     # +4 полутона ≈ 1.0
            arousal = 0.6 * a_db + 0.4 * a_f0
        else:
            arousal = a_db
        out.append(Prosody(db=db, f0=f0, arousal=float(arousal), f0_semitones=st,
                           f0_slope=slope, voiced_ratio=v_ratio))
    return out


def apply_dynamic_contour(orig_clip: np.ndarray, dub_wav: np.ndarray, sr: int, expr: float = 0.8) -> np.ndarray:
    """Переносит плавную динамическую огибающую громкости (акценты внутри фразы) без фазовых искажений."""
    if orig_clip.size < sr // 2 or dub_wav.size < sr // 2 or expr <= 0.05:
        return dub_wav
    # Разбиваем оригинал и дубляж на плавные окна (~250 мс) для переноса макро-динамики фразы
    n_bins = max(min(int(dub_wav.size / (0.25 * sr)), 16), 3)
    orig_edges = np.linspace(0, orig_clip.size, n_bins + 1, dtype=int)
    dub_edges = np.linspace(0, dub_wav.size, n_bins + 1, dtype=int)

    orig_env = np.array([ np.sqrt(np.mean(np.square(orig_clip[orig_edges[i]:orig_edges[i + 1]]))) + 1e-5
                          for i in range(n_bins) ], dtype=np.float32)
    dub_env = np.array([ np.sqrt(np.mean(np.square(dub_wav[dub_edges[i]:dub_edges[i + 1]]))) + 1e-5
                         for i in range(n_bins) ], dtype=np.float32)

    # Игнорируем паузы, нормируем к медианной энергии активных участков
    orig_active = orig_env[orig_env > 0.15 * orig_env.max()]
    dub_active = dub_env[dub_env > 0.15 * dub_env.max()]
    if orig_active.size == 0 or dub_active.size == 0:
        return dub_wav

    orig_rel = orig_env / float(np.median(orig_active))
    dub_rel = np.maximum(dub_env / float(np.median(dub_active)), 0.25)

    # Относительная коррекция огибающей в безопасных пределах [0.75 .. 1.35]
    gain_bins = np.clip((orig_rel / dub_rel) ** (0.5 * expr), 0.75, 1.35)
    # Сглаживаем коэффициенты между соседними окнами
    if n_bins >= 3:
        gain_bins = np.convolve(np.pad(gain_bins, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")

    centers = 0.5 * (dub_edges[:-1] + dub_edges[1:])
    sample_idx = np.arange(dub_wav.size, dtype=np.float32)
    smooth_gain = np.interp(sample_idx, centers, gain_bins).astype(np.float32)
    return (dub_wav * smooth_gain).astype(np.float32)


def align_pitch_resample(wav: np.ndarray, sr: int, target_f0: float | None,
                         expr: float = 0.8, max_semitones: float = 1.5) -> np.ndarray:
    """Мягко подтягивает абсолютную высоту тона (F0) сгенерированной фразы к оригиналу
    через чистый ресемплинг soxr (без фазового вокодера и без эффекта робота)."""
    if target_f0 is None or target_f0 <= 70.0 or wav.size < 4096 or expr <= 0.05:
        return wav
    cur_f0 = median_f0(wav, sr)
    if cur_f0 is None or cur_f0 <= 70.0:
        return wav
    diff_st = 12.0 * np.log2(target_f0 / cur_f0) * float(np.clip(expr, 0.0, 1.0))
    # Ограничиваем сдвиг безопасным диапазоном ±1.5 полутона, чтобы форманты голоса оставались на 100% естественными
    diff_st = float(np.clip(diff_st, -max_semitones, max_semitones))
    if abs(diff_st) < 0.25:
        return wav
    pitch_factor = float(2.0 ** (diff_st / 12.0))
    try:
        import soxr

        # При ресемплинге с sr * pitch_factor -> sr высота тона умножается на pitch_factor без STFT-артефактов
        shifted = soxr.resample(wav.astype(np.float32), int(round(sr * pitch_factor)), sr, quality="HQ")
        return np.asarray(shifted, dtype=np.float32)
    except Exception:
        return wav


def soft_limit(x: np.ndarray, thr: float = 0.85) -> np.ndarray:
    """Мягкий лимитер: громкие места (крик) компрессируются, предотвращая клиппинг без искажения звука."""
    a = np.abs(x)
    over = a > thr
    if not over.any():
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (thr + (1.0 - thr) * np.tanh((a[over] - thr) / (1.0 - thr)))
    return y.astype(np.float32)

