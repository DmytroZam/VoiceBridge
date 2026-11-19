"""Вспомогательные функции: ffmpeg, работа с аудио, GPU."""
from __future__ import annotations

import gc
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def configure_utf8_console() -> None:
    """Обеспечивает вывод в UTF-8 на Windows, чтобы символы (стрелки, эмодзи, языки мира)
    не вызывали UnicodeEncodeError при выводе в консоль (cp1251/cp866)."""
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def patch_windows_asyncio() -> None:
    """Исправляет баг ProactorEventLoop в Windows (Python 3.11), при котором сброс HTTP-соединения
    браузером (например, при подгрузке готового MP4 в плеере) бросает ConnectionResetError [WinError 10054]
    в _ProactorBasePipeTransport._call_connection_lost.
    """
    if os.name != "nt":
        return
    try:
        import socket
        from asyncio import proactor_events

        cls = proactor_events._ProactorBasePipeTransport
        if getattr(cls, "_vb_patched", False):
            return

        def _safe_call_connection_lost(self, exc):
            if self._called_connection_lost:
                return
            try:
                self._protocol.connection_lost(exc)
            finally:
                try:
                    if self._sock is not None and hasattr(self._sock, "shutdown") and self._sock.fileno() != -1:
                        self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    if self._sock is not None:
                        self._sock.close()
                except OSError:
                    pass
                self._sock = None
                server = self._server
                if server is not None:
                    server._detach()
                    self._server = None
                self._called_connection_lost = True

        cls._call_connection_lost = _safe_call_connection_lost
        cls._vb_patched = True
    except Exception:
        pass


_DEVICE_CACHE: str | None = None


def _cli_python(exe: str) -> str:
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        cand = exe[:-11] + "python.exe"
        if os.path.exists(cand):
            return cand
    return exe


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
    # Не передаём флаг OpenMP от родительского torch/numba дочерним CUDA-воркерам
    env.pop("KMP_INIT_AT_FORK", None)
    return env


def setup_cuda_dlls(update_path: bool = False) -> None:
    """На Windows регистрируем каталоги DLL CUDA/cuDNN из torch в поиске DLL,
    не загружая сам torch в процесс UI и не загрязняя PATH для дочерних воркеров."""
    configure_utf8_console()
    if os.name != "nt":
        return
    dirs: list[str] = []
    try:
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            dirs.append(os.path.join(os.path.dirname(spec.origin), "lib"))
    except Exception:
        pass
    site = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    dirs += glob.glob(os.path.join(site, "*", "bin"))
    cur_path = os.environ.get("PATH", "")
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            if update_path and d not in cur_path:
                cur_path = d + os.pathsep + cur_path
    if update_path:
        os.environ["PATH"] = cur_path


def get_device() -> str:
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE
    try:
        if "torch" in sys.modules:
            import torch

            _DEVICE_CACHE = "cuda" if torch.cuda.is_available() else "cpu"
            return _DEVICE_CACHE
        proc = subprocess.run(
            [_cli_python(sys.executable), "-c", "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
            env=_clean_env(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        lines = proc.stdout.strip().splitlines()
        if proc.returncode == 0 and lines and lines[-1] in ("cuda", "cpu"):
            _DEVICE_CACHE = lines[-1]
            return _DEVICE_CACHE
    except Exception:
        pass
    _DEVICE_CACHE = "cpu"
    return _DEVICE_CACHE


def free_gpu() -> None:
    gc.collect()
    try:
        if "torch" not in sys.modules:
            return
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_clean_env(),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command execution failed:\n" + " ".join(cmd) + "\n\n" + proc.stderr[-3000:]
        )
    return proc.stdout


def run_worker(cmd: list[str], cwd: str | None = None, progress=None) -> None:
    """Запускает тяжёлый этап (ASR / MT / TTS / Seed-VC) в изолированном подпроцессе с потоковым прогрессом."""
    if cmd:
        cmd = [_cli_python(cmd[0]), *cmd[1:]]
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=_clean_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        if line.startswith("__PROG__:"):
            parts = line.split(":", 2)
            if len(parts) == 3 and progress:
                try:
                    progress(float(parts[1]), parts[2])
                except ValueError:
                    pass
            continue
        if line:
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(
            f"Worker exited with code {code}:\n" + " ".join(cmd) + "\n\n" + "\n".join(tail[-25:])
        )


def probe(path: str) -> dict:
    out = run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(out)


def media_duration(path: str) -> float:
    return float(probe(path)["format"]["duration"])


def has_audio(path: str) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def extract_audio(src: str, dst: str, sr: int = 44100, channels: int = 2) -> str:
    run([FFMPEG, "-y", "-i", src, "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", dst])
    return dst


def load_wav(path: str, sr: int, mono: bool = True) -> np.ndarray:
    """Читает любой аудиофайл через ffmpeg с ресемплингом."""
    tmp = path + f".{sr}.tmp.wav"
    run([FFMPEG, "-y", "-i", path, "-ac", "1" if mono else "2", "-ar", str(sr), "-c:a", "pcm_f32le", tmp])
    data, _ = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    return data


def save_wav(path: str, data: np.ndarray, sr: int) -> str:
    sf.write(path, data, sr, subtype="PCM_16")
    return path


def normalize_audio(wav: np.ndarray, target_peak: float = 0.72, target_rms: float = 0.08) -> np.ndarray:
    """Нормализует уровень чистого референса: поднимает тихие видео без клиппинга пиков."""
    if wav.size == 0:
        return wav
    w = wav.astype(np.float32, copy=True)
    w -= float(np.mean(w))
    peak = float(np.max(np.abs(w)))
    cur_rms = float(np.sqrt(np.mean(np.square(w))))
    if peak < 1e-5 or cur_rms < 1e-6:
        return w
    gain = min(target_rms / cur_rms, target_peak / peak)
    return (w * float(gain)).astype(np.float32)


def compact_silence(wav: np.ndarray, sr: int, max_pause_sec: float = 0.08, threshold: float = 0.012) -> np.ndarray:
    """Сокращает длинные внутренние паузы внутри фразы с адаптивным порогом под громкость записи."""
    if wav.size < int(sr * 0.3):
        return wav
    peak = float(np.max(np.abs(wav)))
    if peak < 1e-5:
        return wav
    # Адаптивный RMS-порог защищает тихие слоги и согласные на записях с низким уровнем громкости
    eff_thr = min(threshold, max(peak * 0.045, 4e-4))
    win = max(int(0.02 * sr), 1)
    max_pause = max(int(max_pause_sec * sr), win)
    out = []
    silent_run = 0
    for i in range(0, len(wav), win):
        chunk = wav[i : i + win]
        chunk_rms = float(np.sqrt(np.mean(np.square(chunk))))
        if chunk_rms < eff_thr:
            silent_run += len(chunk)
            if silent_run <= max_pause:
                out.append(chunk)
        else:
            silent_run = 0
            out.append(chunk)
    return np.concatenate(out) if out else wav


def time_stretch(wav: np.ndarray, sr: int, ratio: float, tmp_dir: str) -> np.ndarray:
    """Ускоряет аудио в `ratio` раз без изменения высоты тона (ffmpeg atempo)."""
    if abs(ratio - 1.0) < 0.01:
        return wav
    if ratio > 1.02:
        target_len = wav.size / ratio
        wav = compact_silence(wav, sr, max_pause_sec=0.07)
        if wav.size <= target_len:
            return wav
        ratio = wav.size / max(target_len, 1.0)
        if abs(ratio - 1.0) < 0.01:
            return wav
    src = os.path.join(tmp_dir, "_stretch_in.wav")
    dst = os.path.join(tmp_dir, "_stretch_out.wav")
    sf.write(src, wav, sr, subtype="FLOAT")
    filters = [f"aresample=48000"]
    r = ratio
    while r > 2.0:  # atempo в старых сборках ограничен 0.5..2.0
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={r:.4f}")
    filters.append(f"aresample={sr}")
    run([FFMPEG, "-y", "-i", src, "-filter:a", ",".join(filters), "-c:a", "pcm_f32le", dst])
    out, _ = sf.read(dst, dtype="float32")
    return out


def trim_silence(wav: np.ndarray, sr: int, threshold: float = 0.01, pad: float = 0.05) -> np.ndarray:
    if wav.size == 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    if peak < 1e-5:
        return wav[:0]
    eff_thr = min(threshold, max(peak * 0.04, 4e-4))
    idx = np.where(np.abs(wav) > eff_thr)[0]
    if idx.size == 0:
        return wav[:0]
    p = int(pad * sr)
    return wav[max(idx[0] - p, 0): min(idx[-1] + p, len(wav))]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _ffmpeg_filter_path(path: str) -> str:
    """Экранирует путь для аргументов ffmpeg filter_complex (в т.ч. двоеточие диска на Windows)."""
    return os.path.abspath(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def video_dimensions(video: str) -> tuple[int, int]:
    """Возвращает (width, height) первого видеопотока через ffprobe."""
    try:
        for s in probe(video).get("streams", []):
            if s.get("codec_type") == "video":
                w, h = int(s.get("width") or 0), int(s.get("height") or 0)
                if w > 0 and h > 0:
                    return w, h
    except Exception:
        pass
    return 1280, 720


def mux_video(video: str, dub_wav: str, out_path: str, bg_wav: str | None = None,
              bg_volume: float = 0.0, keep_original_track: bool = False,
              subtitle_mode: str = "none", burn_ass: str | None = None,
              soft_srts: list[tuple[str, str, str]] | None = None,
              duration: float | None = None) -> str:
    """Собирает итоговое видео: дубляж + фон, опциональная оригинальная дорожка и субтитры."""
    cmd = [FFMPEG, "-y", "-i", video, "-i", dub_wav]
    next_inp = 2
    use_bg = bg_wav is not None and bg_volume > 0
    if use_bg:
        cmd += ["-i", bg_wav]
        bg_idx = next_inp
        next_inp += 1
        fc_parts = [
            f"[{bg_idx}:a]volume={bg_volume:.3f}[bg]",
            "[1:a]aresample=48000,volume=1.0[d]",
            "[bg]aresample=48000[bg2]",
            "[d][bg2]amix=inputs=2:duration=first:normalize=0[aout]",
        ]
    else:
        fc_parts = ["[1:a]aresample=48000[aout]"]

    burn_in = subtitle_mode in ("burn_tgt", "burn_src") and bool(burn_ass)
    if burn_in and burn_ass:
        fc_parts.append(f"[0:v:0]ass='{_ffmpeg_filter_path(burn_ass)}'[vout]")
        v_map = "[vout]"
    else:
        v_map = "0:v:0"

    soft_inputs: list[tuple[int, str, str]] = []
    if subtitle_mode == "soft" and soft_srts:
        for srt_path, lang3, title in soft_srts:
            if srt_path and os.path.isfile(srt_path):
                cmd += ["-i", srt_path]
                soft_inputs.append((next_inp, lang3 or "und", title or "Subtitles"))
                next_inp += 1

    cmd += ["-filter_complex", ";".join(fc_parts), "-map", v_map, "-map", "[aout]"]
    if keep_original_track:
        cmd += ["-map", "0:a:0?"]
    for s_idx, _, _ in soft_inputs:
        cmd += ["-map", f"{s_idx}:0"]

    if burn_in:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    else:
        cmd += ["-c:v", "copy"]

    cmd += ["-c:a", "aac", "-b:a", "192k"]
    if soft_inputs:
        cmd += ["-c:s", "mov_text", "-disposition:s:0", "default"]
        for out_s, (_, lang3, title) in enumerate(soft_inputs):
            cmd += [f"-metadata:s:s:{out_s}", f"language={lang3}", f"-metadata:s:s:{out_s}", f"title={title}"]
        # Не даём -shortest обрезать видео по концу последнего титра, если речь кончается раньше конца ролика
        if duration and duration > 0:
            cmd += ["-t", f"{duration:.3f}"]
    else:
        cmd += ["-shortest"]

    cmd += ["-metadata:s:a:0", "title=Dub", out_path]
    run(cmd)
    return out_path


def fmt_srt_time(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs_total = int(round(t * 100))
    h = cs_total // 360000
    m = (cs_total % 360000) // 6000
    s = (cs_total % 6000) // 100
    cs = cs_total % 100
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def write_srt(path: str, segments: list[dict], key: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{fmt_srt_time(seg['start'])} --> {fmt_srt_time(seg['end'])}\n{seg[key].strip()}\n\n")
    return path


def write_ass(path: str, segments: list[dict], key: str, width: int, height: int) -> str:
    """Создаёт ASS-субтитры с адаптивным масштабом под вертикальные (9:16) и горизонтальные (16:9) кадры."""
    portrait = height > width
    if portrait:
        play_x = 720
        play_y = max(int(round(720 * height / max(width, 1))), 720)
        font_size, margin_lr, margin_v, outline = 42, 34, 72, 3.0
    else:
        play_y = 720
        play_x = max(int(round(720 * width / max(height, 1))), 720)
        font_size, margin_lr, margin_v, outline = 40, 52, 38, 2.5

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_x}\n"
        f"PlayResY: {play_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline:.1f},1,2,{margin_lr},{margin_lr},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            txt = seg.get(key, "").strip().replace("{", "(").replace("}", ")").replace("\n", "\\N")
            if not txt:
                continue
            f.write(f"Dialogue: 0,{_fmt_ass_time(seg['start'])},{_fmt_ass_time(seg['end'])},Default,,0,0,0,,{txt}\n")
    return path
