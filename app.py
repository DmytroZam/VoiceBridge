"""VoiceBridge — дубляж видео с сохранением голоса. Интерфейс на Gradio."""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from dubber.utils import configure_utf8_console, patch_windows_asyncio, setup_cuda_dlls  # noqa: E402

configure_utf8_console()
patch_windows_asyncio()
setup_cuda_dlls()

import gradio as gr  # noqa: E402

from dubber import pipeline  # noqa: E402
from dubber.languages import SOURCE_CHOICES, TARGET_LANGUAGES  # noqa: E402
from dubber.utils import get_device  # noqa: E402

WHISPER_MODEL = "large-v3"
MT_MODEL = "facebook/nllb-200-distilled-1.3B"
BG_MODES = {
    "Keep music & ambient (Demucs)": "separate",
    "Original audio ducked (Voice-over)": "original",
    "Dubbed voice only": "none",
}
SUBTITLE_MODES = {
    "Off (SRT files only)": "none",
    "Burned-in: Translation": "burn_tgt",
    "Burned-in: Original": "burn_src",
    "Embedded track (soft SRT)": "soft",
}
HEADERS = ["Start, s", "End, s", "Original", "Translation"]

CSS = """
:root { --vb-grad: linear-gradient(135deg, #7c5cff 0%, #c64bff 45%, #ff5c8a 100%); }
.gradio-container { max-width: 1400px !important; margin: auto; }
body, .gradio-container { background: radial-gradient(1200px 600px at 10% -10%, #2a1d5c55, transparent),
  radial-gradient(900px 500px at 110% 10%, #5c1d4a44, transparent), #0b0b14 !important; }
#hero { padding: 28px 32px; border-radius: 22px; background: linear-gradient(135deg,#1a1433cc,#1b0f24cc);
  border: 1px solid #ffffff14; box-shadow: 0 20px 60px #0008, inset 0 1px 0 #ffffff10; margin-bottom: 8px;
  position: relative; overflow: hidden; }
#hero:before { content:""; position:absolute; inset:-40%; background: conic-gradient(from 0deg,#7c5cff22,#ff5c8a22,#7c5cff22);
  animation: spin 18s linear infinite; z-index:0; }
#hero > * { position: relative; z-index: 1; }
@keyframes spin { to { transform: rotate(360deg); } }
#hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0; background: var(--vb-grad);
  -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: -0.02em; }
#hero p { color: #c9c4e6; margin: 6px 0 0; font-size: 1.02rem; }
.vb-steps { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
.vb-step { padding:6px 12px; border-radius:999px; background:#ffffff0d; border:1px solid #ffffff18;
  color:#e7e3ff; font-size:.85rem; transition: all .25s ease; }
.vb-step:hover { background:#7c5cff33; transform: translateY(-2px); }
.vb-card { border-radius: 18px !important; border: 1px solid #ffffff12 !important;
  background: #14121f !important; }
ul.options { z-index: 9999 !important; }
.vb-primary { background: var(--vb-grad) !important; border: none !important; color: white !important;
  font-weight: 700 !important; box-shadow: 0 8px 24px #c64bff40; transition: transform .15s ease, box-shadow .2s ease; }
.vb-primary:hover { transform: translateY(-2px); box-shadow: 0 12px 32px #c64bff66; }
#log textarea { font-family: 'JetBrains Mono', Consolas, monospace !important; font-size: .82rem !important; }
footer { display: none !important; }
"""

HERO = f"""
<div id="hero">
  <h1>🎙️ VoiceBridge</h1>
  <p>Translates video into another language and dubs it in the <b>original speaker's cloned voice</b>. Fully local and open-source.</p>
  <div class="vb-steps">
    <span class="vb-step">① Whisper — speech → text</span>
    <span class="vb-step">② NLLB-200 — translation</span>
    <span class="vb-step">③ Silero v5 / XTTS v2 + Seed-VC — voice cloning</span>
    <span class="vb-step">④ Demucs — background preservation</span>
    <span class="vb-step">⚡ {get_device().upper()}</span>
  </div>
</div>
"""


def _rows(job) -> list[list]:
    return [[round(s["start"], 2), round(s["end"], 2), s["text"], s.get("translation", "")] for s in job.segments]


def _apply_table(job, table):
    """Переносит правки пользователя из таблицы в задание."""
    if table is None:
        return
    rows = table.values.tolist() if hasattr(table, "values") else table
    segs = []
    for r in rows:
        if len(r) < 4 or r[3] is None or str(r[3]).strip() == "":
            continue
        try:
            start, end = float(r[0]), float(r[1])
        except (TypeError, ValueError):
            continue
        segs.append({"start": start, "end": max(end, start + 0.1), "text": str(r[2] or ""),
                     "translation": str(r[3]).strip()})
    segs.sort(key=lambda s: s["start"])
    job.segments = segs


def _progress_fn(progress: gr.Progress):
    return lambda p, msg="": progress(p, desc=msg)


def step_analyze(video, url, src, tgt, bg_label, progress=gr.Progress()):
    url = (url or "").strip()
    if not video and not url:
        raise gr.Error("Please upload a video or paste a URL")
    if url:
        try:
            job = pipeline.new_job_from_url(url, _progress_fn(progress))
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(f"Failed to download video: {e}")
    else:
        job = pipeline.new_job(video)
    try:
        use_demucs = BG_MODES[bg_label] == "separate"
        pipeline.analyze(job, SOURCE_CHOICES[src], tgt, WHISPER_MODEL, MT_MODEL, use_demucs, _progress_fn(progress))
    except Exception as e:
        traceback.print_exc()
        err_msg = str(e) or type(e).__name__
        job.say(f"ERROR: {err_msg}")
        raise gr.Error(f"Transcription/translation error: {err_msg}")
    return job, gr.update(value=_rows(job)), "\n".join(job.log), gr.update(interactive=True)


def step_synthesize(job, table, tgt, bg_label, bg_vol, max_speed, per_seg, temp, keep_orig, emotion=80,
                    deep_clone=True, subtitle_label="Off (SRT files only)", progress=gr.Progress()):
    if job is None:
        raise gr.Error("Please run Step 1 first — Transcribe & Translate")
    _apply_table(job, table)
    try:
        video, files = pipeline.synthesize(job, tgt, BG_MODES[bg_label], bg_vol / 100.0, max_speed,
                                           per_seg, temp, keep_orig, (emotion or 0) / 100.0,
                                           _progress_fn(progress), deep_voice_clone=bool(deep_clone),
                                           subtitle_mode=SUBTITLE_MODES.get(subtitle_label, "none"))
    except Exception as e:
        traceback.print_exc()
        err_msg = str(e) or type(e).__name__
        job.say(f"ERROR: {err_msg}")
        raise gr.Error(f"Dubbing error: {err_msg}")
    return video, files, "\n".join(job.log)


def step_all(video, url, src, tgt, bg_label, bg_vol, max_speed, per_seg, temp, keep_orig, emotion=80,
             deep_clone=True, subtitle_label="Off (SRT files only)", progress=gr.Progress()):
    job, table, log, btn = step_analyze(video, url, src, tgt, bg_label, progress)
    if not job.segments:
        raise gr.Error("No speech found in the video to translate.")
    out, files, log = step_synthesize(job, None, tgt, bg_label, bg_vol, max_speed, per_seg, temp, keep_orig,
                                      emotion, deep_clone, subtitle_label, progress)
    return job, table, out, files, log, btn


def build_theme():
    return gr.themes.Soft(
        primary_hue="violet", secondary_hue="pink", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(body_background_fill_dark="#0b0b14", block_background_fill_dark="#14121f",
          block_border_color_dark="#ffffff12", input_background_fill_dark="#1b1830")


def build_ui() -> gr.Blocks:
    import inspect

    style = {"theme": build_theme(), "css": CSS}
    blocks_kwargs = style if "css" in inspect.signature(gr.Blocks.__init__).parameters else {}
    with gr.Blocks(title="VoiceBridge — AI Video Dubbing with Voice Cloning",
                   js="() => document.body.classList.add('dark')", **blocks_kwargs) as demo:
        job_state = gr.State(None)
        gr.HTML(HERO)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="vb-card"):
                video_in = gr.Video(label="Source Video", sources=["upload"], elem_id="video-input")
                url_in = gr.Textbox(label="…or video URL (YouTube, TikTok, etc.)",
                                    placeholder="https://www.youtube.com/watch?v=… or https://www.tiktok.com/@…",
                                    elem_id="video-url")
                with gr.Row():
                    src_lang = gr.Dropdown(list(SOURCE_CHOICES), value="Auto-detect", label="Source Language",
                                           elem_id="source-lang")
                    tgt_lang = gr.Dropdown(list(TARGET_LANGUAGES), value="Русский", label="Translate to",
                                           elem_id="target-lang")
                bg_mode = gr.Radio(list(BG_MODES), value=list(BG_MODES)[0], label="Background Audio", elem_id="bg-mode")
                subtitle_mode = gr.Dropdown(
                    list(SUBTITLE_MODES),
                    value=list(SUBTITLE_MODES)[0],
                    label="Subtitles in Video",
                    info="Burned-in displays everywhere; Embedded track can be toggled in desktop players.",
                    elem_id="subtitle-mode",
                )
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    bg_volume = gr.Slider(0, 100, value=60, step=5, label="Background Volume, %", elem_id="bg-volume")
                    max_speed = gr.Slider(1.0, 2.0, value=1.4, step=0.05,
                                          label="Max Speech Speed-up (to fit timing)", elem_id="max-speed")
                    temperature = gr.Slider(0.1, 1.0, value=0.75, step=0.05,
                                            label="XTTS Temperature (0.75 — natural expressive voice, lower — stricter)",
                                            elem_id="temperature")
                    emotion = gr.Slider(0, 100, value=80, step=5,
                                        label="Preserve Intonation & Emotion, % (volume, pitch, shouting)",
                                        elem_id="emotion")
                    per_segment = gr.Checkbox(False, label="Clone voice per segment individually "
                                                           "(for multi-speaker videos)",
                                              elem_id="per-segment")
                    deep_clone = gr.Checkbox(True, label="Deep Voice Cloning (Seed-VC timbre refinement — "
                                                         "much closer voice match)",
                                             elem_id="deep-clone")
                    keep_original = gr.Checkbox(False, label="Keep original audio as a second audio track",
                                                elem_id="keep-original")
                with gr.Row():
                    btn_analyze = gr.Button("① Transcribe & Translate", variant="secondary", elem_id="btn-analyze")
                    btn_all = gr.Button("⚡ Run All", elem_classes="vb-primary", elem_id="btn-all")

            with gr.Column(scale=7, elem_classes="vb-card"):
                table = gr.Dataframe(headers=HEADERS, datatype=["number", "number", "str", "str"],
                                     interactive=True, wrap=True, label="Transcript & Translation (editable)",
                                     column_widths=["9%", "9%", "41%", "41%"], max_height=420, elem_id="segments")
                btn_synth = gr.Button("② Dub with Original Voice", elem_classes="vb-primary",
                                      interactive=False, elem_id="btn-synth")
                video_out = gr.Video(label="Result", interactive=False, elem_id="video-output")
                files_out = gr.File(label="Output Files (video, audio, subtitles)", file_count="multiple",
                                    elem_id="files-output")

        with gr.Accordion("📜 Log", open=False):
            log = gr.Textbox(lines=12, max_lines=30, show_label=False, elem_id="log", autoscroll=True)

        def _fix_target(src, tgt):
            # Если язык оригинала совпал с языком перевода — переключаем перевод на другой язык
            src_code = SOURCE_CHOICES.get(src)
            if src_code and src_code == TARGET_LANGUAGES[tgt][0]:
                return gr.update(value="English" if src_code != "en" else "Русский")
            return gr.update()

        src_lang.change(_fix_target, [src_lang, tgt_lang], tgt_lang)

        btn_analyze.click(step_analyze, [video_in, url_in, src_lang, tgt_lang, bg_mode],
                          [job_state, table, log, btn_synth])
        btn_synth.click(step_synthesize, [job_state, table, tgt_lang, bg_mode, bg_volume, max_speed,
                                          per_segment, temperature, keep_original, emotion, deep_clone, subtitle_mode],
                        [video_out, files_out, log])
        btn_all.click(step_all, [video_in, url_in, src_lang, tgt_lang, bg_mode, bg_volume,
                                 max_speed, per_segment, temperature, keep_original, emotion, deep_clone, subtitle_mode],
                      [job_state, table, video_out, files_out, log, btn_synth])
    return demo


if __name__ == "__main__":
    os.makedirs(pipeline.OUTPUT_DIR, exist_ok=True)
    os.makedirs(pipeline.WORK_DIR, exist_ok=True)
    import inspect

    launch_kwargs = {"theme": build_theme(), "css": CSS} \
        if "css" in inspect.signature(gr.Blocks.launch).parameters else {}
    build_ui().queue().launch(inbrowser=True, show_error=True,
                              allowed_paths=[pipeline.OUTPUT_DIR, pipeline.WORK_DIR], **launch_kwargs)
