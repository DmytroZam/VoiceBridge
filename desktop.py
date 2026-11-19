"""VoiceBridge — десктопная версия: тот же интерфейс, но в собственном окне (без браузера)."""
from __future__ import annotations

import inspect
import os

import webview  # pywebview: нативное окно на базе WebView2 (Edge) в Windows

import app  # настраивает окружение, UTF-8 и CUDA DLL при импорте
from dubber import pipeline


def main() -> None:
    os.makedirs(pipeline.OUTPUT_DIR, exist_ok=True)
    os.makedirs(pipeline.WORK_DIR, exist_ok=True)

    launch_kwargs = {"theme": app.build_theme(), "css": app.CSS} \
        if "css" in inspect.signature(app.gr.Blocks.launch).parameters else {}
    demo = app.build_ui().queue()
    # Сервер Gradio поднимается локально в фоне, окно показывает его содержимое
    _, url, _ = demo.launch(
        server_name="127.0.0.1", inbrowser=False, prevent_thread_lock=True, show_error=True,
        allowed_paths=[pipeline.OUTPUT_DIR, pipeline.WORK_DIR], **launch_kwargs,
    )

    webview.create_window("VoiceBridge — AI Video Dubbing", url, width=1280, height=860, min_size=(900, 640),
                          text_select=True)
    webview.start()  # блокируется до закрытия окна
    demo.close()


if __name__ == "__main__":
    main()
