@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
rem Launches without a console window; if troubleshooting, run: .venv\Scripts\python.exe desktop.py
start "" .venv\Scripts\pythonw.exe desktop.py
