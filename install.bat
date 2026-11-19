@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo Python 3.10-3.11 is required. Creating virtual environment...
  py -3.11 -m venv .venv || python -m venv .venv
)
.venv\Scripts\python.exe -m pip install -U pip wheel setuptools
.venv\Scripts\python.exe -m pip install -r requirements.txt torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
echo.
echo Installation complete. Run the app with: run.bat or run_desktop.bat
pause
