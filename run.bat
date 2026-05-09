@echo off
setlocal

cd /d "%~dp0"
"..\.venv-torch\Scripts\python.exe" "run.py"

exit /b %ERRORLEVEL%