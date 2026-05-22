@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%PROJECT_ROOT%\..") do set "WORKSPACE_ROOT=%%~fI"
set "PYTHON_EXE=%WORKSPACE_ROOT%\.venv-torch\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    exit /b 1
)

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "run.py" all
exit /b %ERRORLEVEL%
