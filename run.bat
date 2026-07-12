@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%..\.venv-torch\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
	echo [ERROR] Python not found: %PYTHON_EXE%
	exit /b 1
)

cd /d "%PROJECT_ROOT%"

if "%~1"=="" (
	call "scripts\run_all.bat"
	exit /b %ERRORLEVEL%
)

if /I "%~1"=="help" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage

"%PYTHON_EXE%" "run.py" %*
exit /b %ERRORLEVEL%

:usage
echo Usage: run.bat [world^|flow^|diffusion^|gridworld^|2d^|randomization^|random^|all]
echo.
echo Examples:
echo   run.bat
echo   run.bat all
echo   run.bat world
echo   run.bat flow
echo   run.bat diffusion
echo   run.bat gridworld
echo   run.bat 2d
echo   run.bat randomization
echo   run.bat random
exit /b 0