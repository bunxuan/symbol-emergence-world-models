@echo off
REM Windows wrapper to run the Python metadata script and forward all args
python "%~dp0\..\model\run_with_metadata.py" %*
