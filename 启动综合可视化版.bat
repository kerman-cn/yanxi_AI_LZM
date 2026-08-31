@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "VENV_PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
set "APP_SCRIPT=%~dp0comprehensive_gui.py"

if not exist "%APP_SCRIPT%" goto missing_app
if not exist "%VENV_PYTHON%" goto broken_env

"%VENV_PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 goto broken_env

if /I "%~1"=="--check" goto check_ok
if exist "%VENV_PYTHONW%" goto launch_windowless
goto launch_console

:launch_windowless
start "" "%VENV_PYTHONW%" "%APP_SCRIPT%"
exit /b 0

:launch_console
start "" "%VENV_PYTHON%" "%APP_SCRIPT%"
exit /b 0

:check_ok
echo LAUNCHER_OK
exit /b 0

:missing_app
echo [ERROR] Missing comprehensive_gui.py in the project directory.
pause
exit /b 2

:broken_env
echo [ERROR] The project Python environment is missing or invalid.
echo Recreate it with Python 3.12, then install requirements.txt.
pause
exit /b 3
