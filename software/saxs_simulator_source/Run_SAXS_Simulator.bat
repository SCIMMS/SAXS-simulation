@echo off
setlocal
cd /d "%~dp0"

echo Starting SAXS Simulator...
echo.

REM Use the same "python" command that works in PowerShell first.
REM If that is unavailable, fall back to the Windows Python launcher "py -3".
where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python simulator.py
) else (
    where py >nul 2>nul
    if %ERRORLEVEL%==0 (
        py -3 simulator.py
    ) else (
        echo Python was not found.
        echo Please install Python or add it to PATH.
        echo.
        pause
        exit /b 1
    )
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo SAXS Simulator could not start.
    echo.
    echo The launcher used the Python command available from this folder.
    echo If the program works from PowerShell, check which Python is used there:
    echo.
    echo     where python
    echo     python -m pip show matplotlib
    echo.
    echo If matplotlib is missing in this Python, install it with:
    echo.
    echo     python -m pip install matplotlib numpy scipy pillow pyFAI
    echo.
    pause
)
