$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Starting SAXS Simulator..." -ForegroundColor Cyan

# Use the same "python" command that normally works in PowerShell first.
if (Get-Command python -ErrorAction SilentlyContinue) {
    python simulator.py
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 simulator.py
}
else {
    Write-Host ""
    Write-Host "Python was not found. Please install Python or add it to PATH." -ForegroundColor Red
    Read-Host "Press Enter to close"
}
