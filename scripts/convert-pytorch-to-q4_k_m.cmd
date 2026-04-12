@echo off
setlocal
set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%convert-pytorch-to-q4_k_m.ps1" %*
exit /b %errorlevel%
