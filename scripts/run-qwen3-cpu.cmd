@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run-qwen3-cpu.ps1" %*
exit /b %ERRORLEVEL%
