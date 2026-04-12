@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0test-qwen3-determinism.ps1" %*
exit /b %ERRORLEVEL%
