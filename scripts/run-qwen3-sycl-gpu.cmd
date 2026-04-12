@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run-qwen3-sycl-gpu.ps1" %*
exit /b %ERRORLEVEL%
