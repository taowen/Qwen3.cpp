@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0test-qwen3-sycl-gpu.ps1" %*
exit /b %ERRORLEVEL%
