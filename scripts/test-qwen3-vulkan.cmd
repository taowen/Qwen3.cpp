@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0test-qwen3-vulkan.ps1" %*
exit /b %ERRORLEVEL%
