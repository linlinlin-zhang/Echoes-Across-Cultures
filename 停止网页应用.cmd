@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Echo - stop
if not exist "%~dp0.venv-gpu\Scripts\python.exe" (
  echo 找不到 Echo Python 环境，请保留脚本在 Echo 项目目录中。
  pause
  exit /b 1
)
"%~dp0.venv-gpu\Scripts\python.exe" -X utf8 "%~dp0scripts\manual_intranet_web.py" stop %*
if errorlevel 1 (
  echo 操作未完成，请查看上面的错误信息。
  pause
  exit /b 1
)
exit /b 0
