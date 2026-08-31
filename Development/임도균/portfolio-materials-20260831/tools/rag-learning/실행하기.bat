@echo off
cd /d "%~dp0"
python app.py
if errorlevel 1 (
  echo.
  echo 실행 중 문제가 발생했습니다. Python이 설치되어 있는지 확인해 주세요.
  pause
)

