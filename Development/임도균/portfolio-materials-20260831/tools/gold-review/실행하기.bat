@echo off
chcp 65001 > nul
cd /d "%~dp0"

where npm > nul 2>&1
if errorlevel 1 (
  echo.
  echo Node.js가 설치되어 있지 않습니다.
  echo https://nodejs.org 에서 LTS 버전을 설치한 뒤 다시 실행해 주세요.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo 필요한 파일을 처음 한 번 설치합니다...
  call npm install
  if errorlevel 1 (
    echo.
    echo 설치 중 문제가 발생했습니다.
    pause
    exit /b 1
  )
)

echo.
echo KDIC Gold 검수 도구를 실행합니다.
echo 브라우저가 열리지 않으면 http://localhost:3000 을 직접 여세요.
start "" "http://localhost:3000"
call npm run dev

if errorlevel 1 (
  echo.
  echo 실행 중 문제가 발생했습니다.
  pause
)
