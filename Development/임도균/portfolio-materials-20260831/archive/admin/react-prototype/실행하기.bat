@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist "node_modules" (
  echo 필요한 패키지를 처음 한 번 설치합니다.
  call npm install
  if errorlevel 1 goto :error
)

echo.
echo KDIC RAG 관리자 프로토타입을 실행합니다.
echo 브라우저 주소: http://localhost:3000
echo 종료하려면 이 창에서 Ctrl+C를 누르세요.
echo.
call npm run dev
if errorlevel 1 goto :error
goto :eof

:error
echo.
echo 실행 중 문제가 발생했습니다. Node.js 설치 여부를 확인해 주세요.
pause
