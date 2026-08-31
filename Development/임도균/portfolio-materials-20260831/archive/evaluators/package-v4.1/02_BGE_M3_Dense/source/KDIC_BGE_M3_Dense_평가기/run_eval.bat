@echo off
chcp 65001 > nul
cd /d "%~dp0"

set "DATASET=C:\Users\임도균\Desktop\평가데이터셋_검색평가지표용.xlsx"
set "KDIC_ZIP=C:\Users\임도균\Downloads\KDIC_RAG_V4_7_INTERACTIVE_CHAT (2)\KDIC_output.zip"

python evaluate_bge_m3_dense.py ^
  --dataset "%DATASET%" ^
  --kdic-zip "%KDIC_ZIP%" ^
  --output-dir "%~dp0results"

if errorlevel 1 (
  echo.
  echo 평가 실행 중 문제가 발생했습니다. 위 오류 메시지를 확인해 주세요.
  pause
  exit /b 1
)

echo.
echo 평가가 완료되었습니다. results 폴더를 확인해 주세요.
pause
