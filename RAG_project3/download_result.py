# [10/10] 대화형 세션에서 마지막으로 성공한 질문의 결과 JSON을 다운로드(Colab)하거나 로컬 경로를 출력합니다.
from config import *
from interactive_chat import *

try:
    from google.colab import files

    if not rag_output:
        raise RuntimeError(
            "다운로드할 결과가 없습니다. 먼저 대화형 질문 셀에서 "
            "질문을 한 건 이상 성공적으로 실행하세요."
        )

    result_path = rag_output.get("result_path")
    if not result_path:
        raise RuntimeError("마지막 결과의 저장 경로가 없습니다.")

    files.download(result_path)
except ImportError:
    if rag_output and rag_output.get("result_path"):
        print(
            "로컬 Jupyter에서는 다음 파일을 직접 확인하세요:",
            rag_output["result_path"],
        )
    else:
        print("저장된 마지막 결과가 없습니다.")
