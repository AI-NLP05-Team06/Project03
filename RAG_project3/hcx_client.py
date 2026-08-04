# [5/10] Colab Secret 또는 환경변수에서 HCX_API_KEY를 읽어 HCX(OpenAI 호환) 클라이언트를 생성합니다.
from config import *

try:
    from google.colab import userdata  # pyright: ignore[reportMissingImports]
except ImportError:
    userdata = None


def load_hcx_api_key() -> str:
    """Colab Secret 또는 환경 변수에서 HCX API 키를 읽고 검증합니다."""
    key = None

    if userdata is not None:
        try:
            key = userdata.get("HCX_API_KEY")
        except Exception:
            key = None

    if not key:
        key = os.environ.get("HCX_API_KEY")

    if not key:
        raise RuntimeError(
            "HCX API 키가 없습니다. Colab Secrets에 HCX_API_KEY를 "
            "등록한 뒤 이 셀을 다시 실행하세요."
        )

    key = str(key).strip()
    lines = [line.strip() for line in key.splitlines() if line.strip()]

    if len(lines) != 1:
        raise RuntimeError(
            "HCX_API_KEY는 줄바꿈 없는 API 키 하나여야 합니다."
        )

    key = lines[0]

    if key.lower().startswith("bearer "):
        raise RuntimeError(
            "HCX_API_KEY 앞에 'Bearer '를 붙이지 마세요."
        )

    if any(character.isspace() for character in key):
        raise RuntimeError(
            "HCX_API_KEY 안에 공백 또는 줄바꿈이 포함되어 있습니다."
        )

    return key


HCX_API_KEY = load_hcx_api_key()

hcx_client = OpenAI(
    api_key=HCX_API_KEY,
    base_url=HCX_BASE_URL,
    timeout=HCX_REQUEST_TIMEOUT_SECONDS,
    max_retries=HCX_MAX_RETRIES,
)

print("HCX 클라이언트 설정 완료")
print("- Base URL:", HCX_BASE_URL)
print("- Chat 모델:", HCX_CHAT_MODEL)
print("- 질문 임베딩 모델:", HCX_EMBEDDING_MODEL)
print("- API 키: 등록됨")
