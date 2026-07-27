# [2/10] KDIC_output.zip을 업로드(Colab)하고 압축을 해제해 EXTRACT_ROOT에 풀어둡니다.
from config import *


def resolve_local_kdic_output() -> Path:
    """로컬 환경: KDIC_ZIP_PATH 환경변수(zip 파일 또는 jsonl 3종이 든 폴더)를 찾습니다."""
    local_path = Path(
        os.environ.get("KDIC_ZIP_PATH", "data")
    ).resolve()

    if not local_path.exists():
        raise FileNotFoundError(
            f"KDIC 데이터를 찾을 수 없습니다: {local_path}\n"
            "KDIC_ZIP_PATH 환경변수로 zip 파일 또는 jsonl 3종(documents.jsonl, "
            "chunks.jsonl, chunk_embeddings_hcx.jsonl)이 든 폴더 경로를 지정하세요."
        )

    print("로컬 데이터 사용:", local_path)
    return local_path


def upload_kdic_output_zip() -> Path:
    """Colab 업로드 창에서 ZIP 파일 하나를 받아 로컬 경로로 저장합니다."""
    try:
        from google.colab import files
    except ImportError:
        return resolve_local_kdic_output()

    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.lower().endswith(".zip")]

    if len(zip_names) != 1:
        raise RuntimeError(
            "ZIP 파일을 정확히 1개 업로드해야 합니다. "
            f"현재 ZIP 수={len(zip_names)}"
        )

    uploaded_name = zip_names[0]
    destination = WORK_ROOT / "KDIC_output_uploaded.zip"
    destination.write_bytes(uploaded[uploaded_name])

    print("업로드 완료:", uploaded_name)
    print("저장 경로:", destination)
    return destination


def extract_kdic_output(zip_path: Path) -> Path:
    """기존 압축 해제 폴더를 지우고, ZIP이면 해제하고 폴더면 그대로 복사해둡니다."""
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP/폴더 경로가 없습니다: {zip_path}")

    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    if zip_path.is_dir():
        shutil.copytree(zip_path, EXTRACT_ROOT, dirs_exist_ok=True)
        print("폴더 복사 완료:", EXTRACT_ROOT)
        return EXTRACT_ROOT

    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP 손상 파일 발견: {bad_member}")
        archive.extractall(EXTRACT_ROOT)

    print("압축 해제 완료:", EXTRACT_ROOT)
    return EXTRACT_ROOT


KDIC_ZIP_PATH = upload_kdic_output_zip()
extract_kdic_output(KDIC_ZIP_PATH)
