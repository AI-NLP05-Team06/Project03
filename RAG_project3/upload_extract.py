from config import *


def upload_kdic_output_zip() -> Path:
    """Colab 업로드 창에서 ZIP 파일 하나를 받아 로컬 경로로 저장합니다."""
    try:
        from google.colab import files
    except ImportError as error:
        raise RuntimeError(
            "이 셀은 Google Colab 업로드 기능을 사용합니다. "
            "로컬 Jupyter라면 ZIP 경로를 직접 지정하는 셀을 사용하세요."
        ) from error

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
    """기존 압축 해제 폴더를 지우고 ZIP 전체를 안전하게 해제합니다."""
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP 파일이 없습니다: {zip_path}")

    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP 손상 파일 발견: {bad_member}")
        archive.extractall(EXTRACT_ROOT)

    print("압축 해제 완료:", EXTRACT_ROOT)
    return EXTRACT_ROOT


KDIC_ZIP_PATH = upload_kdic_output_zip()
extract_kdic_output(KDIC_ZIP_PATH)
