"""EC2 entrypoint -- systemd points ExecStart at this file.

Reuses the SAME launch_kdic_service_from_notebook() helper the Colab launcher
uses (2026-08-23-kdic-colab-launcher.py); that function was already
environment-agnostic (importlib.util.spec_from_file_location + a plain
Mapping[str, Any] of pipeline globals), so nothing there needed to change.
Only two things differ from the Colab launcher: where the runtime globals
come from (import kdic_pipeline_engine, not a notebook kernel), and how the
server is kept alive (uvicorn.run() in the main thread, not Cloudflare
Tunnel + a background thread inside a notebook that's already running).

Run with:
    HCX_API_KEY=... ELASTICSEARCH_URL=http://<EC2-2-IP>:9200 \\
    KDIC_ADMIN_TOKEN=... KDIC_DATA_SOURCE=/opt/kdic/data/kdic_data.zip \\
    KDIC_RUNTIME_DIR=/opt/kdic/runtime \\
    python kdic_ec2_launcher.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"파일을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    # 1) Load the EC2-cleaned pipeline engine (real network/ES calls happen here)
    pipeline_module = _load_file("kdic_pipeline_engine", BASE_DIR / "kdic_pipeline_engine.py")

    # Public chat and administrator UI share one versioned guardrail manager.
    guardrail_module = _load_file("kdic_guardrails", BASE_DIR / "kdic_guardrails.py")
    pipeline_module.KDIC_GUARDRAIL_MANAGER = guardrail_module.build_guardrail_manager()

    # Only the three prompts used by the currently active answer routes are
    # versioned.  The pipeline reads active values at call time, while A/B tests
    # use a context-local override and never overwrite public-chat globals.
    prompt_module = _load_file("kdic_prompt_manager", BASE_DIR / "kdic_prompt_manager.py")
    pipeline_module.KDIC_PROMPT_MANAGER = prompt_module.build_prompt_manager(vars(pipeline_module))

    # 2) Reuse the existing, already environment-agnostic launcher helper
    launcher = _load_file(
        "kdic_colab_launcher", BASE_DIR / "2026-08-23-kdic-colab-launcher.py"
    )
    attached = launcher.launch_kdic_service_from_notebook(
        vars(pipeline_module),
        integration_dir=BASE_DIR,
        host=os.getenv("KDIC_API_HOST", "0.0.0.0"),
        port=int(os.getenv("KDIC_API_PORT", "8501")),
        admin_token=os.getenv("KDIC_ADMIN_TOKEN"),
    )
    print(f"KDIC pipeline attached: {attached['pipeline']}")

    # 관리자 파라미터테스트/평가/발행 라우트는 adapter가 아니라 원본 파이프라인
    # 모듈(hybrid_minmax_search, CHUNKS 등)이 필요해서 별도로 연결해준다.
    # kdic_es_publish.publish_approved_preview()가 발행 시점에 이 모듈의
    # CHUNKS/DENSE_MATRIX 등에 새 청크를 직접 append하므로, 서버 재시작이나
    # 별도 리로드 없이 발행 직후 검색에 바로 반영된다.
    attached["api_module"].set_kdic_raw_pipeline_module(pipeline_module)

    # Latest administrator UI and its staged-write/search-evaluation routes.
    # The public chatbot continues to use the same pipeline_module instance,
    # so an approved parameter/chunk change applies to the next chat request.
    admin_extension = _load_file(
        "kdic_admin_extension_aws", BASE_DIR / "kdic_admin_extension_aws.py"
    )
    admin_install = admin_extension.install_admin_routes(
        attached["api_module"],
        BASE_DIR / "kdic-admin-ui.html",
        vars(pipeline_module),
    )
    print(f"KDIC latest admin attached: {admin_install}")

    # 3) start_server_in_thread() above only starts a background thread --
    #    keep this process alive so systemd has something to supervise.
    import threading

    threading.Event().wait()


if __name__ == "__main__":
    main()
