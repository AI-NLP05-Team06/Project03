from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"파일을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def launch_kdic_service_from_notebook(
    runtime_globals: Mapping[str, Any],
    *,
    integration_dir: str | Path = "/content",
    host: str = "0.0.0.0",
    port: int = 8501,
    admin_token: str | None = None,
) -> dict[str, Any]:
    """Attach the executed notebook runtime and start FastAPI in a daemon thread."""

    base = Path(integration_dir).expanduser().resolve()
    adapter_path = base / "2026-08-23-kdic-colab-runtime-adapter.py"
    service_path = base / "2026-08-23-kdic-fastapi-service.py"
    html_path = base / "2026-08-23-kdic-chat-ui.html"
    missing = [str(path) for path in (adapter_path, service_path, html_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("결합 파일이 누락되었습니다: " + " | ".join(missing))
    if admin_token:
        os.environ["KDIC_ADMIN_TOKEN"] = str(admin_token)

    adapter_module = _load_file("kdic_colab_runtime_adapter", adapter_path)
    service_module = _load_file("kdic_fastapi_service", service_path)
    pipeline = adapter_module.build_latest_kdic_pipeline(runtime_globals)
    service_module.set_kdic_pipeline(pipeline)
    server = service_module.start_server_in_thread(host=host, port=port)
    return {
        "pipeline": pipeline.name,
        "server": server,
        "api_module": service_module,
        "adapter": pipeline,
        "html": str(html_path),
        "admin_api_mode": "READ_ONLY",
    }

