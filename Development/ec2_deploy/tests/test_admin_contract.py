from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "kdic_deploy_assets"


def load_extension():
    path = ASSETS / "kdic_admin_extension_aws.py"
    spec = importlib.util.spec_from_file_location("kdic_admin_extension_aws_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyJobs:
    def stats(self):
        return {"statuses": {}, "job_count": 0, "backend": "test"}


def main() -> None:
    os.environ["KDIC_ADMIN_TOKEN"] = "test-admin-token-with-safe-length"
    os.environ["HCX_API_KEY"] = "test-hcx-key-with-safe-length-only"
    app = FastAPI()
    service = SimpleNamespace(
        app=app,
        JOB_STORE=DummyJobs(),
        admin_summary=lambda: {
            "pipeline": {"configured": True, "name": "contract-test"},
            "elasticsearch": {"connected": True, "status": "green"},
        },
        admin_jobs=lambda limit: {"items": []},
        admin_indices=lambda: {"items": []},
        admin_capabilities=lambda: {"mode": "READ_ONLY", "features": [], "disabled_mutations": []},
    )
    runtime = {
        "DENSE_WEIGHT": 0.7,
        "BM25_WEIGHT": 0.3,
        "CANDIDATE_DEPTH": 20,
        "FINAL_TOP_K": 5,
        "QUERY_FUSION_RRF_K": 10,
        "PARENT_CHILD_ENABLED": True,
        "PARENT_CONTEXT_MAX_CHARS": 8192,
        "HCX_API_KEY": os.environ["HCX_API_KEY"],
    }
    extension = load_extension()
    installed = extension.install_admin_routes(service, ASSETS / "kdic-admin-ui.html", runtime)
    assert installed["mode"] == "STAGED_WRITE"

    with TestClient(app, base_url="https://testserver") as client:
        assert client.get("/api/admin-ui/summary").status_code == 401
        bad = client.post("/api/admin-ui/login", json={"admin_token": "wrong-token"})
        assert bad.status_code == 401
        login = client.post(
            "/api/admin-ui/login",
            json={"admin_token": os.environ["KDIC_ADMIN_TOKEN"]},
        )
        assert login.status_code == 200
        assert login.json()["authenticated"] is True
        assert "kdic_admin_session" in client.cookies
        summary = client.get("/api/admin-ui/summary")
        assert summary.status_code == 200
        assert summary.json()["admin_mode"] == "STAGED_WRITE"
        capabilities = client.get("/api/admin-ui/capabilities")
        assert capabilities.status_code == 200
        assert "evaluation_run" in capabilities.json()["features"]
        html = (ASSETS / "kdic-admin-ui.html").read_text(encoding="utf-8")
        assert "KDIC AWS ADMIN" in html
        assert "Colab 런타임" not in html
        assert "/api/admin-ui/login" in html

    print("AWS admin contract test: PASS")


if __name__ == "__main__":
    main()
