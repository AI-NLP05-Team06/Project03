from __future__ import annotations

import importlib.util
import base64
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
        "CHUNKS_BY_ID": {"TEST_chunk_000": {"chunk_id": "TEST_chunk_000", "content": "테스트"}},
    }
    extension = load_extension()
    runtime["fuse_query_results"] = extension._clean
    policy = extension._validate_evaluation_policy(
        20,
        extension.DEFAULT_METRIC_KS,
        [1, 3, 5, 10, 20],
        {"candidate_depth": 20},
        {"candidate_depth": 20},
    )
    assert policy["evaluation_depth"] == 20
    measured = extension._metrics(
        ["MISS", "TEST_chunk_000"],
        ["TEST_chunk_000"],
        False,
        extension.DEFAULT_METRIC_KS,
        [1, 3, 5, 10, 20],
    )
    assert measured["hit_at_3"] == 1.0
    assert measured["curve"]["1"]["recall"] == 0.0
    assert measured["curve"]["3"]["recall"] == 1.0
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
        assert "pipeline_runtime_graph" in capabilities.json()["features"]
        graph = client.get("/api/admin-ui/pipeline/graph")
        assert graph.status_code == 200
        assert graph.json()["read_only"] is True
        assert graph.json()["fingerprint"]
        hybrid = next(node for node in graph.json()["nodes"] if node["id"] == "hybrid")
        assert hybrid["code_available"] is True
        source = client.get("/api/admin-ui/pipeline/nodes/hybrid/source")
        assert source.status_code == 200
        assert source.json()["read_only"] is True
        assert source.json()["source_hash"]
        assert "def _clean" in source.json()["source"]
        assert client.get("/api/admin-ui/pipeline/nodes/question/source").status_code == 404
        assert client.get("/api/admin-ui/evaluations/jobs").status_code == 200
        config_draft = client.put(
            "/api/admin-ui/draft/config",
            json={"values": {"dense_weight": 0.6, "bm25_weight": 0.4}},
        )
        assert config_draft.status_code == 200
        assert client.delete("/api/admin-ui/draft/config").status_code == 200
        added = client.post(
            "/api/admin-ui/draft/chunks",
            json={"chunk": {"chunk_id": "NEW_chunk_000", "content": "신규 테스트 청크"}},
        )
        assert added.status_code == 200
        assert client.delete("/api/admin-ui/draft/additions/NEW_chunk_000").status_code == 200
        csv_bytes = "question_id,question,gold_chunk_ids\nQ1,테스트 질문,TEST_chunk_000\n".encode("utf-8-sig")
        uploaded = client.post(
            "/api/admin-ui/evaluations/upload",
            json={"filename": "test.csv", "content_base64": base64.b64encode(csv_bytes).decode("ascii")},
        )
        assert uploaded.status_code == 200
        dataset_id = uploaded.json()["dataset_id"]
        assert client.delete(f"/api/admin-ui/evaluations/datasets/{dataset_id}").status_code == 200
        html = (ASSETS / "kdic-admin-ui.html").read_text(encoding="utf-8")
        assert "KDIC AWS ADMIN" in html
        assert "Colab 런타임" not in html
        assert "/api/admin-ui/login" in html
        assert "Recall@K 증가 곡선" in html
        assert "평가 검색 깊이" in html
        assert "PIPELINE STUDIO" in html

    print("AWS admin contract test: PASS")


if __name__ == "__main__":
    main()
