from pathlib import Path
import json
from http.server import ThreadingHTTPServer
from threading import Thread
import unittest
from urllib.request import Request, urlopen

from app import Handler
from rag_core import ChunkSearchEngine


class SearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = ChunkSearchEngine(Path(__file__).parent / "data" / "chunks.jsonl")

    def test_dataset_shape(self) -> None:
        self.assertEqual(len(self.engine.chunks), 427)
        self.assertEqual(len(self.engine.business_functions), 6)

    def test_deposit_limit(self) -> None:
        results = self.engine.search(
            "예금은 얼마까지 보호되나요?",
            business_function="예금자보호제도",
            top_k=5,
        )
        self.assertTrue(results)
        self.assertTrue(
            any("1억원" in result.chunk["content"] for result in results),
            "상위 5개 결과에 보호한도 근거가 있어야 합니다.",
        )

    def test_business_filter_blocks_other_domains(self) -> None:
        results = self.engine.search(
            "신청에 필요한 서류와 절차",
            business_function="채무조정 안내",
            top_k=10,
        )
        self.assertTrue(results)
        self.assertTrue(
            all(
                result.chunk["business_function"] == "채무조정 안내"
                for result in results
            )
        )

    def test_web_api(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            with urlopen(f"{base_url}/api/summary", timeout=3) as response:
                summary = json.load(response)
            self.assertEqual(summary["summary"]["chunk_count"], 427)

            body = json.dumps(
                {
                    "question": "예금은 얼마까지 보호되나요?",
                    "business_function": "예금자보호제도",
                    "top_k": 3,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            request = Request(
                f"{base_url}/api/search",
                data=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urlopen(request, timeout=3) as response:
                result = json.load(response)
            self.assertEqual(result["candidate_count"], 133)
            self.assertEqual(len(result["results"]), 3)
            self.assertTrue(result["results"][0]["source_url"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
