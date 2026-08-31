from __future__ import annotations

import unittest

from metrics import (
    average_precision_at_k,
    complete_at_k,
    evaluate_ranking,
    ndcg_at_k,
)


class MetricTests(unittest.TestCase):
    def test_perfect_single_gold(self) -> None:
        result = evaluate_ranking(
            ["A", "X", "Y"],
            gold_ids=["A"],
            primary_gold_ids=["A"],
            supporting_gold_ids=[],
            multi_chunk_required=False,
        )
        self.assertEqual(result["hit_at_3"], 1.0)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["mrr_at_10"], 1.0)
        self.assertEqual(result["ap_at_10"], 1.0)
        self.assertEqual(result["precision_at_5"], 0.2)
        self.assertIsNone(result["complete_at_5"])

    def test_average_precision_multiple_gold(self) -> None:
        self.assertAlmostEqual(
            average_precision_at_k(["A", "X", "B"], ["A", "B"], 10),
            (1.0 + 2 / 3) / 2,
        )

    def test_complete_multi_chunk(self) -> None:
        self.assertEqual(
            complete_at_k(
                ["A", "X", "B"],
                ["A", "B"],
                5,
                multi_chunk_required=True,
            ),
            1.0,
        )
        self.assertEqual(
            complete_at_k(
                ["A", "X"],
                ["A", "B"],
                5,
                multi_chunk_required=True,
            ),
            0.0,
        )

    def test_ndcg_prefers_primary_first(self) -> None:
        ideal = ndcg_at_k(["P", "S"], ["P"], ["S"], 5)
        reversed_score = ndcg_at_k(["S", "P"], ["P"], ["S"], 5)
        self.assertEqual(ideal, 1.0)
        self.assertLess(reversed_score, ideal)


if __name__ == "__main__":
    unittest.main()
