import copy
import math
import unittest
from pathlib import Path

from budget_router.core import BM25, fit, quality, read, rerank, route, summarize, validate


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.corpus = read(Path("data/corpus.json"))
        self.queries = read(Path("data/queries.json"))

    def test_split_integrity(self):
        docs, queries = validate(self.corpus, self.queries)
        self.assertEqual((len(docs), len(queries)), (24, 38))
        for mutate in ("group", "relevance", "id", "split", "question", "answers"):
            q = copy.deepcopy(self.queries)
            dev, evaluation = q["queries"][0], q["queries"][-1]
            evaluation[mutate] = {
                "group": dev["group"],
                "relevance": dev["relevance"],
                "id": dev["id"],
                "split": "test",
                "question": "",
                "answers": [3],
            }[mutate]
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate(self.corpus, q)

    def test_bad_document_and_labels(self):
        for change in ({"id": "archive"}, {"text": ""}, {"extra": 1}):
            c = copy.deepcopy(self.corpus)
            c["documents"][1].update(change)
            with self.assertRaises(ValueError):
                validate(c, self.queries)
        for relevance in ({"missing": 1}, {"archive": True}, {"archive": -1}):
            q = copy.deepcopy(self.queries)
            q["queries"][0]["relevance"] = relevance
            with self.assertRaises(ValueError):
                validate(self.corpus, q)

    def test_bm25_hand_calculation(self):
        engine = BM25(
            [
                {"id": "a", "title": "red", "text": "red"},
                {"id": "b", "title": "blue", "text": "blue"},
            ]
        )
        ranking = engine.retrieve("red")
        self.assertEqual(ranking[0][0], "a")
        self.assertAlmostEqual(ranking[0][1], math.log(2) * 2 * 2.2 / 3.2)
        self.assertEqual(engine.signals("red", ranking)["confidence"], 1)
        self.assertEqual(engine.retrieve("unknown"), [("a", 0), ("b", 0)])
        self.assertEqual(engine.signals("unknown", engine.retrieve("unknown"))["confidence"], 0)
        with self.assertRaises(ValueError):
            BM25([])

    def test_route_boundaries_and_all_budgets(self):
        p = {"budget": 12, "low": 0.25, "high": 0.5}
        for confidence, expected in ((0, 12), (0.25, 4), (0.49, 4), (0.5, 0), (1, 0)):
            self.assertEqual(route({"confidence": confidence}, p, 24), expected)
        for budget in range(25):
            for n in range(25):
                k = route({"confidence": 0}, p | {"budget": budget}, n)
                self.assertLessEqual(k, min(budget, n))
                self.assertIn(k, (0, 4, 12))
        for patch in ({"budget": -1}, {"budget": True}, {"low": 0.6}, {"high": float("nan")}):
            with self.assertRaises(ValueError):
                route({"confidence": 0}, p | patch, 24)
        with self.assertRaises(ValueError):
            route({"confidence": float("inf")}, p, 24)

    def test_rerank_prefix_tail_and_stable_ties(self):
        self.assertEqual(rerank(["a", "b", "c", "d"], [1, 3, 3]), ["b", "c", "a", "d"])
        self.assertEqual(rerank(["a"], []), ["a"])
        for scores in ([1, 2], [float("nan")]):
            with self.assertRaises(ValueError):
                rerank(["a"], scores)

    def test_metrics(self):
        self.assertEqual(quality(["a", "b"], {"a": 1}), {"ndcg3": 1, "mrr": 1})
        self.assertAlmostEqual(quality(["a", "b"], {"b": 1})["ndcg3"], 1 / math.log2(3))
        self.assertEqual(quality(["a"], {}), {"ndcg3": 0, "mrr": 0})
        with self.assertRaises(ValueError):
            quality(["a", "a"], {})

    def test_fit_uses_dev_and_penalty(self):
        row = {
            "split": "dev",
            "retrieval": list(range(12)),
            "signals": {"confidence": 0.2},
            "alternatives": {
                str(k): {"quality": {"ndcg3": q}} for k, q in ((0, 0), (4, 1), (12, 1))
            },
        }
        cfg = {"budgets": [12], "penalties": [0, 1], "thresholds": [0, 0.5, 1.01]}
        policies = fit([row], cfg)
        self.assertEqual(route(row["signals"], policies[0], 12), 4)
        self.assertEqual(route(row["signals"], policies[1], 12), 0)
        with self.assertRaises(ValueError):
            fit([row | {"split": "eval"}], cfg)

    def test_frontier(self):
        base = {"quality": {"ndcg3": 1, "mrr": 1}, "answerable": True, "router_ms": 0}
        rows = [
            base | {"policy": "a", "depth": 0, "elapsed_ms": 1},
            base | {"policy": "b", "depth": 4, "elapsed_ms": 2},
        ]
        self.assertEqual(summarize(rows)["frontier"], ["a"])
        self.assertEqual(summarize(rows)["policies"]["a"]["p95_ms"], 1)
