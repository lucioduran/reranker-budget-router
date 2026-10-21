"""Deterministic retrieval, routing and evaluation primitives."""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate(corpus: dict, queries: dict) -> tuple[list[dict], list[dict]]:
    if corpus.get("schema_version") != 1 or queries.get("schema_version") != 1:
        raise ValueError("Unsupported schema version")
    docs, questions = corpus["documents"], queries["queries"]
    if not docs or not questions:
        raise ValueError("Corpus and queries must be nonempty")
    ids: set[str] = set()
    for doc in docs:
        if set(doc) != {"id", "title", "text"} or any(
            not isinstance(v, str) or not v.strip() for v in doc.values()
        ):
            raise ValueError("Invalid document")
        if doc["id"] in ids or not tokens(doc["title"] + " " + doc["text"]):
            raise ValueError("Duplicate or empty document")
        ids.add(doc["id"])
    seen: set[str] = set()
    groups: dict[str, set[str]] = {"dev": set(), "eval": set()}
    relevant: dict[str, set[str]] = {"dev": set(), "eval": set()}
    for q in questions:
        if set(q) != {"id", "split", "group", "question", "answers", "relevance"}:
            raise ValueError("Invalid query fields")
        if any(not isinstance(q[k], str) or not q[k].strip() for k in ("id", "group", "question")):
            raise ValueError("Invalid query text")
        if q["id"] in seen or q["split"] not in groups or not tokens(q["question"]):
            raise ValueError("Duplicate query, invalid split or empty question")
        if not isinstance(q["answers"], list) or not all(
            isinstance(a, str) and a.strip() for a in q["answers"]
        ):
            raise ValueError("Invalid answers")
        if not isinstance(q["relevance"], dict) or any(
            d not in ids or type(g) is not int or g not in (1, 2, 3)
            for d, g in q["relevance"].items()
        ):
            raise ValueError("Invalid relevance labels")
        seen.add(q["id"])
        groups[q["split"]].add(q["group"])
        relevant[q["split"]].update(q["relevance"])
    if (
        not all(groups.values())
        or groups["dev"] & groups["eval"]
        or (relevant["dev"] & relevant["eval"])
    ):
        raise ValueError("Splits must be nonempty with disjoint groups and relevant documents")
    return docs, questions


class BM25:
    def __init__(self, documents: list[dict]):
        if not documents:
            raise ValueError("Empty corpus")
        self.docs = {d["id"]: d for d in documents}
        if len(self.docs) != len(documents):
            raise ValueError("Duplicate document IDs")
        self.counts = {d["id"]: Counter(tokens(d["title"] + " " + d["text"])) for d in documents}
        self.lengths = {d: sum(c.values()) for d, c in self.counts.items()}
        self.average = mean(self.lengths.values())
        if not self.average:
            raise ValueError("Empty token corpus")
        df: Counter = Counter()
        for c in self.counts.values():
            df.update(c.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - count + 0.5) / (count + 0.5)) for t, count in df.items()}

    def retrieve(self, question: str) -> list[tuple[str, float]]:
        terms = sorted(set(tokens(question)))
        scores = []
        for doc, c in self.counts.items():
            norm = 1.2 * (0.25 + 0.75 * self.lengths[doc] / self.average)
            score = sum(self.idf.get(t, 0) * c[t] * 2.2 / (c[t] + norm) for t in terms)
            scores.append((doc, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))

    def signals(self, question: str, ranking: list[tuple[str, float]]) -> dict[str, float]:
        top = ranking[0][1]
        second = ranking[1][1] if len(ranking) > 1 else 0.0
        gap = (top - second) / top if top else 0.0
        terms = set(tokens(question))
        coverage = len(terms & self.counts[ranking[0][0]].keys()) / max(1, len(terms))
        return {"gap": gap, "coverage": coverage, "confidence": min(gap, coverage)}


def route(signals: dict[str, float], policy: dict, candidates: int) -> int:
    budget = policy["budget"]
    low, high = policy["low"], policy["high"]
    if type(budget) is not int or budget < 0 or type(candidates) is not int or candidates < 0:
        raise ValueError("Budget and candidate count must be nonnegative integers")
    if not finite(low) or not finite(high) or not 0 <= low <= high <= 1.01:
        raise ValueError("Invalid thresholds")
    confidence = signals["confidence"]
    if not finite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    desired = 0 if confidence >= high else 4 if confidence >= low else 12
    return max(k for k in (0, 4, 12) if k <= min(desired, budget, candidates))


def rerank(ranking: list[str], logits: list[float]) -> list[str]:
    if len(logits) > len(ranking) or not all(finite(s) for s in logits):
        raise ValueError("Invalid reranker logits")
    prefix = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    return [ranking[i] for i in prefix] + ranking[len(logits) :]


def quality(ranking: list[str], relevance: dict[str, int]) -> dict[str, float]:
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate ranked document")
    dcg = sum((2 ** relevance.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:3]))
    ideal = sum(
        (2**g - 1) / math.log2(i + 2)
        for i, g in enumerate(sorted(relevance.values(), reverse=True)[:3])
    )
    rr = next((1 / (i + 1) for i, d in enumerate(ranking) if d in relevance), 0.0)
    return {"ndcg3": dcg / ideal if ideal else 0.0, "mrr": rr}


def fit(rows: list[dict], config: dict) -> list[dict]:
    if not rows or any(r["split"] != "dev" for r in rows):
        raise ValueError("Fit requires development-only evidence")
    policies = []
    for budget in config["budgets"]:
        for penalty in config["penalties"]:
            choices = []
            for low in config["thresholds"]:
                for high in config["thresholds"]:
                    if low > high:
                        continue
                    p = {"budget": budget, "low": low, "high": high}
                    depths = [route(r["signals"], p, len(r["retrieval"])) for r in rows]
                    score = mean(
                        r["alternatives"][str(k)]["quality"]["ndcg3"]
                        for r, k in zip(rows, depths, strict=True)
                    )
                    cost = mean(depths)
                    choices.append((-(score - penalty * cost), cost, low, high))
            _, _, low, high = min(choices)
            policies.append(
                {
                    "name": f"router-b{budget}-p{penalty:g}",
                    "budget": budget,
                    "penalty": penalty,
                    "low": low,
                    "high": high,
                }
            )
    return policies


def summarize(traces: list[dict]) -> dict:
    if not traces:
        raise ValueError("Empty trace")
    result = {}
    for name in sorted({r["policy"] for r in traces}):
        rows = [r for r in traces if r["policy"] == name]
        latencies = sorted(r["elapsed_ms"] for r in rows)
        answerable = [r for r in rows if r["answerable"]]
        result[name] = {
            "ndcg3": mean(r["quality"]["ndcg3"] for r in rows),
            "answerable_ndcg3": mean(r["quality"]["ndcg3"] for r in answerable)
            if answerable
            else 0.0,
            "mrr": mean(r["quality"]["mrr"] for r in rows),
            "mean_pairs": mean(r["depth"] for r in rows),
            "mean_ms": mean(latencies),
            "p95_ms": latencies[math.ceil(0.95 * len(rows)) - 1],
            "mean_router_ms": mean(r["router_ms"] for r in rows),
            "samples": len(rows),
        }
    frontier = []
    for name, row in result.items():
        dominated = any(
            other["ndcg3"] >= row["ndcg3"]
            and other["mean_pairs"] <= row["mean_pairs"]
            and other["mean_ms"] <= row["mean_ms"]
            and (
                other["ndcg3"] > row["ndcg3"]
                or other["mean_pairs"] < row["mean_pairs"]
                or other["mean_ms"] < row["mean_ms"]
            )
            for other in result.values()
        )
        if not dominated:
            frontier.append(name)
    return {"policies": result, "frontier": frontier}
