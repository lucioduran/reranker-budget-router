"""Local model execution and replayable experiment evidence."""

import importlib.metadata
import os
import platform
import subprocess
import time
from pathlib import Path

from .core import (
    BM25,
    digest,
    finite,
    fit,
    quality,
    read,
    rerank,
    route,
    summarize,
    validate,
    write,
)


def configuration(path: Path) -> dict:
    c = read(path)
    if (
        c["depths"] != [0, 4, 12]
        or c["budgets"] != [4, 12]
        or c["penalties"] != [0.0, 0.01, 0.04]
        or c["thresholds"] != [0.0, 0.1, 0.25, 0.5, 1.01]
        or type(c["threads"]) is not int
        or not 1 <= c["threads"] <= 4
        or type(c["repetitions"]) is not int
        or c["repetitions"] < 1
        or c["max_length"] != 256
        or len(c["revision"]) != 40
        or any(ch not in "0123456789abcdef" for ch in c["revision"])
    ):
        raise ValueError("Invalid experiment configuration")
    return c


def download(config: dict, cache: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        config["model_id"],
        revision=config["revision"],
        cache_dir=str(cache),
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "README.md",
        ],
    )


class LocalReranker:
    def __init__(self, config: dict, cache: Path):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.limit = config["max_length"]
        args = {
            "revision": config["revision"],
            "cache_dir": str(cache),
            "local_files_only": True,
            "trust_remote_code": False,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **args)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                config["model_id"], use_safetensors=True, **args
            )
            .cpu()
            .eval()
        )

    def score(self, question: str, docs: list[dict]) -> list[float]:
        if not docs:
            return []
        encoded = self.tokenizer(
            [question] * len(docs),
            [d["title"] + " " + d["text"] for d in docs],
            padding=True,
            truncation=True,
            max_length=self.limit,
            return_tensors="pt",
        )
        with self.torch.inference_mode():
            return self.model(**encoded).logits.flatten().tolist()


def policies_for(bundle: dict) -> list[dict]:
    return [{"name": "never", "fixed": 0}, {"name": "always-12", "fixed": 12}, *bundle["policies"]]


def counterfactual(engine: BM25, q: dict, model: LocalReranker) -> dict:
    retrieval = engine.retrieve(q["question"])
    ids = [d for d, _ in retrieval]
    alternatives = {}
    for k in (0, 4, 12):
        start = time.perf_counter_ns()
        logits = model.score(q["question"], [engine.docs[d] for d in ids[:k]])
        elapsed = (time.perf_counter_ns() - start) / 1e6
        ranking = rerank(ids, logits)
        alternatives[str(k)] = {
            "logits": logits,
            "ranking": ranking,
            "quality": quality(ranking, q["relevance"]),
            "inference_ms": elapsed,
        }
    return {
        "id": q["id"],
        "split": q["split"],
        "retrieval": retrieval,
        "signals": engine.signals(q["question"], retrieval),
        "alternatives": alternatives,
    }


def measured(engine: BM25, q: dict, model: LocalReranker, policy: dict, rep: int) -> dict:
    start = time.perf_counter_ns()
    retrieval = engine.retrieve(q["question"])
    route_start = time.perf_counter_ns()
    signals = engine.signals(q["question"], retrieval)
    k = policy["fixed"] if "fixed" in policy else route(signals, policy, len(retrieval))
    router_ms = (time.perf_counter_ns() - route_start) / 1e6
    ids = [d for d, _ in retrieval]
    logits = model.score(q["question"], [engine.docs[d] for d in ids[:k]])
    ranking = rerank(ids, logits)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return {
        "id": q["id"],
        "policy": policy["name"],
        "repetition": rep,
        "depth": k,
        "logits": logits,
        "ranking": ranking,
        "quality": quality(ranking, q["relevance"]),
        "answerable": bool(q["relevance"]),
        "elapsed_ms": elapsed_ms,
        "router_ms": router_ms,
    }


def routing_errors(rows: list[dict], traces: list[dict], policies: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        for p in policies:
            allowed = [k for k in (0, 4, 12) if k <= p["budget"]]
            oracle = max(row["alternatives"][str(k)]["quality"]["ndcg3"] for k in allowed)
            oracle_depth = min(
                k for k in allowed if row["alternatives"][str(k)]["quality"]["ndcg3"] == oracle
            )
            chosen = next(t for t in traces if t["id"] == row["id"] and t["policy"] == p["name"])
            actual = chosen["quality"]["ndcg3"]
            if actual < oracle or chosen["depth"] > oracle_depth:
                result.append(
                    {
                        "id": row["id"],
                        "policy": p["name"],
                        "depth": chosen["depth"],
                        "oracle_depth": oracle_depth,
                        "regret": max(0.0, oracle - actual),
                        "type": "quality_loss" if actual < oracle else "avoidable_compute",
                    }
                )
    return result


def run(root: Path, output: Path, split: str, cache: Path, policy_path: Path | None) -> None:
    config = configuration(root / "config/experiment.json")
    docs, queries = validate(read(root / "data/corpus.json"), read(root / "data/queries.json"))
    if len(docs) < 12:
        raise ValueError("Experiment requires at least twelve documents")
    bundle = None
    if split == "eval":
        if policy_path is None:
            raise ValueError("Evaluation requires frozen development policies")
        verify(root, policy_path.parent)
        bundle = read(policy_path)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {
        str(p): digest(root / p)
        for p in map(Path, ["data/corpus.json", "data/queries.json", "config/experiment.json"])
    }
    start = time.perf_counter()
    model = LocalReranker(config, cache)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model.score("Warmup question", docs[:12])
    warmup_seconds = time.perf_counter() - start
    engine = BM25(docs)
    selected = [q for q in queries if q["split"] == split]
    rows = []
    for q in selected:
        rows.append(counterfactual(engine, q, model))
        print(f"{split} counterfactual {len(rows)}/{len(selected)}", flush=True)
    write(output / "counterfactual.json", rows)
    if split == "dev":
        bundle = {
            "policies": fit(rows, config),
            "development_sha256": digest(output / "counterfactual.json"),
            "input_hashes": hashes,
        }
    assert bundle is not None
    write(output / "policies.json", bundle)
    policies = policies_for(bundle)
    traces = []
    with (output / "trace.jsonl").open("w", encoding="utf-8") as stream:
        import json

        for rep in range(config["repetitions"]):
            for i, q in enumerate(selected):
                offset = (i + rep) % len(policies)
                for p in policies[offset:] + policies[:offset]:
                    trace = measured(engine, q, model, p, rep)
                    traces.append(trace)
                    stream.write(json.dumps(trace, allow_nan=False) + "\n")
                    stream.flush()
            print(f"{split} timed repetition {rep + 1}/{config['repetitions']}", flush=True)
    write(output / "summary.json", summarize(traces))
    write(output / "errors.json", routing_errors(rows, traces, bundle["policies"]))
    metadata = {
        "split": split,
        "input_hashes": hashes,
        "config": config,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "numpy", "tokenizers", "huggingface-hub")
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    if platform.system() == "Darwin":
        metadata["hardware"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], text=True
        ).strip()
    write(output / "metadata.json", metadata)
    write(
        output / "manifest.json",
        {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    )


def verify(root: Path, output: Path) -> dict:
    import json

    manifest = read(output / "manifest.json")
    expected_files = {
        "counterfactual.json",
        "policies.json",
        "trace.jsonl",
        "summary.json",
        "errors.json",
        "metadata.json",
    }
    if set(manifest) != expected_files:
        raise ValueError("Invalid evidence manifest")
    for name, sha in manifest.items():
        if digest(output / name) != sha:
            raise ValueError(f"Evidence hash mismatch: {name}")
    config = configuration(root / "config/experiment.json")
    docs, queries = validate(read(root / "data/corpus.json"), read(root / "data/queries.json"))
    meta = read(output / "metadata.json")
    expected_inputs = ["data/corpus.json", "data/queries.json", "config/experiment.json"]
    if meta["input_hashes"] != {p: digest(root / p) for p in expected_inputs}:
        raise ValueError("Input hashes differ")
    if meta["config"] != config or meta["split"] not in ("dev", "eval"):
        raise ValueError("Invalid metadata")
    selected = {q["id"]: q for q in queries if q["split"] == meta["split"]}
    rows = read(output / "counterfactual.json")
    if [r["id"] for r in rows] != list(selected):
        raise ValueError("Counterfactual query coverage mismatch")
    engine = BM25(docs)
    for row in rows:
        q = selected[row["id"]]
        retrieval = engine.retrieve(q["question"])
        if (
            row["split"] != q["split"]
            or row["retrieval"] != [list(x) for x in retrieval]
            or row["signals"] != engine.signals(q["question"], retrieval)
            or set(row["alternatives"]) != {"0", "4", "12"}
        ):
            raise ValueError("Retrieval evidence differs")
        for k, alt in row["alternatives"].items():
            ranking = rerank([d for d, _ in retrieval], alt["logits"])
            if (
                len(alt["logits"]) != int(k)
                or alt["ranking"] != ranking
                or alt["quality"] != quality(ranking, q["relevance"])
                or not finite(alt["inference_ms"])
                or alt["inference_ms"] < 0
            ):
                raise ValueError("Counterfactual ranking or metric differs")
    bundle = read(output / "policies.json")
    if bundle["input_hashes"] != meta["input_hashes"]:
        raise ValueError("Policy input mismatch")
    if meta["split"] == "dev":
        if bundle["policies"] != fit(rows, config) or bundle["development_sha256"] != digest(
            output / "counterfactual.json"
        ):
            raise ValueError("Development policy differs")
    else:
        dev_path = output.parent / "dev"
        verify(root, dev_path)
        if bundle != read(dev_path / "policies.json"):
            raise ValueError("Evaluation policy differs from frozen development policy")
    policies = {p["name"]: p for p in policies_for(bundle)}
    traces = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    seen = set()
    for t in traces:
        key = (t["id"], t["policy"], t["repetition"])
        if (
            key in seen
            or t["id"] not in selected
            or t["policy"] not in policies
            or type(t["repetition"]) is not int
            or not 0 <= t["repetition"] < config["repetitions"]
        ):
            raise ValueError("Invalid or duplicate timed sample")
        seen.add(key)
        q, p = selected[t["id"]], policies[t["policy"]]
        retrieval = engine.retrieve(q["question"])
        signals = engine.signals(q["question"], retrieval)
        depth = p["fixed"] if "fixed" in p else route(signals, p, len(retrieval))
        ranking = rerank([d for d, _ in retrieval], t["logits"])
        if (
            t["depth"] != depth
            or len(t["logits"]) != depth
            or t["ranking"] != ranking
            or t["quality"] != quality(ranking, q["relevance"])
            or t["answerable"] != bool(q["relevance"])
            or not finite(t["router_ms"])
            or not finite(t["elapsed_ms"])
            or not 0 <= t["router_ms"] <= t["elapsed_ms"]
        ):
            raise ValueError("Timed sample differs")
    if len(seen) != len(selected) * len(policies) * config["repetitions"]:
        raise ValueError("Missing timed samples")
    if read(output / "summary.json") != summarize(traces):
        raise ValueError("Summary differs")
    if read(output / "errors.json") != routing_errors(rows, traces, bundle["policies"]):
        raise ValueError("Routing errors differ")
    return {"verified": True, "split": meta["split"], "queries": len(rows), "samples": len(traces)}
