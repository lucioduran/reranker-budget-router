"""Render measured evidence as a compact Markdown report."""

import argparse
from collections import Counter
from pathlib import Path

from budget_router.core import read
from budget_router.experiment import verify


def render(root: Path, evidence: Path) -> str:
    verify(root, evidence)
    summary = read(evidence / "summary.json")
    meta = read(evidence / "metadata.json")
    errors = read(evidence / "errors.json")
    rows = read(evidence / "counterfactual.json")
    policies = read(evidence / "policies.json")["policies"]
    queries = {q["id"]: q for q in read(root / "data/queries.json")["queries"]}
    lines = [
        "# Held-out results",
        "",
        f"Hardware: {meta.get('hardware', meta['processor']).replace(chr(10), '; ')}.",
        f"Platform: {meta['platform']}. Python {meta['python']}. Two CPU threads.",
        f"Versions: `{meta['versions']}`.",
        "",
        f"Model load: {meta['load_seconds']:.3f} s. Warmup: {meta['warmup_seconds']:.3f} s.",
        "Neither is included in warm request latency. Each policy has three serial repetitions",
        "of 26 queries. These are 26 query units, not 78 independent relevance observations.",
        "",
        "| Policy | nDCG@3 | Answerable nDCG@3 | MRR | Mean pairs | Mean ms | p95 ms | Router ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in summary["policies"].items():
        lines.append(
            f"| {name} | {s['ndcg3']:.4f} | {s['answerable_ndcg3']:.4f} | "
            f"{s['mrr']:.4f} | {s['mean_pairs']:.2f} | {s['mean_ms']:.3f} | "
            f"{s['p95_ms']:.3f} | {s['mean_router_ms']:.4f} |"
        )
    lines += [
        "",
        "Observed nondominated quality/mean-latency/pair frontier: "
        + ", ".join(f"`{n}`" for n in summary["frontier"])
        + ".",
        "",
        "The frontier uses raw measured means. Tiny differences between identical routing",
        "decisions are timing noise, not evidence of one policy being faster.",
        "",
        "## Frozen decisions",
        "",
        "| Policy | Low threshold | High threshold | Budget |",
        "| --- | ---: | ---: | ---: |",
    ]
    for p in policies:
        lines.append(f"| {p['name']} | {p['low']} | {p['high']} | {p['budget']} |")
    lines += [
        "",
        "These policies were fitted on 12 development queries only. If both thresholds are",
        "zero, the fitted policy always skips reranking, irrespective of the budget. This",
        "is a development-set outcome, not proof that adaptive routing works on broader data.",
        "",
        "## Per-query counterfactual quality",
        "",
        "All rows are visible, including unanswerable queries. Each entry is nDCG@3.",
        "Counterfactual depths are measured independently of the timed policy repetitions.",
        "",
        "| Query | Answerable | Never | Depth 4 | Depth 12 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        values = [r["alternatives"][str(k)]["quality"]["ndcg3"] for k in (0, 4, 12)]
        lines.append(
            f"| {r['id']} | {bool(queries[r['id']]['relevance'])} | "
            + " | ".join(f"{v:.4f}" for v in values)
            + " |"
        )
    counts = Counter(e["type"] for e in errors)
    lines += [
        "",
        "## Routing errors",
        "",
        f"Policy-query error counts: {dict(counts)}. These count each budget/penalty policy",
        "separately; they are not independent queries. See [errors.json](eval/errors.json)",
        "for chosen depth, cheapest budget-feasible oracle depth and nDCG regret.",
        "",
        "## Interpretation limits",
        "",
        "This is a small authored fixture, with no external validity claim. A lexical quality",
        "ceiling can prevent development tuning from learning when to spend compute.",
        "If evaluation reranking improves a query, a skip-only policy cannot capture that gain.",
        "Two unanswerable questions contribute zero to all quality means; answerable-only",
        "quality is shown separately. Repeated timings do not increase the relevance sample size.",
        "",
        "No dollar, energy, concurrent-load or latency-SLA claim follows from pair counts.",
        "The model is frozen, the corpus is small, and no human relevance validation was run.",
        "The first development measurement was rejected because unordered token summation",
        "failed cross-process replay. It was rerun after deterministic token ordering was fixed.",
        "The accepted dev and eval evidence passed the verifier. No evaluation tuning was done.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evidence", type=Path, default=Path("results/eval"))
    parser.add_argument("--output", type=Path, default=Path("results/report.md"))
    args = parser.parse_args()
    rendered = render(Path("."), args.evidence)
    if args.check:
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Report differs from verified evidence")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
