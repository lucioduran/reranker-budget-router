# Reranker Budget Router

[![CI](https://github.com/lucioduran/reranker-budget-router/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/reranker-budget-router/actions/workflows/ci.yml)

Choose how many retrieval candidates to rerank under a hard pair budget. Inspect
local model measurements, frozen policies, and every routing decision.

## Problem

Reranking every retrieved document adds compute even when retrieval is already correct.
Skipping reranking can lose quality on ambiguous queries. A router needs measured
tradeoffs and visible failure cases rather than an assumed speedup.

## Scope

A standard-library BM25 retriever and confidence router choose 0, 4, or 12 candidates.
A real pinned MiniLM cross-encoder runs locally on CPU. The experiment compares six
development-tuned budget/penalty profiles with never-rerank and always-rerank-12 baselines.
The included corpus contains 24 fictional documents, 12 development questions and
26 held-out evaluation questions. See [provenance](data/README.md).

## Quick start

Python 3.11 or newer is required for the dependency-free replay:

```sh
git clone https://github.com/lucioduran/reranker-budget-router.git
cd reranker-budget-router
python -m budget_router verify --output results/dev
python -m budget_router verify --output results/eval
python -m budget_router route 'Where can I restore a deleted notebook?' --budget 4
```

The route command returns the selected depth, confidence signals and candidate IDs.
It makes no model call. No account, API key or paid service is required.

## Usage

```sh
python -m budget_router route 'Where do drones charge?' --budget 12 --policy router-b12-p0
python -m budget_router route 'Where do drones charge?' --budget 0
```

`--budget` is the maximum number of query-document pairs to score for one request.
Unsupported depths round down to 0, 4, or 12. Negative budgets fail. The selected
policy supplies thresholds; overriding its budget does not refit it.

`download` fetches pinned model files. `run --split dev` fits policies only on development
labels. `run --split eval --policies PATH` requires verified development evidence.
`verify --output PATH` recomputes retrieval, routing, metrics and aggregate reports.
Commands exit with 0 on success and 2 on invalid input or missing resources.
The Python primitives in `budget_router.core` can be used without inference dependencies.

## Architecture

```text
BM25 retrieval -> score gap and token coverage -> budget-constrained depth
                                                    |
                               pinned CPU cross-encoder on selected prefix
                                                    |
                               stable prefix reorder + untouched tail
```

Development grid search freezes threshold policies before evaluation. Timed policy
runs include retrieval, signals, routing, tokenization and actual inference. Separate
counterfactual scores expose oracle regret and avoidable computation. Read the
[design](docs/design.md) for exact formulas, tie-breaking and measurement boundaries.

## Reproduction

Use Python 3.11 on macOS or Linux. Install a project-local environment:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-models.txt -r requirements-dev.txt
python -m budget_router download
python scripts/reproduce.py --output build/reproduction
```

The reproduction script holds an exclusive Unix file lock for both splits, limits
CPU library threads to two, and verifies each split. Pass a shared `--lock PATH` when
coordinating multiple projects. Output must not exist. Evaluation and development
folders must be siblings named `eval` and `dev`; the verifier checks the frozen policy
against the sibling development evidence. Model downloads need network access;
benchmark inference uses only local cached files.

The published runs used the series-wide wrapper
`/Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py` around
every heavy invocation. [Model notices](docs/model-notices.md) and
[requirements-local.lock](requirements-local.lock) record the model revision and
actual installed packages. Metadata in each result directory records hardware,
versions, model loading, warmup and source commit. Later commits add evidence and docs.

## Results

On the held-out fixture, every learned policy skips reranking. Never-rerank achieves
0.8947 nDCG@3 at 0.108 ms mean latency; always-rerank-12 achieves 0.9089 at
101.321 ms. The router misses the benefit on `eval-iris-1` and adds overhead to
the never-rerank baseline. This experiment does not demonstrate a router advantage.

Measurements and interpretation are in [results/report.md](results/report.md).
Machine-readable [evaluation summary](results/eval/summary.json),
[routing errors](results/eval/errors.json), and [timed traces](results/eval/trace.jsonl)
expose all decisions. Cost means scored pairs, not dollars or energy.

## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m mypy budget_router
python -m coverage run --branch --source=budget_router -m unittest discover -v
python -m coverage report --fail-under=80
python -m budget_router verify --output results/dev
python -m budget_router verify --output results/eval
```

CI runs these checks on Linux, macOS and Windows with Python 3.11.9. Tests cover
budget boundaries, split leakage, ranking math, tie stability, policy fitting,
fresh-process replay and rehashed evidence tampering. Unit tests use an explicitly
labeled test double; published measurements use the real model. CI does not download
weights or rerun neural inference.

## Limitations

The tiny authored corpus is not a production workload or an independent benchmark.
Questions in the same document group are correlated. Development has just six groups.
Two evaluation questions have no relevant document and receive zero quality; there
is no abstention mechanism. The model cannot recover a document outside its candidate
prefix. English tokenization and fixed 256-token truncation limit generality.

Three warm serial repetitions on one CPU are insufficient for load, concurrency,
energy, dollar-cost or tail-latency claims. Pair budgets are hard; latency budgets
are not implemented. Hash replay checks consistency, not measurement authenticity.
No human relevance study, external benchmark, production deployment, security audit
or interoperability certification was performed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep evaluation data out of policy tuning and
publish new evidence when changing retrieval or routing behavior.

## License

Code and fixture data are MIT licensed. See [LICENSE](LICENSE). Downloaded model
weights have their own Apache-2.0 license; see [model notices](docs/model-notices.md).
