# Held-out results

Hardware: Intel(R) Xeon(R) W-3223 CPU @ 3.50GHz; 51539607552.
Platform: macOS-26.7.1-x86_64-i386-64bit. Python 3.11.13. Two CPU threads.
Versions: `{'torch': '2.2.2', 'transformers': '4.46.3', 'numpy': '1.26.4', 'tokenizers': '0.20.3', 'huggingface-hub': '0.26.2'}`.

Model load: 1.591 s. Warmup: 0.743 s.
Neither is included in warm request latency. Each policy has three serial repetitions
of 26 queries. These are 26 query units, not 78 independent relevance observations.

| Policy | nDCG@3 | Answerable nDCG@3 | MRR | Mean pairs | Mean ms | p95 ms | Router ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always-12 | 0.9089 | 0.9846 | 0.9038 | 12.00 | 101.321 | 126.578 | 0.0065 |
| never | 0.8947 | 0.9692 | 0.8846 | 0.00 | 0.108 | 0.152 | 0.0062 |
| router-b12-p0 | 0.8947 | 0.9692 | 0.8846 | 0.00 | 0.114 | 0.176 | 0.0094 |
| router-b12-p0.01 | 0.8947 | 0.9692 | 0.8846 | 0.00 | 0.112 | 0.154 | 0.0097 |
| router-b12-p0.04 | 0.8947 | 0.9692 | 0.8846 | 0.00 | 0.109 | 0.139 | 0.0092 |
| router-b4-p0 | 0.8947 | 0.9692 | 0.8846 | 0.00 | 0.160 | 0.201 | 0.0165 |
| router-b4-p0.01 | 0.8947 | 0.9692 | 0.8846 | 0.00 | 0.127 | 0.194 | 0.0126 |
| router-b4-p0.04 | 0.8947 | 0.9692 | 0.8846 | 0.00 | 0.116 | 0.152 | 0.0100 |

Observed nondominated quality/mean-latency/pair frontier: `always-12`, `never`.

The frontier uses raw measured means. Tiny differences between identical routing
decisions are timing noise, not evidence of one policy being faster.

## Frozen decisions

| Policy | Low threshold | High threshold | Budget |
| --- | ---: | ---: | ---: |
| router-b4-p0 | 0.0 | 0.0 | 4 |
| router-b4-p0.01 | 0.0 | 0.0 | 4 |
| router-b4-p0.04 | 0.0 | 0.0 | 4 |
| router-b12-p0 | 0.0 | 0.0 | 12 |
| router-b12-p0.01 | 0.0 | 0.0 | 12 |
| router-b12-p0.04 | 0.0 | 0.0 | 12 |

These policies were fitted on 12 development queries only. If both thresholds are
zero, the fitted policy always skips reranking, irrespective of the budget. This
is a development-set outcome, not proof that adaptive routing works on broader data.

## Per-query counterfactual quality

All rows are visible, including unanswerable queries. Each entry is nDCG@3.
Counterfactual depths are measured independently of the timed policy repetitions.

| Query | Answerable | Never | Depth 4 | Depth 12 |
| --- | --- | ---: | ---: | ---: |
| eval-garden-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-garden-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-harbor-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-harbor-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-iris-1 | True | 0.6309 | 1.0000 | 1.0000 |
| eval-iris-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-junction-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-junction-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-keel-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-keel-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-lantern-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-lantern-2 | True | 0.6309 | 0.6309 | 0.6309 |
| eval-mirror-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-mirror-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-nova-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-nova-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-orbit-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-orbit-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-prism-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-prism-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-quartz-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-quartz-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-rivet-1 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-rivet-2 | True | 1.0000 | 1.0000 | 1.0000 |
| eval-unknown-weather | False | 0.0000 | 0.0000 | 0.0000 |
| eval-unknown-captain | False | 0.0000 | 0.0000 | 0.0000 |

## Routing errors

Policy-query error counts: {'quality_loss': 6}. These count each budget/penalty policy
separately; they are not independent queries. See [errors.json](eval/errors.json)
for chosen depth, cheapest budget-feasible oracle depth and nDCG regret.

## Interpretation limits

This is a small authored fixture, with no external validity claim. A lexical quality
ceiling can prevent development tuning from learning when to spend compute.
If evaluation reranking improves a query, a skip-only policy cannot capture that gain.
Two unanswerable questions contribute zero to all quality means; answerable-only
quality is shown separately. Repeated timings do not increase the relevance sample size.

No dollar, energy, concurrent-load or latency-SLA claim follows from pair counts.
The model is frozen, the corpus is small, and no human relevance validation was run.
The first development measurement was rejected because unordered token summation
failed cross-process replay. It was rerun after deterministic token ordering was fixed.
The accepted dev and eval evidence passed the verifier. No evaluation tuning was done.
