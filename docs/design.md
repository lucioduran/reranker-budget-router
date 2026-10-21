# Experiment design

BM25 retrieves all documents using lowercase ASCII word tokens, k1=1.2 and b=0.75.
A router sees only the normalized top-two score gap and query-token coverage of the
first document. It never receives relevance labels. Confidence is the smaller of
gap and coverage. High confidence skips reranking; intermediate confidence scores
four candidates; low confidence scores twelve. A hard per-query pair budget rounds
down to a supported depth (0, 4, 12). Budgets below four skip reranking.

Grid search on development queries chooses two ordered confidence thresholds for
each budget and compute penalty. The objective is mean nDCG@3 minus penalty times
mean scored pairs. Ties prefer fewer pairs, then lower thresholds. There is no
model training. The pretrained cross-encoder is frozen. Policies are saved before
evaluation and tied to the development evidence by SHA-256.

Counterfactual runs score every supported depth on development and evaluation.
Reranking reorders only the prefix and appends the untouched retrieval tail. Fixed
baselines never rerank or always rerank twelve candidates. Separate timed policy
runs include retrieval, feature extraction, routing, tokenization and inference.
Three serial warm repetitions run with two CPU threads. Policy order rotates by
query and repetition. Cold loading and one model warmup are recorded separately.
Counterfactual scores support per-query oracle regret and visible routing errors.
The oracle sees labels, so it is diagnostic and not a deployable policy.

Cost means query-document pairs scored, not dollars, energy or tokens. Pair count
is a hard cap; latency is an observation, not an SLA. Token lengths affect runtime.
All metrics include unanswerable queries with nDCG and reciprocal rank equal to zero;
answerable-only quality is also reported. No abstention mechanism is implemented.
The quality/latency/pair frontier uses measured mean end-to-end latency. A policy is
dominated when another is no worse on every axis and strictly better on one.

The standard-library verifier reconstructs retrieval, features, rankings, routing,
quality, budget compliance and aggregates from saved logits and timings. It verifies
input hashes and policy fit. It cannot prove that saved logits came from a model or
that timings are authentic. Full reproduction runs real local inference.
