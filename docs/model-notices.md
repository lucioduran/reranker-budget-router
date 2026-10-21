# Model and environment

The model is [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Its model card declares Apache-2.0.
Weights are downloaded separately and are not redistributed in this repository.
The model is trained on MS MARCO; see its card for training details and limitations.
The [Sentence Transformers documentation](https://sbert.net/docs/cross_encoder/pretrained_models.html)
describes the query-document scoring interface. This project uses Transformers directly,
CPU float32 inference, eval mode, no remote code, safetensors only, and a 256-token
combined input limit. Raw logits are ranking scores, not calibrated probabilities.

`requirements-local.lock` records the actual macOS ARM environment, including development
tools. It is not a universal cross-platform lock. Core replay uses only Python's standard
library. Model requirements pin top-level inference dependencies. PyTorch 2.2.2 availability
depends on Python and platform. Python 3.11 is the tested inference version.

No paid APIs, hosted inference, rented GPUs or external scoring service are used.
Model download requires network access; inference uses cached files only.
