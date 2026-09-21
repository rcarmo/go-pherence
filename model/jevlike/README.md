## Jevlike in Go

Jevlike scores a context against a changing list of text options in one pass. Each option queries the context through a shared attention head; the result is one logit per option. The byte encoder can be trained from scratch, or a frozen local decoder can supply token hidden states while only the head is trained.

The runtime is native Go. Projection, dot and linear-backward accumulation operations use the SIMD runtime, including Plan 9 assembly on supported CPUs and portable fallbacks elsewhere. Python is needed only for importing/exporting upstream PyTorch checkpoints and regenerating reference fixtures.

```sh
go run ./cmd/jevlike --help
go run ./cmd/jevlike synthetic -output-dir /tmp/choices
go run ./cmd/jevlike train -help
go run ./cmd/jevlike predict -help
go run ./cmd/jevlike eval -help
```

JSONL rows contain a string `context`, at least two non-empty strings in `options`, and a zero-based integer `label`. The tiny encoder truncates UTF-8 bytes, not characters. Defaults are width 64, rank 64, 192 context bytes and 32 option bytes.

The [frozen Qwen3 experiment](../../docs/experiments/jevlike-qwen3/README.md)
adds pinned preparation, compact GPU extraction, selected-token scoring, cached
head training and isolated K/V-prefix APIs. Those bounded runtime checks passed;
direct scorers failed option-order gates and frozen heads failed learning gates.
The [final evaluation](../../docs/experiments/jevlike-qwen3/final-report-20260921.md)
completed 1,440/1,440 originals: Instruction reached 81.25%, while the three heads
averaged 22.20% against a 23.52% random expectation. Jevlike is not a validated
decision model. [Current status](../../docs/experiments/jevlike-qwen3/status-report-20260919.md)
and [command families](../../docs/guides/commands.md#jevlike-experiment-commands)
separate available APIs from the failed quality gates. Conversion uses an isolated
Python helper; Go owns task and split contracts.

## Frozen models and checkpoints

`frozen-train`, `frozen-predict` and `frozen-eval` take `-encoder-model` pointing to a local model directory with safetensors, configuration and `tokenizer.json`. The native dense causal decoder supplies final-normalised hidden states without computing logits. Option vectors are means of their token states.

The tokenizer does not automatically add special tokens. `-encoder-bos` explicitly prepends a token ID; the default is -1 (none). Match the tokenizer and special-token policy to the source checkpoint. Arbitrary Hugging Face encoders are not supported: Gemma4 per-layer inputs and mixture-of-experts models are explicitly rejected by the hidden-state API. An opt-in Qwen2.5-0.5B fixture checks real-checkpoint hidden states and scorer logits against Transformers; it is not a guarantee for every supported decoder.

Native checkpoints are versioned JSON containing config and named, shaped tensors. Loads reject missing/duplicate tensors, invalid shapes and non-finite values; saves use a temporary file and rename. Frozen checkpoints contain only the head and encoder reference, not backbone weights.

```sh
python scripts/jevlike_checkpoint.py to-json source.pt model.json
python scripts/jevlike_checkpoint.py to-pt model.json restored.pt
```

The converter requires PyTorch and uses `weights_only=True` without an unsafe pickle fallback. Treat imported model files as trusted inputs anyway; the native JSON loader is not a resource-limited service endpoint.

## Visual adapters without games

`VisionEncoder` consumes 120x160 RGB-plus-motion observations and produces an 8x10 patch grid. Its convolutions use SIMD GEMM. `VisionActionScorer` accepts those features, adds fixed 2D positions after key projection, supports selected action embeddings and multiple reads, and returns logits, value estimates and optional attention traces.

The package includes frame-difference packing and NCHW conversion. It does not run game environments, record videos or train a visual policy. The upstream plain-convolution control policy and visual checkpoint conversion are not implemented.

## Tests and current limits

Run `go test ./model/jevlike ./cmd/jevlike` and `go vet ./model/jevlike ./cmd/jevlike`. Tests cover PyTorch tiny/head forward and gradient fixtures, finite-difference gradients, synthetic training loss, checkpoint round trips, context shuffling and CLI workflows. The complete visual encoder and multi-read action scorer match deterministic PyTorch fixtures, covering preprocessing, patch features, logits, values and attention traces. This does not establish parity for a complete trained game policy.

Synthetic generation is deterministic in Go but does not reproduce Python's RNG sequence. Initialisation uses upstream-compatible distributions (unit-normal embedding/position weights and bounded uniform projections), but does not reproduce PyTorch's random stream. ECE includes confidence exactly equal to one in its last bin, unlike the upstream half-open bin loop.

Pinned workload measurements and commands are in [VALIDATION.md](VALIDATION.md). Upstream attribution and licence notices are in [PROVENANCE.md](PROVENANCE.md).
