# Compact Qwen3 and direct-token validation

The experimental compact path loads the pinned Qwen3-4B-Base BF16 matrices
without expanding the entire transformer to F32. Weights stay on the RTX 3060;
one matrix at a time widens and transposes into reusable F32 scratch. A causal
prompt pass produces all final-normalised token rows for feature extraction, or
only the last row for direct choice scoring. The supported limit is 512 tokens.
Unsupported configs, overlength inputs and memory-budget violations fail rather
than silently falling back or truncating.

## Numerical gate

The independent local Transformers fixture uses CPU F32 eager attention over
BF16 source weights, no BOS/EOS insertion and no vocabulary projection. Five
inputs cover singleton options, Unicode, unequal lengths and 401 tokens. The
predeclared hidden-state bounds are maximum absolute error 0.005 and RMS 0.0002.

The first implementations failed the long-input bound: the ordinary CPU path
reached max 0.01593, and the initial compact GPU path reached 0.02047. Sequential
CPU execution reproduced the drift, ruling out batching as the explanation.
Layer traces showed amplification around large activation outliers. Diagnostic
higher-precision and split-accumulator comparisons isolated projection reduction
and float32 RoPE conventions as significant contributors.

The retained experimental path uses opt-in compensated F32 SGEMM accumulation
and Qwen float32 inverse-frequency/angle construction. It passes the unchanged
bound: **max 0.002205, RMS 0.00001169 across 1,111,040 values**. Ordinary SGEMM,
CPU/GGML RoPE and quantised default contracts remain unchanged. The legacy CPU
long-input path is not promoted under this new fixture.

Five direct-scoring prompts additionally match Transformers' rendered text and
token IDs exactly. Selected-token logits differ by at most **1.17e-5**, against a
predeclared 0.005 gate; all five argmax choices agree. The Base model makes wrong
answers on some of these questions, so this is parity evidence, not quality.
These fixtures are machine-authored engineering probes, not the planned
independently human-reviewed evaluation set.

## Prompt and output contract

`LoadQwen3ChoicePrompt` accepts the verified template SHA256
`87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` and implements its
single-user, no-tools, `enable_thinking=false` branch. Other template identities
are rejected. The checkpoint's `<think>` markers are non-special added tokens
supplied by `tokenizer_config.json`; `LoadWithConfig` preserves them atomically
without falsely marking ordinary tokens as special.

Every candidate code is checked after appending it to the complete rendered
prompt: it must add exactly one distinct ordinary token without altering the
prompt IDs. Unsupported candidate counts, reserved input tokens and overlength
prompts are rejected. Two to 26 letter-coded candidates are currently supported.

`ScoreChoices` returns stable application IDs, raw logits, argmax and temperature
softmax over the permitted candidates. Those are conditional candidate
preferences, not calibrated probabilities of correctness. An explicit unknown
answer is a candidate; deferral is a separate policy. No continuation is decoded
and no generated probability JSON is parsed.

The actual tied or untied BF16 output head is used, with optional bias. Only
selected rows are read. Their dot products use CPU F64 accumulation and transfer
only the final hidden row; CLI output explicitly reports this projection backend.
The transformer runs on GPU. Fresh CLI startup verifies all asset hashes and
requires a verified shard index, so directory names/hidden width are insufficient
identity checks.

## Memory, latency and lifecycle

At the 512-token allocation bound:

| Component | Bytes |
|---|---:|
| BF16 transformer matrices | 7,266,631,680 |
| F32 norms | 784,384 |
| RoPE table | 262,144 |
| Shared activations and F32 matrix scratch | 197,132,288 |
| Encoder-owned total | 7,464,810,496 |

Measured free GPU memory fell from 12,107,251,712 to 4,500,881,408 bytes, including
driver/allocation overhead. Closing the encoder restored 12,107,251,712 bytes
exactly in repeated tests. Close is idempotent, serialises with requests, and does
not tear down the process-global CUDA context. A CUDA initialisation OS-thread
lock leak discovered during exit testing was balanced; repeated short-lived
init-owner and direct-scoring test runs now exit cleanly.

Observed warm/checkpoint-cached load was about 1–1.7 s. The 401-token hidden-state
probe took about 7.15 s. Direct probes at 71–86 tokens took 1.48–1.74 s; one fresh
CLI run also spent 4.55 s checking all asset hashes. These are bounded fixture
measurements, not final p50/p95 or a quality/latency promotion result.

## Reproduction

```bash
MODEL=checkpoints/jevlike-qwen3/Qwen--Qwen3-4B-Base/906bfd4b4dc7f14ee4320094d8b41684abff8539
.venv-speaker/bin/python scripts/jevlike-qwen3-reference.py \
  --model "$MODEL" --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output checkpoints/jevlike-qwen3/reference/qwen3-4b-f32.json
.venv-speaker/bin/python scripts/jevlike-direct-reference.py \
  --model "$MODEL" --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output checkpoints/jevlike-qwen3/reference/direct-base-v1.json

JEVLIKE_QWEN3_MODEL_DIR="$PWD/$MODEL" \
JEVLIKE_QWEN3_REFERENCE="$PWD/checkpoints/jevlike-qwen3/reference/qwen3-4b-f32.json" \
JEVLIKE_QWEN3_BACKEND=gpu go test ./model/jevlike -run '^TestQwen3FrozenReference$' -v -count=1

JEVLIKE_QWEN3_MODEL_DIR="$PWD/$MODEL" \
JEVLIKE_DIRECT_REFERENCE="$PWD/checkpoints/jevlike-qwen3/reference/direct-base-v1.json" \
  go test ./model/jevlike -run '^TestDirectQwen3Reference$' -v -count=1

go run ./cmd/jevlike score -encoder-model "$MODEL" \
  -verified-assets checkpoints/jevlike-qwen3/model-verified.json -request request.json
```

Reference outputs must not already exist. GPU runs require adequate free VRAM;
the existing model service was stopped with permission and restored after each
measurement. No training cache has been generated. Instruction-checkpoint
selection, held-out quality, calibration and final performance comparisons remain
open under [issue #2](https://github.com/rcarmo/go-pherence/issues/2).
