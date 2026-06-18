# Gemma4 MTP llama.cpp first-difference notes

Current strict selected-logit parity remains red with five stable mismatches. Default accepted-token parity is green.

## Exact fixture flow

- Prompt fixture tokens: `[10979, 236764]`
- Target context tokens for llama.cpp/Go comparison: `[2, 10979]`
- Verifier batch tokens: `[236764, 564, 236789]`
- Row 1 / position 3 is the dominant strict-logit outlier.

## Go row-vector dump

Use the committed Go trace hook:

```bash
GO_PHERENCE_MTP_TRACE_SUMMARY=1 \
GO_PHERENCE_MTP_TRACE_ROW=1 \
GO_PHERENCE_MTP_TRACE_POS=3 \
GO_PHERENCE_MTP_TRACE_DUMP_DIR=/tmp/go_mtp_dump \
GO_PHERENCE_GEMMA4_MTP_STRICT=1 \
GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=$PWD/tmp/gemma4-mtp-llamacpp-fixture.json \
GO_PHERENCE_GEMMA4_MAIN=$PWD/models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf \
GO_PHERENCE_GEMMA4_MTP_DRAFTER=$PWD/models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf \
GOTMPDIR=$PWD/.gotmp \
go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1 -v
```

## Local llama.cpp trace harness

The local `/tmp/llama.cpp-gemma4-mtp/examples/speculative-simple` was temporarily patched to support:

- `LLAMA_SPEC_TRACE_ONCE=1`
- `LLAMA_SPEC_PROMPT_TOKENS=2,10979,236764`
- `LLAMA_SPEC_DRAFT_TOKENS=564,236789`
- `LLAMA_SPEC_TRACE_ROW=1`
- `LLAMA_SPEC_DEBUG_TENSORS=1`
- `LLAMA_SPEC_DEBUG_FILTER=...`
- `LLAMA_SPEC_DUMP_DIR=/tmp/llama_mtp_dump`

The important point is that BOS **must** be included in `LLAMA_SPEC_PROMPT_TOKENS`; otherwise the apparent first divergence shifts to RoPE because positions are off by one.

Example:

```bash
LLAMA_SPEC_DUMP_DIR=/tmp/llama_mtp_dump \
LLAMA_SPEC_TRACE_ROW=1 \
LLAMA_SPEC_TRACE_ONCE=1 \
LLAMA_SPEC_PROMPT_TOKENS='2,10979,236764' \
LLAMA_SPEC_DRAFT_TOKENS='564,236789' \
LLAMA_SPEC_DEBUG_TENSORS=1 \
LLAMA_SPEC_DEBUG_FILTER='__fattn__-0|attn_out-0|l_out-0' \
/tmp/llama.cpp-gemma4-mtp/build/bin/llama-speculative-simple \
  -m models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -p x -n 3 -fa on -b 8 -ub 8
```

## Current first-difference result

With exact tokens and row selection fixed, Go and llama match exactly or within about `2e-6` through:

- layer 0 `attn_norm`
- layer 0 `q_proj`
- layer 0 `q_norm`
- layer 0 `q_pos`

First meaningful divergence:

- llama: `__fattn__-0`
- Go: `attn_pre_o`
- row 1 / position 3
- max abs ≈ `0.0060415`
- mean abs ≈ `0.0002053`
- top lane: head 3, dim 29

This then amplifies:

- layer 0 `attn_out` max abs ≈ `0.1384`
- layer 0 `l_out` max abs ≈ `0.0275`

## Interpretation

The strict gap is rooted in the verifier attention output composition, not in:

- tokenizer/prompt trimming
- projections / quant matmuls
- per-layer input rows/projection
- RMSNorm / no-scale RMSNorm
- RoPE ordering/factors
- GEGLU
- tail LM head / softcap
- shared-KV source mapping
- SWA mask ranges

A simple replacement with exact ggml `flash_attn_ext` is not sufficient when applied globally, verifier-only, or late-layer-only; those attempts improve some rows/tokens but create tradeoffs. The likely remaining work is a faithful pure-Go port of the relevant `ggml_flash_attn_ext` F16 K/V online-softmax composition, especially `ggml_vec_scale_f16` / `ggml_vec_mad_f16` lane/store behavior.
