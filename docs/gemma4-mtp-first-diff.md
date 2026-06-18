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

Additional K/V dump comparison confirms the attention inputs are aligned:

- layer 0 `k_norm` / `k_pos`: max abs about `9e-8`
- layer 0 `v_norm`: max abs about `7e-7`

Exact row-vector dump comparison for row 1 / position 3 / layer 0:

```text
attn_norm: max=0         mean=0
q_norm:    max≈1.9e-6    mean≈1.1e-7
q_pos:     max≈1.9e-6    mean≈1.2e-7
k_norm:    max≈9e-8      mean≈1.3e-8
k_pos:     max≈9e-8      mean≈1.4e-8
v_norm:    max≈7e-7      mean≈1.3e-7
```

First meaningful divergence:

- llama: `__fattn__-0`
- Go: `attn_pre_o`
- row 1 / position 3
- max abs ≈ `0.0060415`
- mean abs ≈ `0.0002053`
- top lane: head 3, dim 29

Top lanes from the original `__fattn__` vs default Go `attn_pre_o` dump include:

```text
idx=797  head=3 dim=29   llama=13.2350025  go=13.2410440  delta=+0.0060415
idx=693  head=2 dim=181  llama= 2.9872022  go= 2.9902635  delta=+0.0030613
idx=1174 head=4 dim=150  llama= 2.0064831  go= 2.0036089  delta=-0.0028741
idx=541  head=2 dim=29   llama=10.3588076  go=10.3614120  delta=+0.0026045
idx=29   head=0 dim=29   llama=13.8169575  go=13.8147545  delta=-0.0022030
```

After porting ggml's AVX-style F16 `vec_scale`, `vec_mad`, and `vec_dot` semantics into the default-off pure flash diagnostic path, the same saved llama `__fattn__-0` row1 dump compared to a regenerated Go layer0/pos3 `attn_pre_o` dump improves to:

```text
max_abs ≈ 0.00059104
mean_abs ≈ 0.000003126
idx=598 llama=0.60993665 go=0.60934561 delta=-0.00059104
idx=908 llama=0.42969593 go=0.42923722 delta=-0.00045872
idx=299 llama=0.42867798 go=0.42836937 delta=-0.00030860
idx=379 llama=0.43676037 go=0.43645179 delta=-0.00030857
idx=548 llama=-0.40136808 go=-0.40107319 delta=+0.00029489
```

This confirms the local layer0/pos3 attention kernel is now about an order of magnitude closer, but still not bit-exact; the strict selected-logit A/B improves row1's dominant token while still trading off row2, so the path remains diagnostic/default-off.

The same regenerated pure-flash layer0/pos3 trace still shows downstream amplification after the O projection/residual path:

```text
llama attn_out-0 row1 vs Go attn_out layer0/pos3:
  max_abs ≈ 0.0263062
  mean_abs ≈ 0.000401663
  top lane idx=30 llama=286.9996948 go=287.0260010 delta=+0.0263062

llama l_out-0 row1 vs Go l_out layer0/pos3:
  max_abs ≈ 0.0275288
  mean_abs ≈ 0.000981535
  top lane idx=30 llama=18.1525288 go=18.1250000 delta=-0.0275288
```

So after the flash kernel improvement, the next fidelity target is the composition around O projection, post-attention residual/norm/layer-scalar/BF16 store boundaries rather than broad Q/K/V/RoPE or LM-head logic.

A new Go `l_out_pre_bf16` trace point shows that llama's saved `l_out-0` row1 dump corresponds much more closely to Go's pre-BF16 layer output than to Go's post-BF16 stored layer output:

```text
llama l_out-0 row1 vs Go l_out_pre_bf16 layer0/pos3:
  max_abs ≈ 0.00790989
  mean_abs ≈ 0.000793528
  idx30 llama=18.1525288 go=18.1536274 delta=+0.00109863

llama l_out-0 row1 vs Go post-BF16 l_out layer0/pos3:
  max_abs ≈ 0.0275288
  mean_abs ≈ 0.000981535
  idx30 llama=18.1525288 go=18.1250000 delta=-0.0275288
```

That means part of the apparent `l_out` mismatch is trace-boundary naming: llama's trace is likely pre-store/pre-BF16, while Go's historical `l_out` trace was after the Gemma4 BF16 store boundary. The runtime BF16 boundary may still be correct, but future trace comparisons should use `l_out_pre_bf16` when comparing against llama's `l_out-*` dumps.

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
