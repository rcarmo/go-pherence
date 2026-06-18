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
- `LLAMA_SPEC_DUMP_DIR=/tmp/llama_mtp_dump` (patched filenames include an occurrence index for repeated callbacks, e.g. `llama_result_output_occ1_row1_n262144.f32`)

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

A default-off diagnostic gate can now skip the final layer BF16 store for selected layer/position:

```bash
GO_PHERENCE_MTP_SKIP_LAYER_BF16=1
GO_PHERENCE_MTP_SKIP_LAYER_BF16_LAYER=0
GO_PHERENCE_MTP_SKIP_LAYER_BF16_POS=3
```

For layer0/pos3 alone, this improves the row1 dominant selected-logit delta from about `-0.3837` to about `-0.0648`, while row2 worsens (`token236751` about `+0.0922`). Combining this with the pure-flash diagnostic worsens row1 again (`token236757` about `-0.4117`) while improving row2, so the residual is a graph-composition interaction rather than a single independent replacement rule.

Layer0 BF16-store skip matrix without pure-flash substitution:

```text
skip layer0 pos2:
  row0 max≈0.0542, row1 token236757≈-0.3651, row2 max≈0.0443
skip layer0 pos3:
  row0 unchanged, row1 token236757≈-0.0648, row2 token236751≈+0.0922
skip layer0 pos4:
  row0 unchanged, row1 unchanged, row2 max≈0.0399
skip all layer0 verifier positions:
  row0 max≈0.1062, row1 token236757≈-0.0837, row2 token236751≈-0.1077
```

This reinforces that the BF16 boundary affects the verifier rows through the batched graph composition and next-row state, not as a globally removable layer rule.

Direct inspection of llama.cpp `src/models/gemma4.cpp` shows the Gemma4 graph does **not** cast layer output to BF16 after `out_scaled` / `l_out`; it does:

```cpp
if (model.layers[il].out_scale) {
    cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
    cb(cur, "out_scaled", il);
}
cur = build_cvec(cur, il);
cb(cur, "l_out", il);
inpL = cur;
```

No `ggml_cast(..., GGML_TYPE_BF16)` appears at that boundary. A default-off all-layer/all-position skip of Go's Gemma4 layer BF16 store gives:

```text
GO_PHERENCE_MTP_SKIP_LAYER_BF16=1:
  row0 max≈0.0429
  row1 token236757≈-0.3045
  row2 token236751≈-0.1024

GO_PHERENCE_MTP_SKIP_LAYER_BF16=1 + GO_PHERENCE_MTP_PURE_FLASH=1:
  row0 max≈0.1354
  row1 token236757≈-0.2670
  row2 token236757≈+0.00051 but row2 token236751≈-0.0820
```

So Go's Gemma4 layer BF16 boundary is a graph deviation from llama.cpp. It was removed from the default Gemma4 MTP layer path after confirming default accepted-token parity stays green. The strict selected-logit baseline then becomes 6 mismatches:

```text
row0 token236751 got=14.1446142 want=14.1260710 delta=+0.0185432
row0 token236757 got=15.2410393 want=15.1981421 delta=+0.0428972
row1 token236751 got= 7.3078938 want= 7.3486571 delta=-0.0407634
row1 token236757 got=13.6337891 want=13.9382925 delta=-0.3045034
row2 token236751 got=20.1533260 want=20.2557411 delta=-0.1024151
row2 token236757 got=28.6179848 want=28.6220856 delta=-0.0041008
```

That is not strict-green, but it is a closer backend graph: the previous Gemma4 BF16 store was compensating for other remaining differences and has been removed from the production Gemma4 MTP prompt/verifier path. A code audit confirms `forwardMTPPromptLayer` now only applies BF16 stores for `gemma3_text` at input/FFN/output boundaries; Gemma4 remains F32 through those layer boundaries and only uses the ggml-compatible F16/BF16 semantics where llama.cpp does (flash K/V, GELU table/quantized tensors). The ordinary CPU `Generate` layer path was also aligned to the same rule: Gemma4 no longer uses BF16 layer-boundary/QKV/FFN norm-store shortcuts there; only Gemma3 keeps those BF16 path variants. GPU-specific Gemma4 BF16 stores still need a separate audit before the GPU path is treated as llama.cpp-equivalent, but they are not the current strict MTP verifier path. The default-off skip gate remains mostly useful for Gemma3 or future diagnostic toggles, not as the Gemma4 MTP production behavior.

After removing the Gemma4 BF16 layer-store deviation, the pure-flash diagnostic A/B shifts:

```text
pure flash layer0 pos2:
  row0 token236757≈+0.0742, row1 token236757≈-0.0805, row2 max≈0.0178
pure flash layer0 pos3:
  row0 unchanged, row1 token236757≈-0.1939, row2 token236751≈+0.0408
pure flash layer0 pos4:
  row0/row1 unchanged, row2 token236751≈-0.1587
pure flash all layer0 positions:
  row0 token236751≈-0.1841, row1 token236757≈-0.2357, row2 max≈0.0481
pure flash all layers/positions:
  row0 max≈0.1354, row1 token236757≈-0.2670, row2 token236757≈+0.00051 but row2 token236751≈-0.0820
```

The ggml-style pure flash path is now the default Gemma4 MTP verifier attention path because llama.cpp uses `ggml_flash_attn_ext` and the default accepted-token gate stays green. `GO_PHERENCE_MTP_PURE_FLASH=0` remains as an explicit diagnostic opt-out. Matching the exact AVX `GGML_F32x8_REDUCE` tree in the F16 vecdot port further improved strict selected logits: row2 token236751 is now within tolerance, row1 token236757 shrank to about `-0.1882`, and five residual strict mismatches remain. Layer0/pos2 had been the best bounded diagnostic for row1+row2 before making flash default, but it worsened row0, reinforcing that the remaining issue is verifier-row coupling/state propagation rather than a local attention-only fix.

With the graph-aligned default (Gemma4 no layer-output BF16 store), the layer0/pos3 trace ladder is:

```text
default F32 attention:
  llama l_out-0 row1 vs Go l_out layer0/pos3 max≈0.0121098 mean≈0.00111134

pure flash layer0/pos3:
  llama __fattn__-0 row1 vs Go attn_pre_o max≈0.00059104 mean≈0.000003126 before exact AVX reduction-tree matching; real ggml flash-oracle tests now show layer0 row1 PureRef max≈9.54e-7 against cgo ggml
  llama attn_out-0 row1 vs Go attn_out max≈0.0263062 mean≈0.000401663
  llama l_out-0 row1 vs Go l_out max≈0.00790989 mean≈0.000793528
```

This confirms the BF16 boundary is no longer obscuring trace comparisons; the remaining layer0 local mismatch is now dominated by post-attention composition/MLP propagation, not by the layer-output store.

After adding row-aware production-path trace labels for `attn_post_norm` and `ffn_post_norm`, the saved llama layer0 row1 `ffn_post_norm` dump compares to Go as:

```text
llama ffn_post_norm-0 row1 vs Go ffn_post_norm layer0/pos3:
  max_abs ≈ 0.0592184
  mean_abs ≈ 0.00680059
  top lane idx=355 llama=-0.5546291 go=-0.4954108 delta=+0.0592184
```

This is now a larger local row1 drift than the post-flash `attn_pre_o` residual, so the next first-difference walk should inspect the MLP path between `attn_out` and `ffn_post_norm` (FFN norm, gate/up projection, GEGLU, down projection, FFN post norm) on the current graph-aligned production path.

A refreshed llama trace with broader filters (`attn_norm`, K/V norm/pos, `__fattn__`, `attn_out`, `ffn_norm`, `ffn_out`, `ffn_post_norm`, `l_out`) compared to the current row-aware Go trace gives this layer0 row1 ladder:

```text
attn_norm      max=0                 mean=0
k_norm         max≈8.94e-8           mean≈1.31e-8
k_pos          max≈8.94e-8           mean≈1.36e-8
v_norm         max≈7.15e-7           mean≈1.28e-7
attn_pre_o     max≈0.000590742       mean≈3.08e-6
attn_out       max≈0.0263062         mean≈0.000401663
ffn_norm       max≈0.0011330         mean≈0.0000983
ffn_out        max≈0.0105557         mean≈0.0016610
ffn_post_norm  max≈0.0592184         mean≈0.0068006
l_out          max≈0.0079099         mean≈0.0007935
```

`q_proj`/`q_norm`/`q_pos` were not emitted by that refreshed llama filter despite existing Go row-aware dumps, but historical dumps showed them aligned within ~2e-6. Full-row verifier dump mapping was also clarified: llama's layer0 `Kcur_pos`/`Vcur_normed` `n1536` tensors map to Go verifier positions `(2,3,4)`, not prompt rows, and match Go within `~1.7e-6` max for V and `~1.8e-7` max for K. Llama's full `__fattn__-0` `n6144` dump likewise maps to Go `attn_pre_o` rows `(2,3,4)`:

```text
full __fattn__ rows (2,3,4): max≈0.000590742 mean≈1.49e-6
  row2 max≈0.000554383 mean≈1.15e-6
  row3 max≈0.000590742 mean≈3.08e-6
  row4 max≈0.000053577 mean≈2.25e-7
```

So the current first non-trivial refreshed drift remains a very small `attn_pre_o` residual, and the largest local amplification is visible by `ffn_post_norm`; cache row layout is not the active gap. A real-asset actual-input Q/K/V projection oracle also compares Go `gemvGGUFTo` against ggml `mul_mat` exactly for layer0 row1 after `attn_norm`:

```text
blk.0.attn_q.weight actual input ggml-vs-go max=0 mean=0
blk.0.attn_k.weight actual input ggml-vs-go max=0 mean=0
blk.0.attn_v.weight actual input ggml-vs-go max=0 mean=0
```

Thus any small refreshed `q_proj` dump delta is not a Go Q/K/V projection kernel error. An all-layer row1 `l_out` trace comparison shows the residual accumulates across layers rather than appearing as a single abrupt post-layer0 break:

```text
layer00 max≈0.00791 mean≈0.000794
layer01 max≈0.00837 mean≈0.000628
layer02 max≈0.01996 mean≈0.00225
layer03 max≈0.04304 mean≈0.00437
layer04 max≈0.05171 mean≈0.00529
layer08 max≈0.05936 mean≈0.00386
layer10 max≈0.10193 mean≈0.00695
layer11 max≈0.10535 mean≈0.00963
layer16 max≈0.34935 mean≈0.02875  # peak in this trace
layer20 max≈0.18275 mean≈0.02874
layer24 max≈0.17304 mean≈0.03319
layer32 max≈0.24638 mean≈0.02854
layer40 max≈0.14312 mean≈0.02779
layer41 max≈0.10824 mean≈0.01598
```

A focused row1 layer16 refresh (where the max drift peaks) confirms the drift is inherited into the layer rather than created by one isolated layer16 op:

```text
layer15 l_out -> layer16 input max≈0.236385 mean≈0.0252965
layer16 attn_norm            max≈0.633961 mean≈0.0509786
layer16 k_norm               max≈0.0100745 mean≈0.0015274
layer16 k_pos                max≈0.0100646 mean≈0.0015347
layer16 v_norm               max≈0.122396 mean≈0.0214687
layer16 attn_pre_o           max≈0.0885595 mean≈0.0090426
layer16 attn_out             max≈0.356415 mean≈0.0266757
layer16 ffn_norm             max≈0.0638602 mean≈0.0061099
layer16 ffn_out              max≈0.0086391 mean≈0.0014295
layer16 ffn_post_norm        max≈0.235194 mean≈0.0377614
layer16 l_out                max≈0.349350 mean≈0.0287527
```

Thus layer16 is an amplification point in the accumulated trajectory, not the first local semantic mismatch. It is also a dense `sliding_attention` layer (`hasKV=true`, no MoE/router/expert tensors), so the peak is not a MoE-specific routing/expert issue.

Final-tail comparison with occurrence-indexed llama dumps confirms the strict selected-logit deltas are primarily inherited hidden-state drift, not a new LM-head-only issue. Across all three verifier rows:

```text
row0 result_norm   max≈0.420859 mean≈0.0806129
row0 result_output max≈0.622573 mean≈0.105793
row1 result_norm   max≈0.492623 mean≈0.0854791
row1 result_output max≈0.521339 mean≈0.0991913
row2 result_norm   max≈0.418368 mean≈0.0843934
row2 result_output max≈0.464146 mean≈0.0796333
```

Detailed row1 selected-logit comparison:

```text
result_norm row1:   max≈0.492623 mean≈0.0854791
result_output row1: max≈0.521339 mean≈0.0991913
selected logits:
  token236751 llama=7.3486571 go=7.3860683 delta=+0.0374112
  token236757 llama=13.9382925 go=13.7500830 delta=-0.1882095
  token236789 llama=27.4089108 go=27.4293861 delta=+0.0204754
  token564    llama=13.4591370 go=13.4725132 delta=+0.0133762
```

The LM-head/tail path is therefore not the next isolated kernel target; the hidden trajectory entering final norm is already off. A ggml-tagged strict-fixture oracle now confirms Go's final RMSNorm exactly matches cgo ggml on the actual verifier hidden rows:

```text
strict fixture verifier row0 final norm ggml-vs-go max=0 mean=0
strict fixture verifier row1 final norm ggml-vs-go max=0 mean=0
strict fixture verifier row2 final norm ggml-vs-go max=0 mean=0
selected LM-head probes from the same normalized rows also match cgo ggml exactly (tokens 564, 236751, 236757, 236789; all rows diff=0)
```

A strengthened real layer0 flash oracle now also compares the production pure-Go flash reference against cgo ggml batched `FlashAttnF32F16Batch` on actual verifier rows:

```text
layer0 batched-vs-row ggml flash row0/1/2 max=0
layer0 batched-vs-pure flash row0 max≈1.43e-6 mean≈2.34e-8
layer0 batched-vs-pure flash row1 max≈9.54e-7 mean≈2.44e-9
layer0 batched-vs-pure flash row2 max=0
```

So the pure-Go flash implementation is effectively matching the local cgo ggml oracle; the remaining saved llama trace residual (`~5.9e-4`) is no longer explained by Go-side flash math or K/V cache layout alone.

The gated full-layer batch verifier path was also re-tested after the graph/flash fixes:

```bash
GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS=1 go run ./cmd/models/gemma4mtpparity -fixture tmp/gemma4-mtp-llamacpp-fixture.json
```

It produces the same strict selected-logit deltas as the default sequential row fallback (row1 token236757 about `-0.1882`, row2 token236751 within tolerance). A direct Go trace comparison of row1 `l_out` for all 42 layers also shows sequential fallback and gated full-layer batch outputs are bit-identical (`max overall = 0`). Therefore the current gap is not caused by sequential verifier-row lowering versus the gated full-layer batch scaffold.

A real-asset Go-vs-ggml oracle using the actual layer0 row1 `attn_out` input now covers the FFN kernels. Findings:

```text
ffn_norm: Go RMSNorm matches ggml within 1e-6
ffn_gate / ffn_up / ffn_down Q4 projections: exact against ggml mul_mat for the actual row input
GEGLU: existing ggml table-based Go path is much closer than direct tanh-GELU
  table max≈0.00021246 mean≈2.47e-8
  direct max≈0.00772476 mean≈3.94e-5
ffn_post_norm: Go RMSNorm matches ggml within about 7.6e-6
```

So the FFN kernels themselves are not the active gap; the observed `ffn_post_norm` drift is propagated from the upstream `attn_out` input difference.

`build_cvec` was inspected as a possible row-coupled state source. In llama.cpp it is only `llama_adapter_cvec::apply_to`, which adds a control-vector tensor when one is configured; otherwise `tensor_for(il)` returns `nullptr` and the input tensor is returned unchanged. The local parity harness does not pass `--control-vector`, so control vectors are inactive and Go's lack of an explicit cvec add is not a current parity gap.

The real `per_layer_model_proj` oracle showed that rounding the projection input to F16 matches local cgo ggml more closely than the current F32-input path, but making that the production behavior worsened strict end-to-end selected-logit parity back to six mismatches (`row1 token236757≈-0.2196`, `row2 token236751≈-0.0685`, `row2 token236757≈+0.0133`). This is therefore documented as a local oracle insight, not a safe standalone fix.

Row-aware PLI trace labels matching llama's `inp_per_layer`, `pe_in`, and `per_layer_embd_out` were added. The llama `inp_per_layer` dump emits one 256-wide slice per layer and per graph occurrence; for the matching verifier occurrence, the best mapping is `occ = 42 + layer`, which matches Go per-layer inputs closely:

```text
layer0  inp_per_layer best occ=42 max≈0.0002766 mean≈1.67e-5
layer1  inp_per_layer best occ=43 max≈0.0001190 mean≈1.79e-5
layer2  inp_per_layer best occ=44 max≈0.0003152 mean≈2.18e-5
layer16 inp_per_layer best occ=58 max≈0.0004044 mean≈2.32e-5
layer41 inp_per_layer best occ=83 max≈0.0008812 mean≈2.56e-5
```

For the matching verifier occurrence (`occ1` in the llama PLI branch dump), layer0 row1 compares as:

```text
pe_in              max≈0.0594177 mean≈0.00685975
per_layer_embd_out max≈0.106006  mean≈0.00722661
l_out              max≈0.0079099 mean≈0.00079353
```

This shows the PLI branch amplifies the upstream `pe_in` difference locally, but the final layer residual/scale reduces the max at `l_out`. The PLI matmul/embedding local kernels themselves are covered by ggml oracles, so the branch appears to propagate existing hidden drift rather than introducing a standalone kernel mismatch.

The original, pre-fix layer0 attention drift amplified into downstream layer0 checkpoints:

- layer 0 `attn_out` max abs was about `0.1384`
- layer 0 `l_out` max abs was about `0.0275`

After the graph/flash fixes, layer0 row1 `attn_pre_o` is within about `5.9e-4` of the saved llama dump and within `1e-5` of the local cgo ggml oracle on real rows; the larger strict-logit gap is now explained by accumulated hidden-state trajectory drift across layers.

## Interpretation

The current strict gap is **not** explained by:

- tokenizer/prompt trimming
- Q/K/V/O/FFN projections or quant matmuls (actual-row ggml oracles pass)
- per-layer token embedding rows, `inp_per_layer` row construction, PLI gate/proj matmuls, or local PLI projection kernels
- RMSNorm / no-scale RMSNorm / final norm (actual-row ggml oracles pass)
- RoPE ordering/factors for the traced rows
- GEGLU local kernel (table path is the correct ggml mode)
- tail LM head / softcap / suppress-token handling (actual-row ggml probes pass)
- shared-KV source mapping
- SWA mask ranges
- verifier sequential-row fallback vs gated full-layer batch lowering
- cvec/control vectors (inactive in the local harness)
- MoE routing at the observed layer16 drift peak (layer16 is dense sliding attention)

The remaining problem is an accumulated hidden-trajectory mismatch: small residual differences are present from the first attention block, then compound through dense layers and are visible at final `result_norm`. Local replacements that improve a single oracle (e.g. F16-rounded per-layer projection input) can worsen the end-to-end strict fixture, so future fixes should be validated against the full strict selected-logit gate, not only one local tensor distance.
