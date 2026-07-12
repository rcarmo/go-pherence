# Q4_K 8×8 ggml CPU oracle

`actual_ggml_q4k_oracle_*` is a deterministic external fixture for the tuned
llama.cpp CPU repack path at revision
`4a6735f1cf0594250958bcc839267696c7b998a4`.

The generator created canonical `Q4_K [2816,8]` weights and eight deterministic
F32 activation rows, then executed a real ggml graph:

1. `ggml_new_tensor_2d(..., GGML_TYPE_Q4_K, 2816, 8)`
2. allocation with `ggml_backend_cpu_repack_buffer_type()`
3. `ggml_mul_mat(q4, f32)` on a one-thread CPU backend
4. `ggml_backend_graph_compute()` and tensor download

The runtime printed:

```text
repack: repack tensor q4_K_8x8 with q4_K_8x8
max repack-vs-vecdot: 3.69548798e-06
```

It was built against the reference tree with:

```sh
g++ -std=c++17 -O2 tmp_actual_ggml_q4k_oracle.cpp \
  -I/workspace/projects/llama.cpp/ggml/include \
  -L/workspace/projects/llama.cpp/build/bin \
  -Wl,-rpath,/workspace/projects/llama.cpp/build/bin \
  -lggml -lggml-base -lggml-cpu -o tmp_actual_ggml_q4k_oracle
./tmp_actual_ggml_q4k_oracle actual_ggml_q4k_oracle.json
```

The JSON preserves both downloaded `mul_mat` output and independently repeated
vec-dot output. Binary files are the exact canonical tensor bytes supplied to
ggml, avoiding a self-derived Go input generator.

SHA-256:

```text
a447abf67ea54c9037b7741ca47719a06b2985ebe14375744f5f6197c2a90f68  actual_ggml_q4k_oracle.json
54475e48cc0c749d061abe921bafb5d87556e84428b0b432bc5c12c168a0dab5  actual_ggml_q4k_oracle_act_f32.bin
44a7c910bf29d2a245e045b5f52e0ba5207d3cc822bf7b2c102508b77d35dd26  actual_ggml_q4k_oracle_q4.bin
```

## Q4_K 8×8 single-column GEMV oracle

`actual_ggml_q4k_gemv_oracle_*` covers the repacked GEMV selected by
`MUL_MAT_ID` for each expert assignment at the same pinned revision. The
external generator created canonical `Q4_K [2816,8]` weights and one F32
activation column, allocated weights with
`ggml_backend_cpu_repack_buffer_type()`, executed a real one-thread
`ggml_mul_mat`, and cross-checked the repacked bytes through
`ggml_gemv_q4_K_8x8_q8_K()`.

The runtime reported repacking, a `6.67572021e-06` graph-versus-scalar-vecdot
gap, and zero graph-versus-direct-GEMV gap. The generator command was:

```sh
g++ -std=c++17 -O2 /workspace/tmp_actual_ggml_q4k_gemv_oracle.cpp \
  -I/workspace/projects/llama.cpp/ggml/include \
  -I/workspace/projects/llama.cpp/ggml/src \
  -L/workspace/projects/llama.cpp/build/bin \
  -Wl,-rpath,/workspace/projects/llama.cpp/build/bin \
  -lggml -lggml-base -lggml-cpu -o /workspace/tmp_actual_ggml_q4k_gemv_oracle
GGML_LOG_LEVEL=debug /workspace/tmp_actual_ggml_q4k_gemv_oracle \
  /workspace/actual_ggml_q4k_gemv_oracle.json
```

SHA-256:

```text
8538f71a56388127949f0f461cbd7331e93cdfeb34bc8c5aa7e5afa8a377ab50  actual_ggml_q4k_gemv_oracle.json
81380616b3080c906eaaf0cff1e3cef7849e8b8aa817eaa8715cb43c1ed7d939  actual_ggml_q4k_gemv_oracle_act_f32.bin
367ca4b78f0e4fa69d933fec0c7c938f63a8a02ba3abf42fc50750aa170c2efb  actual_ggml_q4k_gemv_oracle_q4.bin
```

## Q4_K 8x8 MUL_MAT_ID oracle

`actual_ggml_mul_mat_id_oracle_*` is a deterministic external fixture for the
real one-thread CPU `ggml_mul_mat_id` path at the same pinned llama.cpp
revision `4a6735f1cf0594250958bcc839267696c7b998a4`.

The standalone generator built canonical `Q4_K [256,8,4]` expert weights,
canonical `F32 [256,2,3]` source activations, and `I32 [2,3]` ids with
repeated and out-of-order expert selection:

- token 0 -> slots `[2, 0]`
- token 1 -> slots `[1, 2]`
- token 2 -> slots `[2, 1]`

It then executed:

1. `ggml_new_tensor_3d(..., GGML_TYPE_Q4_K, 256, 8, 4)` for expert weights
2. `ggml_new_tensor_3d(..., GGML_TYPE_F32, 256, 2, 3)` for `src1`
3. `ggml_new_tensor_2d(..., GGML_TYPE_I32, 2, 3)` for ids
4. allocation with `ggml_backend_cpu_repack_buffer_type()`
5. `ggml_mul_mat_id(as, b, ids)` on a one-thread CPU backend
6. `ggml_backend_graph_compute()` and tensor download

The JSON records provenance plus the contiguous tensor layouts used by both
GGML and the Go reproduction test:

- `src0_as`: canonical raw Q4_K bytes laid out as `[expert][out_row][row_bytes]`
- `src1_b`: F32 contiguous `[token][slot][k]` matching ggml `ne=[k,n_ids,tokens]`
- `ids`: I32 contiguous `[token][slot]` with slot as `ne0`
- `dst`: F32 contiguous `[token][slot][out_row]` matching ggml `ne=[out,n_ids,tokens]`

Row grouping provenance matches `forward_mul_mat_id`: scan tokens outermost,
slots innermost, append each `(slot, token)` pair to the selected expert group,
run each expert matmul, then scatter back to `[slot, token]` in the output.

Generator command:

```sh
g++ -std=c++17 -O2 /workspace/tmp_actual_ggml_mul_mat_id_oracle.cpp \
  -I/workspace/projects/llama.cpp/ggml/include \
  -I/workspace/projects/llama.cpp/ggml/src \
  -L/workspace/projects/llama.cpp/build/bin \
  -Wl,-rpath,/workspace/projects/llama.cpp/build/bin \
  -lggml -lggml-base -lggml-cpu -o /workspace/tmp_actual_ggml_mul_mat_id_oracle
GGML_LOG_LEVEL=debug /workspace/tmp_actual_ggml_mul_mat_id_oracle \
  /workspace/actual_ggml_mul_mat_id_oracle.json
```

The runtime reported the repack log `repack tensor q4_moe with q4_K_8x8` and a
`1.90734863e-06` max graph-versus-independent-vecdot gap.

SHA-256:

```text
23c1ddfe1d8399078c33ae836f8834799524195cc34f5e25bf454c5e11f3ed5a  actual_ggml_mul_mat_id_oracle.json
e5161bb529a64c999ca375178a1fa1356b72e0f65d96f348b86477d289401783  actual_ggml_mul_mat_id_oracle_q4.bin
b662aca276d6c93ed2887b0c18960bff37063bdcf9f009fffabe1b8306cf0a58  actual_ggml_mul_mat_id_oracle_act_f32.bin
778b5a2b0d0fcbb14612f50be317c35b1c97dd05efd61846830372ad45fb5365  actual_ggml_mul_mat_id_oracle_ids_i32.bin
```

## Actual ggml MoE micrograph oracle

`actual_ggml_moe_micrograph_oracle_*` records a one-thread CPU graph at llama.cpp
revision `4a6735f1cf0594250958bcc839267696c7b998a4`. It covers repacked Q4_K
`mul_mat_id` gate/up projection, strided gate/up views, GELU multiplication,
Q8_0 down `mul_mat_id`, selected-expert weighting, and slot reduction for four
experts, two selected slots, and three tokens. The Go fixture validates the
materialized projection and downstream boundaries; it intentionally does not
download the non-contiguous gate/up views as if they were contiguous tensors.

SHA-256:

```text
4ca669bafd90869d821e50e790b87f8f449fd9aa1d67c79925f453f4913a9634  actual_ggml_moe_micrograph_oracle.json
cd97828ed241275e633df0db846650143e29f34e29986636b1c59da4aec61013  actual_ggml_moe_micrograph_oracle_down_q8.bin
f45f42d085cdbff4c8b4e2098b553c69a4aa6beb1f61357c3332270ad1aefd83  actual_ggml_moe_micrograph_oracle_gate_up_q4.bin
778b5a2b0d0fcbb14612f50be317c35b1c97dd05efd61846830372ad45fb5365  actual_ggml_moe_micrograph_oracle_ids_i32.bin
f4f79eb962be82a1a37074892be168a195540b90755bffad00f56ab3ddc69f53  actual_ggml_moe_micrograph_oracle_selected_weights_f32.bin
0e3548ac8a63c2072c49f58e26404cb47453c38e14bbc2177c0b8542d2c44f8f  actual_ggml_moe_micrograph_oracle_src1_f32.bin
```
