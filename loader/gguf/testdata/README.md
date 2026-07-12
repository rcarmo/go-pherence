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
