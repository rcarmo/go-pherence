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
