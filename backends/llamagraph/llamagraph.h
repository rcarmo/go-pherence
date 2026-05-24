// llamagraph.h — go-pherence GGML LLaMA decode graph (full declarations)
#pragma once

#include <stddef.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#define GPLL_MAX_LAYERS 128

typedef struct {
    int tok_embd;
    int output;
    int wq[GPLL_MAX_LAYERS];
    int wk[GPLL_MAX_LAYERS];
    int wv[GPLL_MAX_LAYERS];
    int wo[GPLL_MAX_LAYERS];
    int ffn_gate[GPLL_MAX_LAYERS];
    int ffn_up[GPLL_MAX_LAYERS];
    int ffn_down[GPLL_MAX_LAYERS];
} gpll_weight_types;

typedef struct {
    struct ggml_tensor *attn_norm;
    struct ggml_tensor *wq, *wk, *wv, *wo;
    struct ggml_tensor *ffn_norm;
    struct ggml_tensor *ffn_gate, *ffn_up, *ffn_down;
} gpll_layer;

typedef struct gpll_model {
    int n_vocab, n_embd, n_heads, n_heads_kv, n_layers, n_ff, n_ctx;
    float rope_base, rms_eps;
    int rope_dims, n_embd_head;
    int n_threads;

    ggml_backend_t          backend;
    struct ggml_context    *ctx_weights;
    ggml_backend_buffer_t   buf_weights;
    struct ggml_tensor     *tok_embd;
    struct ggml_tensor     *output_norm;
    struct ggml_tensor     *output;
    gpll_layer              layers[GPLL_MAX_LAYERS];

    struct ggml_context    *ctx_kv;
    ggml_backend_buffer_t   buf_kv;
    struct ggml_tensor     *k_cache[GPLL_MAX_LAYERS];
    struct ggml_tensor     *v_cache[GPLL_MAX_LAYERS];

    int n_past;
} gpll_model;

gpll_model *gpll_init(
        int n_vocab, int n_embd, int n_heads, int n_heads_kv,
        int n_layers, int n_ff, int n_ctx,
        float rope_base, float rms_eps, int rope_dims,
        int n_threads, gpll_weight_types *wt);

void gpll_set_tok_embd   (gpll_model *m, const void *d, size_t n);
void gpll_set_output_norm(gpll_model *m, const void *d, size_t n);
void gpll_set_output     (gpll_model *m, const void *d, size_t n);
void gpll_set_attn_norm  (gpll_model *m, int il, const void *d, size_t n);
void gpll_set_wq         (gpll_model *m, int il, const void *d, size_t n);
void gpll_set_wk         (gpll_model *m, int il, const void *d, size_t n);
void gpll_set_wv         (gpll_model *m, int il, const void *d, size_t n);
void gpll_set_wo         (gpll_model *m, int il, const void *d, size_t n);
void gpll_set_ffn_norm   (gpll_model *m, int il, const void *d, size_t n);
void gpll_set_ffn_gate   (gpll_model *m, int il, const void *d, size_t n);
void gpll_set_ffn_up     (gpll_model *m, int il, const void *d, size_t n);
void gpll_set_ffn_down   (gpll_model *m, int il, const void *d, size_t n);

int  gpll_decode(gpll_model *m, int token_id, float *out_logits);
void gpll_reset (gpll_model *m);
void gpll_free  (gpll_model *m);
