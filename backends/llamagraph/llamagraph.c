// llamagraph.c — Full GGML LLaMA decode graph driven by go-pherence GGUF loader.
// Uses libggml-base directly; mirrors llama.cpp's internal graph builder.

#include "llamagraph.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"







// ---------------------------------------------------------------------------
// Init — creates all tensor shapes; call setters afterwards.
// ---------------------------------------------------------------------------
gpll_model *gpll_init(
        int n_vocab, int n_embd, int n_heads, int n_heads_kv,
        int n_layers, int n_ff, int n_ctx,
        float rope_base, float rms_eps, int rope_dims,
        int n_threads, gpll_weight_types *wt)
{
    gpll_model *m = (gpll_model *)calloc(1, sizeof(gpll_model));
    m->n_vocab     = n_vocab;
    m->n_embd      = n_embd;
    m->n_heads     = n_heads;
    m->n_heads_kv  = n_heads_kv;
    m->n_layers    = n_layers;
    m->n_ff        = n_ff;
    m->n_ctx       = n_ctx;
    m->rope_base   = rope_base;
    m->rms_eps     = rms_eps;
    m->rope_dims   = rope_dims;
    m->n_embd_head = n_embd / n_heads;
    m->n_threads   = n_threads;
    m->n_past      = 0;

    ggml_backend_load_all();
    m->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(m->backend, n_threads);

    // --- Weight tensors (no_alloc=true; allocated below) ---
    int n_tensors = 3 + n_layers * 8;
    size_t ctx_wsize = ggml_tensor_overhead() * n_tensors * 2 + 65536;
    struct ggml_init_params wp = { ctx_wsize, NULL, true };
    m->ctx_weights = ggml_init(wp);

    int n_embd_kv = m->n_embd_head * n_heads_kv;

    m->tok_embd    = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->tok_embd, n_embd, n_vocab);
    m->output_norm = ggml_new_tensor_1d(m->ctx_weights, GGML_TYPE_F32, n_embd);
    m->output      = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->output,   n_embd, n_vocab);

    for (int il = 0; il < n_layers; il++) {
        gpll_layer *l = &m->layers[il];
        l->attn_norm = ggml_new_tensor_1d(m->ctx_weights, GGML_TYPE_F32, n_embd);
        l->wq        = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wq[il],       n_embd, n_embd);
        l->wk        = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wk[il],       n_embd, n_embd_kv);
        l->wv        = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wv[il],       n_embd, n_embd_kv);
        l->wo        = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wo[il],       n_embd, n_embd);
        l->ffn_norm  = ggml_new_tensor_1d(m->ctx_weights, GGML_TYPE_F32, n_embd);
        l->ffn_gate  = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_gate[il], n_embd, n_ff);
        l->ffn_up    = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_up[il],   n_embd, n_ff);
        l->ffn_down  = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_down[il], n_ff,   n_embd);
    }

    m->buf_weights = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);

    // --- KV cache (F32, no_alloc=true; allocated below) ---
    size_t ctx_kvsize = ggml_tensor_overhead() * n_layers * 2 + 256;
    struct ggml_init_params kp = { ctx_kvsize, NULL, true };
    m->ctx_kv = ggml_init(kp);

    for (int il = 0; il < n_layers; il++) {
        m->k_cache[il] = ggml_new_tensor_3d(m->ctx_kv, GGML_TYPE_F32,
                             m->n_embd_head, n_ctx, n_heads_kv);
        m->v_cache[il] = ggml_new_tensor_3d(m->ctx_kv, GGML_TYPE_F32,
                             m->n_embd_head, n_ctx, n_heads_kv);
    }
    m->buf_kv = ggml_backend_alloc_ctx_tensors(m->ctx_kv, m->backend);

    return m;
}

// ---------------------------------------------------------------------------
// Weight setters — each just copies bytes into the pre-allocated tensor.
// ---------------------------------------------------------------------------
void gpll_set_tok_embd   (gpll_model *m, const void *d, size_t n) { ggml_backend_tensor_set(m->tok_embd,    d, 0, n); }
void gpll_set_output_norm(gpll_model *m, const void *d, size_t n) { ggml_backend_tensor_set(m->output_norm, d, 0, n); }
void gpll_set_output     (gpll_model *m, const void *d, size_t n) { ggml_backend_tensor_set(m->output,      d, 0, n); }

void gpll_set_attn_norm (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].attn_norm, d, 0, n); }
void gpll_set_wq        (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].wq,        d, 0, n); }
void gpll_set_wk        (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].wk,        d, 0, n); }
void gpll_set_wv        (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].wv,        d, 0, n); }
void gpll_set_wo        (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].wo,        d, 0, n); }
void gpll_set_ffn_norm  (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].ffn_norm,  d, 0, n); }
void gpll_set_ffn_gate  (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].ffn_gate,  d, 0, n); }
void gpll_set_ffn_up    (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].ffn_up,    d, 0, n); }
void gpll_set_ffn_down  (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].ffn_down,  d, 0, n); }

// ---------------------------------------------------------------------------
// Decode — build and run a full single-token GGML graph
// ---------------------------------------------------------------------------
int gpll_decode(gpll_model *m, int token_id, float *out_logits) {
    int pos = m->n_past;

    size_t scratch_size = 512ULL * 1024 * 1024;
    void *scratch_mem = malloc(scratch_size);
    if (!scratch_mem) return -1;
    struct ggml_init_params gp = { scratch_size, scratch_mem, false };
    struct ggml_context *ctx = ggml_init(gp);
    if (!ctx) { free(scratch_mem); return -1; }

    struct ggml_tensor *pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor *tok_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    int32_t tok32 = (int32_t)token_id, pos32 = (int32_t)pos;
    memcpy(tok_t->data, &tok32, sizeof(int32_t));
    memcpy(pos_t->data, &pos32, sizeof(int32_t));

    struct ggml_tensor *x = ggml_get_rows(ctx, m->tok_embd, tok_t);
    x = ggml_reshape_1d(ctx, x, m->n_embd);

    float attn_scale = 1.0f / sqrtf((float)m->n_embd_head);
    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16384, false);

    // k_cache[il] layout: F32 [head_dim, n_ctx, n_kv_heads]
    //   nb[0]=4, nb[1]=head_dim*4, nb[2]=n_ctx*head_dim*4
    // element at [d, s, h] = base + d*4 + s*head_dim*4 + h*n_ctx*head_dim*4
    size_t hf     = (size_t)m->n_embd_head * sizeof(float);
    size_t hf_ctx = hf * (size_t)m->n_ctx;   // nb[2] of k_cache (stride per kv head)

    for (int il = 0; il < m->n_layers; il++) {
        gpll_layer *l = &m->layers[il];

        struct ggml_tensor *xn = ggml_rms_norm(ctx, x, m->rms_eps);
        xn = ggml_mul(ctx, xn, l->attn_norm);

        struct ggml_tensor *q = ggml_mul_mat(ctx, l->wq, xn);
        struct ggml_tensor *k = ggml_mul_mat(ctx, l->wk, xn);
        struct ggml_tensor *v = ggml_mul_mat(ctx, l->wv, xn);

        // ggml_rope_ext asserts a->ne[2] == b->ne[0] (n_tokens == pos_count)
        // So RoPE shape: [head_dim, n_heads, n_tokens=1]
        q = ggml_reshape_3d(ctx, q, m->n_embd_head, m->n_heads,    1);
        k = ggml_reshape_3d(ctx, k, m->n_embd_head, m->n_heads_kv, 1);
        v = ggml_reshape_3d(ctx, v, m->n_embd_head, m->n_heads_kv, 1);

        q = ggml_rope_ext(ctx, q, pos_t, NULL, m->rope_dims,
                          GGML_ROPE_TYPE_NORMAL, m->n_ctx,
                          m->rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(ctx, k, pos_t, NULL, m->rope_dims,
                          GGML_ROPE_TYPE_NORMAL, m->n_ctx,
                          m->rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // flash_attn_ext asserts ggml_can_mul_mat(k, q):
        //   q.ne[2] % k.ne[2] == 0  →  n_heads % n_kv_heads == 0
        // So flash_attn shape: q=[head_dim, n_tokens=1, n_heads]
        // For single-token (ne[2]=1 in RoPE shape ↔ ne[1]=1 in FA shape):
        //   contiguous memory is identical, reshape is free.
        q = ggml_reshape_3d(ctx, q, m->n_embd_head, 1, m->n_heads);

        // KV cache write: k has shape [head_dim, n_kv_heads, 1] (contiguous)
        // k_slot: view of k_cache at seq position pos, same shape [head_dim, n_kv_heads, 1]
        //   nb[1] = stride per kv head = hf_ctx  (heads are separated by full sequence)
        //   nb[2] = arbitrary (ne[2]=1)
        //   offset = pos * hf  (skip to position pos)
        struct ggml_tensor *k_slot = ggml_view_3d(ctx, m->k_cache[il],
            m->n_embd_head, m->n_heads_kv, 1,
            hf_ctx,                   // nb[1]: stride per kv head
            hf_ctx * m->n_heads_kv,   // nb[2]: unused (ne[2]=1)
            hf * (size_t)pos);        // offset: position pos

        struct ggml_tensor *v_slot = ggml_view_3d(ctx, m->v_cache[il],
            m->n_embd_head, m->n_heads_kv, 1,
            hf_ctx, hf_ctx * m->n_heads_kv, hf * (size_t)pos);

        // Add KV writes to graph FIRST so they execute before flash_attn reads
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_slot));

        // KV cache read: [head_dim, pos+1, n_kv_heads]
        //   nb[1] = hf (stride per seq pos, contiguous in that dimension)
        //   nb[2] = hf_ctx (stride per kv head, non-contiguous: skips full sequence)
        struct ggml_tensor *k_used = ggml_view_3d(ctx, m->k_cache[il],
            m->n_embd_head, pos + 1, m->n_heads_kv,
            hf, hf_ctx, 0);
        struct ggml_tensor *v_used = ggml_view_3d(ctx, m->v_cache[il],
            m->n_embd_head, pos + 1, m->n_heads_kv,
            hf, hf_ctx, 0);

        // flash_attn_ext: q=[head_dim,1,n_heads], k/v=[head_dim,pos+1,n_kv_heads]
        // ggml_can_mul_mat(k_used, q): q.ne[2]=n_heads % k.ne[2]=n_kv_heads == 0 ✓
        struct ggml_tensor *attn = ggml_flash_attn_ext(ctx,
            q, k_used, v_used, NULL, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_DEFAULT);

        // attn out: [head_dim, 1, n_heads] → reshape to [n_embd]
        attn = ggml_reshape_1d(ctx, attn, m->n_embd);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, l->wo, attn));

        xn = ggml_rms_norm(ctx, x, m->rms_eps);
        xn = ggml_mul(ctx, xn, l->ffn_norm);
        struct ggml_tensor *gate = ggml_silu(ctx, ggml_mul_mat(ctx, l->ffn_gate, xn));
        struct ggml_tensor *up   = ggml_mul_mat(ctx, l->ffn_up, xn);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, l->ffn_down,
                                           ggml_mul(ctx, gate, up)));
    }

    x = ggml_rms_norm(ctx, x, m->rms_eps);
    x = ggml_mul(ctx, x, m->output_norm);
    struct ggml_tensor *logits = ggml_mul_mat(ctx, m->output, x);
    ggml_build_forward_expand(gf, logits);

    struct ggml_cplan plan = ggml_graph_plan(gf, m->n_threads, 0);
    void *work_buf = NULL;
    if (plan.work_size > 0) {
        work_buf = malloc(plan.work_size);
        if (!work_buf) { ggml_free(ctx); free(scratch_mem); return -1; }
        plan.work_data = (uint8_t *)work_buf;
    }

    int rc = ggml_graph_compute(gf, &plan);

    if (rc == GGML_STATUS_SUCCESS) {
        memcpy(out_logits, logits->data, (size_t)m->n_vocab * sizeof(float));
        m->n_past++;
    }

    if (work_buf) free(work_buf);
    ggml_free(ctx);
    free(scratch_mem);
    return rc == GGML_STATUS_SUCCESS ? 0 : -rc;
}

void gpll_reset(gpll_model *m) { m->n_past = 0; }

void gpll_free(gpll_model *m) {
    if (!m) return;
    if (m->buf_weights) ggml_backend_buffer_free(m->buf_weights);
    if (m->ctx_weights) ggml_free(m->ctx_weights);
    if (m->buf_kv)      ggml_backend_buffer_free(m->buf_kv);
    if (m->ctx_kv)      ggml_free(m->ctx_kv);
    if (m->backend)     ggml_backend_free(m->backend);
    free(m);
}
