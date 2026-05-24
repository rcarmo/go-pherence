// llamagraph.c — GGML LLaMA/Qwen3 decode graph with SpacemiT IME2 repack.
#include "llamagraph.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef ggml_backend_buffer_type_t * (*get_extra_bufts_fn)(ggml_backend_dev_t);

gpll_model *gpll_init(
        int n_vocab, int n_embd, int n_heads, int n_heads_kv,
        int n_layers, int n_ff, int n_ctx,
        float rope_base, float rms_eps, int rope_dims,
        int n_threads, int has_qk_norm, gpll_weight_types *wt)
{
    gpll_model *m = (gpll_model *)calloc(1, sizeof(gpll_model));
    m->n_vocab = n_vocab; m->n_embd = n_embd; m->n_heads = n_heads;
    m->n_heads_kv = n_heads_kv; m->n_layers = n_layers; m->n_ff = n_ff;
    m->n_ctx = n_ctx; m->rope_base = rope_base; m->rms_eps = rms_eps;
    m->rope_dims = rope_dims; m->n_threads = n_threads; m->n_past = 0;
    m->has_qk_norm = has_qk_norm;
    int wq0_out = wt->wq_out[0] > 0 ? wt->wq_out[0] : n_embd;
    m->n_embd_head = wq0_out / n_heads;
    int n_embd_kv = m->n_embd_head * n_heads_kv;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    m->backend = ggml_backend_dev_init(dev, NULL);
    ggml_backend_cpu_set_n_threads(m->backend, n_threads);

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    get_extra_bufts_fn get_extras = (get_extra_bufts_fn)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    m->repack_buft = NULL;
    if (get_extras) { 
        ggml_backend_buffer_type_t *extras = get_extras(dev);
        if (extras && extras[0]) m->repack_buft = extras[0];
    }

    // --- Repacked weight tensors (QKV + FFN matmuls) ---
    int n_mm = n_layers * 7;
    size_t mm_size = ggml_tensor_overhead() * (n_mm + 4) * 2 + 65536;
    struct ggml_init_params mmp = { mm_size, NULL, true };
    m->ctx_weights = ggml_init(mmp);

    for (int il = 0; il < n_layers; il++) {
        gpll_layer *l = &m->layers[il];
        int qo = wt->wq_out[il] > 0 ? wt->wq_out[il] : n_embd;
        int ko = wt->wk_out[il] > 0 ? wt->wk_out[il] : n_embd_kv;
        int vo = wt->wv_out[il] > 0 ? wt->wv_out[il] : n_embd_kv;
        int wi = wt->wo_in[il]  > 0 ? wt->wo_in[il]  : n_embd;
        l->wq       = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wq[il], n_embd, qo);
        l->wk       = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wk[il], n_embd, ko);
        l->wv       = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wv[il], n_embd, vo);
        l->wo       = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wo[il], wi, n_embd);
        l->ffn_gate = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_gate[il], n_embd, n_ff);
        l->ffn_up   = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_up[il], n_embd, n_ff);
        l->ffn_down = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_down[il], n_ff, n_embd);
    }
    if (m->repack_buft)
        m->buf_weights = ggml_backend_alloc_ctx_tensors_from_buft(m->ctx_weights, m->repack_buft);
    else
        m->buf_weights = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);

    // --- Plain tensors (embeddings, norms) ---
    int n_pl = 3 + n_layers * 4;
    size_t pl_size = ggml_tensor_overhead() * n_pl * 2 + 65536;
    struct ggml_init_params plp = { pl_size, NULL, true };
    struct ggml_context *ctx_plain = ggml_init(plp);
    m->tok_embd    = ggml_new_tensor_2d(ctx_plain, (enum ggml_type)wt->tok_embd, n_embd, n_vocab);
    m->output      = ggml_new_tensor_2d(ctx_plain, (enum ggml_type)wt->output, n_embd, n_vocab);
    m->output_norm = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, n_embd);
    for (int il = 0; il < n_layers; il++) {
        gpll_layer *l = &m->layers[il];
        l->attn_norm = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, n_embd);
        l->ffn_norm  = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, n_embd);
        if (has_qk_norm) {
            l->q_norm = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, m->n_embd_head);
            l->k_norm = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, m->n_embd_head);
        }
    }
    m->buf_plain = ggml_backend_alloc_ctx_tensors_from_buft(ctx_plain, ggml_backend_cpu_buffer_type());

    // --- KV cache (F32) ---
    size_t kv_size = ggml_tensor_overhead() * n_layers * 2 + 256;
    struct ggml_init_params kvp = { kv_size, NULL, true };
    m->ctx_kv = ggml_init(kvp);
    for (int il = 0; il < n_layers; il++) {
        m->k_cache[il] = ggml_new_tensor_2d(m->ctx_kv, GGML_TYPE_F32, n_embd_kv, n_ctx);
        m->v_cache[il] = ggml_new_tensor_2d(m->ctx_kv, GGML_TYPE_F32, n_embd_kv, n_ctx);
    }
    m->buf_kv = ggml_backend_alloc_ctx_tensors_from_buft(m->ctx_kv, ggml_backend_cpu_buffer_type());

    return m;
}

// --- Setters ---
void gpll_set_tok_embd   (gpll_model *m, const void *d, size_t n) { ggml_backend_tensor_set(m->tok_embd, d, 0, n); }
void gpll_set_output_norm(gpll_model *m, const void *d, size_t n) { ggml_backend_tensor_set(m->output_norm, d, 0, n); }
void gpll_set_output     (gpll_model *m, const void *d, size_t n) { ggml_backend_tensor_set(m->output, d, 0, n); }
void gpll_set_attn_norm (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].attn_norm, d, 0, n); }
void gpll_set_wq        (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].wq, d, 0, n); }
void gpll_set_wk        (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].wk, d, 0, n); }
void gpll_set_wv        (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].wv, d, 0, n); }
void gpll_set_wo        (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].wo, d, 0, n); }
void gpll_set_ffn_norm  (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].ffn_norm, d, 0, n); }
void gpll_set_ffn_gate  (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].ffn_gate, d, 0, n); }
void gpll_set_ffn_up    (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].ffn_up, d, 0, n); }
void gpll_set_ffn_down  (gpll_model *m, int il, const void *d, size_t n) { ggml_backend_tensor_set(m->layers[il].ffn_down, d, 0, n); }
void gpll_set_q_norm    (gpll_model *m, int il, const void *d, size_t n) { if(m->layers[il].q_norm) ggml_backend_tensor_set(m->layers[il].q_norm, d, 0, n); }
void gpll_set_k_norm    (gpll_model *m, int il, const void *d, size_t n) { if(m->layers[il].k_norm) ggml_backend_tensor_set(m->layers[il].k_norm, d, 0, n); }
void gpll_set_has_qk_norm(gpll_model *m, int v) { m->has_qk_norm = v; }
void gpll_tie_output_embeddings(gpll_model *m) { m->output = m->tok_embd; }
void gpll_set_mtp_enorm (gpll_model *m, const void *d, size_t n) { (void)m;(void)d;(void)n; }
void gpll_set_mtp_hnorm (gpll_model *m, const void *d, size_t n) { (void)m;(void)d;(void)n; }
void gpll_set_mtp_eh_proj(gpll_model *m, const void *d, size_t n) { (void)m;(void)d;(void)n; }
void gpll_set_mtp_shared_head_norm(gpll_model *m, const void *d, size_t n) { (void)m;(void)d;(void)n; }




// --- Decode ---
int gpll_decode(gpll_model *m, int token_id, float *out_logits) {
    if (!m->scratch_mem) {
        m->scratch_size = 1024ULL * 1024 * 1024;
        m->scratch_mem = malloc(m->scratch_size);
        if (!m->scratch_mem) return -1;
    }
    int pos = m->n_past;

    int n_embd_kv = m->n_embd_head * m->n_heads_kv;
    int n_q_embd = m->n_embd_head * m->n_heads;
    size_t hf   = (size_t)m->n_embd_head * sizeof(float);
    size_t hfkv = (size_t)n_embd_kv * sizeof(float);

    if (m->ctx_compute) ggml_free(m->ctx_compute);
    struct ggml_init_params gp = { m->scratch_size, m->scratch_mem, false };
    m->ctx_compute = ggml_init(gp);
    struct ggml_context *ctx = m->ctx_compute;

    struct ggml_tensor *pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor *tok_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    int32_t tok32 = (int32_t)token_id, pos32 = (int32_t)pos;
    memcpy(tok_t->data, &tok32, sizeof(int32_t));
    memcpy(pos_t->data, &pos32, sizeof(int32_t));

    struct ggml_tensor *x = ggml_get_rows(ctx, m->tok_embd, tok_t);
    x = ggml_reshape_1d(ctx, x, m->n_embd);
    float attn_scale = 1.0f / sqrtf((float)m->n_embd_head);
    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16384, false);

    int max_layers = m->n_layers;
    for (int il = 0; il < max_layers; il++) {
        gpll_layer *l = &m->layers[il];
        struct ggml_tensor *xn = ggml_rms_norm(ctx, x, m->rms_eps);
        xn = ggml_mul(ctx, xn, l->attn_norm);

        struct ggml_tensor *q = ggml_mul_mat(ctx, l->wq, xn);
        struct ggml_tensor *k = ggml_mul_mat(ctx, l->wk, xn);
        struct ggml_tensor *v = ggml_mul_mat(ctx, l->wv, xn);

        q = ggml_reshape_3d(ctx, q, m->n_embd_head, m->n_heads, 1);
        k = ggml_reshape_3d(ctx, k, m->n_embd_head, m->n_heads_kv, 1);
        v = ggml_reshape_3d(ctx, v, m->n_embd_head, m->n_heads_kv, 1);

        if (m->has_qk_norm && l->q_norm && l->k_norm) {
            q = ggml_rms_norm(ctx, q, m->rms_eps);
            q = ggml_mul(ctx, q, l->q_norm);
            k = ggml_rms_norm(ctx, k, m->rms_eps);
            k = ggml_mul(ctx, k, l->k_norm);
        }

        q = ggml_rope_ext(ctx, q, pos_t, NULL, m->rope_dims,
                          GGML_ROPE_TYPE_NORMAL, m->n_ctx,
                          m->rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(ctx, k, pos_t, NULL, m->rope_dims,
                          GGML_ROPE_TYPE_NORMAL, m->n_ctx,
                          m->rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // q for flash_attn: [head_dim, 1, n_heads]
        q = ggml_reshape_3d(ctx, q, m->n_embd_head, 1, m->n_heads);

        // KV cache: use direct k/v for attention (pos=0) or cache (pos>0)
        // Flatten k/v for cache write
        struct ggml_tensor *k_flat = ggml_reshape_1d(ctx, k, n_embd_kv);
        struct ggml_tensor *v_flat = ggml_reshape_1d(ctx, v, n_embd_kv);

        // Always write to cache (side-effect nodes)
        struct ggml_tensor *k_slot = ggml_view_1d(ctx, m->k_cache[il], n_embd_kv, hfkv * (size_t)pos);
        struct ggml_tensor *v_slot = ggml_view_1d(ctx, m->v_cache[il], n_embd_kv, hfkv * (size_t)pos);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k_flat, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v_flat, v_slot));

        // For attention: pos=0 uses direct k/v; pos>0 reads full cache
        struct ggml_tensor *k_attn, *v_attn;
        if (pos == 0) {
            k_attn = ggml_reshape_3d(ctx, k_flat, m->n_embd_head, 1, m->n_heads_kv);
            v_attn = ggml_reshape_3d(ctx, v_flat, m->n_embd_head, 1, m->n_heads_kv);
        } else {
            k_attn = ggml_view_3d(ctx, m->k_cache[il],
                m->n_embd_head, pos + 1, m->n_heads_kv, hfkv, hf, 0);
            v_attn = ggml_view_3d(ctx, m->v_cache[il],
                m->n_embd_head, pos + 1, m->n_heads_kv, hfkv, hf, 0);
        }

        struct ggml_tensor *attn = ggml_flash_attn_ext(ctx, q, k_attn, v_attn, NULL, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_DEFAULT);

        // Output projection: reshape to [n_q_embd], project to [n_embd]
        attn = ggml_reshape_1d(ctx, attn, n_q_embd);
        struct ggml_tensor *wo_out = ggml_mul_mat(ctx, l->wo, attn);
        wo_out = ggml_reshape_1d(ctx, wo_out, m->n_embd);
        x = ggml_add(ctx, x, wo_out);

        // FFN
        xn = ggml_rms_norm(ctx, x, m->rms_eps);
        xn = ggml_mul(ctx, xn, l->ffn_norm);
        struct ggml_tensor *gate = ggml_silu(ctx, ggml_mul_mat(ctx, l->ffn_gate, xn));
        struct ggml_tensor *up = ggml_mul_mat(ctx, l->ffn_up, xn);
        struct ggml_tensor *ffn_out = ggml_mul_mat(ctx, l->ffn_down, ggml_mul(ctx, gate, up));
        ffn_out = ggml_reshape_1d(ctx, ffn_out, m->n_embd);
        x = ggml_add(ctx, x, ffn_out);
    }

    // Final norm + LM head
    x = ggml_rms_norm(ctx, x, m->rms_eps);
    x = ggml_mul(ctx, x, m->output_norm);
    struct ggml_tensor *logits = ggml_mul_mat(ctx, m->output, x);
    ggml_build_forward_expand(gf, logits);

    enum ggml_status rc = ggml_backend_graph_compute(m->backend, gf);
    if (rc == GGML_STATUS_SUCCESS) {
        memcpy(out_logits, logits->data, (size_t)m->n_vocab * sizeof(float));

        m->n_past++;
    }
    return rc == GGML_STATUS_SUCCESS ? 0 : -(int)rc;
}

int gpll_decode_mtp(gpll_model *m, int token_id, float *out_logits, float *out_logits_mtp) {
    int rc = gpll_decode(m, token_id, out_logits);
    (void)out_logits_mtp;
    return rc;
}

void gpll_reset(gpll_model *m) { m->n_past = 0; }

void gpll_free(gpll_model *m) {
    if (!m) return;
    if (m->work_buf) free(m->work_buf);
    if (m->ctx_compute) ggml_free(m->ctx_compute);
    if (m->scratch_mem) free(m->scratch_mem);
    if (m->buf_plain) ggml_backend_buffer_free(m->buf_plain);
    if (m->buf_weights) ggml_backend_buffer_free(m->buf_weights);
    if (m->ctx_weights) ggml_free(m->ctx_weights);
    if (m->buf_kv) ggml_backend_buffer_free(m->buf_kv);
    if (m->ctx_kv) ggml_free(m->ctx_kv);
    if (m->backend) ggml_backend_free(m->backend);
    free(m);
}
