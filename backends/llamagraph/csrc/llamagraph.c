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

    // Get the CPU device and its extra buffer types (SpacemiT repack)
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    m->backend = ggml_backend_dev_init(dev, NULL);
    ggml_backend_cpu_set_n_threads(m->backend, n_threads);

    // Get the SpacemiT repack buffer type via the registry proc address API.
    // This transforms Q2_K/Q4_K/etc weights into 256x32 tiled IME2-native format.
    typedef ggml_backend_buffer_type_t * (*get_extra_bufts_fn)(ggml_backend_dev_t);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    get_extra_bufts_fn get_extras = (get_extra_bufts_fn)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    m->repack_buft = NULL;
    if (get_extras) {
        ggml_backend_buffer_type_t *extras = get_extras(dev);
        if (extras && extras[0]) m->repack_buft = extras[0];
    }


    // --- Weight tensors split into two contexts ---
    // ctx_weights: mul_mat weight matrices → allocated with repack buft (IME2)
    // ctx_plain: embeddings and norms → allocated with plain CPU buft  
    int n_mm_tensors = n_layers * 7;    // wq/wk/wv/wo/gate/up/down per layer (output in plain)
    int n_plain_tensors = 3 + n_layers * 4;  // tok_embd + output + output_norm + (attn_norm + ffn_norm + q_norm + k_norm) per layer
    int n_tensors = n_mm_tensors + n_plain_tensors;
    // Context for mul_mat weights (will be repacked for IME2)
    size_t ctx_mm_size = ggml_tensor_overhead() * n_mm_tensors * 2 + 65536;
    struct ggml_init_params mmp = { ctx_mm_size, NULL, true };
    m->ctx_weights = ggml_init(mmp);

    // Context for plain tensors (tok_embd, norms — used in get_rows/mul, NOT mul_mat)
    size_t ctx_pl_size = ggml_tensor_overhead() * n_plain_tensors * 2 + 65536;
    struct ggml_init_params plp = { ctx_pl_size, NULL, true };
    struct ggml_context *ctx_plain = ggml_init(plp);

    int n_embd_kv = m->n_embd_head * n_heads_kv;

    m->tok_embd    = ggml_new_tensor_2d(ctx_plain, (enum ggml_type)wt->tok_embd, n_embd, n_vocab);
    m->output_norm = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, n_embd);
    m->output      = ggml_new_tensor_2d(ctx_plain, (enum ggml_type)wt->output,   n_embd, n_vocab);

    for (int il = 0; il < n_layers; il++) {
        gpll_layer *l = &m->layers[il];
        l->attn_norm = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, n_embd);
        l->wq        = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wq[il],       n_embd, n_embd);
        l->wk        = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wk[il],       n_embd, n_embd_kv);
        l->wv        = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wv[il],       n_embd, n_embd_kv);
        l->wo        = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->wo[il],       n_embd, n_embd);
        l->ffn_norm  = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, n_embd);
        l->ffn_gate  = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_gate[il], n_embd, n_ff);
        l->ffn_up    = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_up[il],   n_embd, n_ff);
        l->ffn_down  = ggml_new_tensor_2d(m->ctx_weights, (enum ggml_type)wt->ffn_down[il], n_ff,   n_embd);
        // QK norm: F32 [head_dim] — broadcast across heads
        if (m->has_qk_norm) {
            l->q_norm = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, m->n_embd_head);
            l->k_norm = ggml_new_tensor_1d(ctx_plain, GGML_TYPE_F32, m->n_embd_head);
        } else {
            l->q_norm = NULL;
            l->k_norm = NULL;
        }
    }

    // Allocate mul_mat weights through the repack buffer type (triggers IME2 tile transform).
    if (m->repack_buft) {
        m->buf_weights = ggml_backend_alloc_ctx_tensors_from_buft(m->ctx_weights, m->repack_buft);
    } else {
        m->buf_weights = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    }
    // MTP tensors: eh_proj goes in repacked (it's a mul_mat weight), norms go in plain
    m->n_mtp_layers = 0; // Set from Go side after init if model has MTP
    m->mtp_enorm = NULL;
    m->mtp_hnorm = NULL;
    m->mtp_eh_proj = NULL;
    m->mtp_shared_head_norm = NULL;

    // Allocate plain tensors (tok_embd, norms, QK norms) with standard CPU buffer
    m->buf_plain = ggml_backend_alloc_ctx_tensors(ctx_plain, m->backend);

    // --- KV cache (F32, no_alloc=true; allocated below) ---
    size_t ctx_kvsize = ggml_tensor_overhead() * n_layers * 2 + 256;
    struct ggml_init_params kp = { ctx_kvsize, NULL, true };
    m->ctx_kv = ggml_init(kp);

    for (int il = 0; il < n_layers; il++) {
        int n_embd_kv = m->n_embd_head * n_heads_kv;
        m->k_cache[il] = ggml_new_tensor_2d(m->ctx_kv, GGML_TYPE_F32, n_embd_kv, n_ctx);
        m->v_cache[il] = ggml_new_tensor_2d(m->ctx_kv, GGML_TYPE_F32, n_embd_kv, n_ctx);
    }
    m->buf_kv = ggml_backend_alloc_ctx_tensors(m->ctx_kv, m->backend);

    return m;
}

// ---------------------------------------------------------------------------
// Weight setters — each just copies bytes into the pre-allocated tensor.
// ---------------------------------------------------------------------------
void gpll_set_has_qk_norm(gpll_model *m, int v) { m->has_qk_norm = v; }

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
void gpll_set_q_norm    (gpll_model *m, int il, const void *d, size_t n) { if (m->layers[il].q_norm) ggml_backend_tensor_set(m->layers[il].q_norm, d, 0, n); }
void gpll_set_k_norm    (gpll_model *m, int il, const void *d, size_t n) { if (m->layers[il].k_norm) ggml_backend_tensor_set(m->layers[il].k_norm, d, 0, n); }
void gpll_set_mtp_enorm (gpll_model *m, const void *d, size_t n) { if (m->mtp_enorm) ggml_backend_tensor_set(m->mtp_enorm, d, 0, n); }
void gpll_set_mtp_hnorm (gpll_model *m, const void *d, size_t n) { if (m->mtp_hnorm) ggml_backend_tensor_set(m->mtp_hnorm, d, 0, n); }
void gpll_set_mtp_eh_proj(gpll_model *m, const void *d, size_t n) { if (m->mtp_eh_proj) ggml_backend_tensor_set(m->mtp_eh_proj, d, 0, n); }
void gpll_set_mtp_shared_head_norm(gpll_model *m, const void *d, size_t n) { if (m->mtp_shared_head_norm) ggml_backend_tensor_set(m->mtp_shared_head_norm, d, 0, n); }

// ---------------------------------------------------------------------------
// Decode — build and run a full single-token GGML graph
// ---------------------------------------------------------------------------
// gpll_build_graph — call once after weight loading.
// Builds a static decode graph for a single token at position pos_t,
// re-using a persistent scratch buffer so no malloc/free per step.
// gpll_build_graph — allocates the persistent scratch buffer and pre-sizes
// the work buffer. Called lazily on first decode.
int gpll_build_graph(gpll_model *m) {
    // The 512MB scratch buffer is used only for graph metadata (tensor headers)
    // when using no_alloc=true. Actual data goes through ggml_gallocr.
    m->scratch_size = 512ULL * 1024 * 1024;   // 4 MB for tensor metadata
    m->scratch_mem  = malloc(m->scratch_size);
    if (!m->scratch_mem) return -1;
    m->work_size = 0;
    m->work_buf  = NULL;
    return 0;
}

int gpll_decode(gpll_model *m, int token_id, float *out_logits) {
    if (!m->scratch_mem) { if (gpll_build_graph(m) != 0) return -1; }
    int pos = m->n_past;
    int n_embd_kv = m->n_embd_head * m->n_heads_kv;
    size_t hf    = (size_t)m->n_embd_head * sizeof(float);  // head_dim bytes
    size_t hfkv  = (size_t)n_embd_kv      * sizeof(float);  // n_embd_kv bytes (one full position)

    if (m->ctx_compute) ggml_free(m->ctx_compute);
    struct ggml_init_params gp = { m->scratch_size, m->scratch_mem, false };
    m->ctx_compute = ggml_init(gp);
    if (!m->ctx_compute) return -1;
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

    for (int il = 0; il < m->n_layers; il++) {
        gpll_layer *l = &m->layers[il];
        struct ggml_tensor *xn = ggml_rms_norm(ctx, x, m->rms_eps);
        xn = ggml_mul(ctx, xn, l->attn_norm);
        struct ggml_tensor *q = ggml_mul_mat(ctx, l->wq, xn);
        struct ggml_tensor *k = ggml_mul_mat(ctx, l->wk, xn);
        struct ggml_tensor *v = ggml_mul_mat(ctx, l->wv, xn);

        // Reshape for RoPE and optional QK norm
        q = ggml_reshape_3d(ctx, q, m->n_embd_head, m->n_heads,    1);
        k = ggml_reshape_3d(ctx, k, m->n_embd_head, m->n_heads_kv, 1);
        // QK norm (Qwen3+): per-head RMSNorm on Q and K before RoPE
        if (m->has_qk_norm && l->q_norm && l->k_norm) {
            q = ggml_rms_norm(ctx, q, m->rms_eps);
            q = ggml_mul(ctx, q, l->q_norm);
            k = ggml_rms_norm(ctx, k, m->rms_eps);
            k = ggml_mul(ctx, l->k_norm, k);
        }
        v = ggml_reshape_3d(ctx, v, m->n_embd_head, m->n_heads_kv, 1);
        q = ggml_rope_ext(ctx, q, pos_t, NULL, m->rope_dims, GGML_ROPE_TYPE_NORMAL,
                          m->n_ctx, m->rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(ctx, k, pos_t, NULL, m->rope_dims, GGML_ROPE_TYPE_NORMAL,
                          m->n_ctx, m->rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Reshape q to [head_dim, 1, n_heads] for flash_attn
        q = ggml_reshape_3d(ctx, q, m->n_embd_head, 1, m->n_heads);

        // KV cache write: k_cache[il] is F32 [n_embd_kv, n_ctx] (llama.cpp layout)
        //   Position s: base + s * n_embd_kv * 4  (all heads contiguous per position)
        // Flatten k from [head_dim, n_kv_heads, 1] → [n_embd_kv]
        struct ggml_tensor *k_flat = ggml_reshape_1d(ctx, k, n_embd_kv);
        struct ggml_tensor *v_flat = ggml_reshape_1d(ctx, v, n_embd_kv);
        struct ggml_tensor *k_slot = ggml_view_1d(ctx, m->k_cache[il], n_embd_kv,
                                                   hfkv * (size_t)pos);
        struct ggml_tensor *v_slot = ggml_view_1d(ctx, m->v_cache[il], n_embd_kv,
                                                   hfkv * (size_t)pos);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k_flat, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v_flat, v_slot));

        // KV read for flash_attn: [head_dim, pos+1, n_kv_heads]
        //   nb[1] = head_dim*4  (stride per KV head within a position)
        //   nb[2] = n_embd_kv*4 (stride per sequence position)
        //   → reading along seq_len: jumps n_embd_kv*4 = pos-sequential
        // k_cache [n_embd_kv, n_ctx]: nb[1]=hfkv (stride per seq pos), nb[2]=hf (per KV head)
        struct ggml_tensor *k_used = ggml_view_3d(ctx, m->k_cache[il],
            m->n_embd_head, pos + 1, m->n_heads_kv,
            hfkv,  // nb[1]: stride per sequence position (= n_embd_kv floats)
            hf,    // nb[2]: stride per KV head (= head_dim floats, within a position)
            0);
        struct ggml_tensor *v_used = ggml_view_3d(ctx, m->v_cache[il],
            m->n_embd_head, pos + 1, m->n_heads_kv,
            hfkv, hf, 0);

        // flash_attn: q=[head_dim,1,n_heads], k/v=[head_dim,pos+1,n_kv_heads]
        struct ggml_tensor *attn = ggml_flash_attn_ext(ctx,
            q, k_used, v_used, NULL, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_DEFAULT);
        attn = ggml_reshape_1d(ctx, attn, m->n_embd);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, l->wo, attn));

        xn = ggml_rms_norm(ctx, x, m->rms_eps);
        xn = ggml_mul(ctx, xn, l->ffn_norm);
        struct ggml_tensor *gate = ggml_silu(ctx, ggml_mul_mat(ctx, l->ffn_gate, xn));
        struct ggml_tensor *up   = ggml_mul_mat(ctx, l->ffn_up, xn);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, l->ffn_down, ggml_mul(ctx, gate, up)));
    }
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
    // For now: run normal decode, then separately compute MTP projection
    // TODO: integrate MTP into the main graph for better performance
    int rc = gpll_decode(m, token_id, out_logits);
    if (rc != 0 || !m->mtp_eh_proj) return rc;

    // MTP projection: concat(norm(embd), norm(hidden)) → eh_proj → norm → LM head
    // The hidden state is the last residual before output_norm+LM head
    // We need to save it during decode — for now return 0 (TODO)
    (void)out_logits_mtp;
    return 0;
}

void gpll_reset(gpll_model *m) { m->n_past = 0; }

void gpll_free(gpll_model *m) {
    if (!m) return;
    if (m->work_buf)    free(m->work_buf);
    if (m->ctx_compute) ggml_free(m->ctx_compute);
    if (m->scratch_mem) free(m->scratch_mem);
    if (m->buf_plain)   ggml_backend_buffer_free(m->buf_plain);
    if (m->buf_weights) ggml_backend_buffer_free(m->buf_weights);
    if (m->ctx_weights) ggml_free(m->ctx_weights);
    if (m->buf_kv)      ggml_backend_buffer_free(m->buf_kv);
    if (m->ctx_kv)      ggml_free(m->ctx_kv);
    if (m->backend)     ggml_backend_free(m->backend);
    free(m);
}
