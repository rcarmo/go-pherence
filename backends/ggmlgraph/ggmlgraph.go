//go:build ggml && cgo && linux

package ggmlgraph

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -lggml -lggml-base -lggml-cpu -lm -lstdc++
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>

static void gp_backend_init_all(void) {
    ggml_backend_load_all();
    ggml_cpu_init();
}

struct gp_graph {
    struct ggml_context * ctx;
    struct ggml_cgraph * graph;
    struct ggml_tensor * w;
    struct ggml_tensor * x;
    struct ggml_tensor * y;
    int in_dim;
    int out_dim;
    int n_threads;
};

static struct gp_graph * gp_new_mulmat(int typ, const void * raw, size_t raw_bytes, int in_dim, int out_dim, int n_threads) {
    gp_backend_init_all();
    size_t mem = raw_bytes + (size_t)in_dim*sizeof(float) + (size_t)out_dim*sizeof(float) + 16*1024*1024;
    struct ggml_init_params params = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return NULL;
    struct gp_graph * g = (struct gp_graph *)calloc(1, sizeof(struct gp_graph));
    if (!g) { ggml_free(ctx); return NULL; }
    g->ctx = ctx; g->in_dim = in_dim; g->out_dim = out_dim; g->n_threads = n_threads;
    // GGML convention: matrix a has ne[0]=in_dim, ne[1]=out_dim.
    g->w = ggml_new_tensor_2d(ctx, (enum ggml_type)typ, in_dim, out_dim);
    g->x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, 1);
    if (!g->w || !g->x) { ggml_free(ctx); free(g); return NULL; }
    memcpy(ggml_get_data(g->w), raw, raw_bytes);
    g->y = ggml_mul_mat(ctx, g->w, g->x);
    ggml_set_name(g->y, "y");
    g->graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(g->graph, g->y);
    return g;
}

static int gp_mulmat_run(struct gp_graph * g, const float * x, float * out) {
    memcpy(ggml_get_data(g->x), x, (size_t)g->in_dim*sizeof(float));
    int rc = ggml_graph_compute_with_ctx(g->ctx, g->graph, g->n_threads);
    if (rc != 0) return rc;
    memcpy(out, ggml_get_data(g->y), (size_t)g->out_dim*sizeof(float));
    return 0;
}

static void gp_free(struct gp_graph * g) {
    if (!g) return;
    if (g->ctx) ggml_free(g->ctx);
    free(g);
}


struct gp_qkv_graph {
    struct ggml_context * ctx;
    struct ggml_cgraph * graph;
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * x;
    struct ggml_tensor * q;
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    int in_dim;
    int q_dim;
    int kv_dim;
    int n_threads;
};

static struct gp_qkv_graph * gp_new_qkv(int typ_q, const void * raw_q, size_t raw_q_bytes,
                                        int typ_k, const void * raw_k, size_t raw_k_bytes,
                                        int typ_v, const void * raw_v, size_t raw_v_bytes,
                                        int in_dim, int q_dim, int kv_dim, int n_threads) {
    gp_backend_init_all();
    size_t mem = raw_q_bytes + raw_k_bytes + raw_v_bytes + (size_t)(in_dim + q_dim + 2*kv_dim)*sizeof(float) + 64*1024*1024;
    struct ggml_init_params params = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return NULL;
    struct gp_qkv_graph * g = (struct gp_qkv_graph *)calloc(1, sizeof(struct gp_qkv_graph));
    if (!g) { ggml_free(ctx); return NULL; }
    g->ctx = ctx; g->in_dim = in_dim; g->q_dim = q_dim; g->kv_dim = kv_dim; g->n_threads = n_threads;
    g->wq = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_q, in_dim, q_dim);
    g->wk = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_k, in_dim, kv_dim);
    g->wv = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_v, in_dim, kv_dim);
    g->x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, 1);
    if (!g->wq || !g->wk || !g->wv || !g->x) { ggml_free(ctx); free(g); return NULL; }
    memcpy(ggml_get_data(g->wq), raw_q, raw_q_bytes);
    memcpy(ggml_get_data(g->wk), raw_k, raw_k_bytes);
    memcpy(ggml_get_data(g->wv), raw_v, raw_v_bytes);
    g->q = ggml_mul_mat(ctx, g->wq, g->x);
    g->k = ggml_mul_mat(ctx, g->wk, g->x);
    g->v = ggml_mul_mat(ctx, g->wv, g->x);
    ggml_set_name(g->q, "q"); ggml_set_name(g->k, "k"); ggml_set_name(g->v, "v");
    g->graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(g->graph, g->q);
    ggml_build_forward_expand(g->graph, g->k);
    ggml_build_forward_expand(g->graph, g->v);
    return g;
}

static int gp_qkv_run(struct gp_qkv_graph * g, const float * x, float * q, float * k, float * v) {
    memcpy(ggml_get_data(g->x), x, (size_t)g->in_dim*sizeof(float));
    int rc = ggml_graph_compute_with_ctx(g->ctx, g->graph, g->n_threads);
    if (rc != 0) return rc;
    memcpy(q, ggml_get_data(g->q), (size_t)g->q_dim*sizeof(float));
    memcpy(k, ggml_get_data(g->k), (size_t)g->kv_dim*sizeof(float));
    memcpy(v, ggml_get_data(g->v), (size_t)g->kv_dim*sizeof(float));
    return 0;
}

static void gp_qkv_free(struct gp_qkv_graph * g) {
    if (!g) return;
    if (g->ctx) ggml_free(g->ctx);
    free(g);
}

struct gp_mlp_graph {
    struct ggml_context * ctx;
    struct ggml_cgraph * graph;
    struct ggml_tensor * wg;
    struct ggml_tensor * wu;
    struct ggml_tensor * wd;
    struct ggml_tensor * x;
    struct ggml_tensor * y;
    int in_dim;
    int ffn_dim;
    int out_dim;
    int n_threads;
};

static struct gp_mlp_graph * gp_new_mlp(int typ_g, const void * raw_g, size_t raw_g_bytes,
                                        int typ_u, const void * raw_u, size_t raw_u_bytes,
                                        int typ_d, const void * raw_d, size_t raw_d_bytes,
                                        int in_dim, int ffn_dim, int out_dim, int n_threads) {
    gp_backend_init_all();
    size_t mem = raw_g_bytes + raw_u_bytes + raw_d_bytes + (size_t)(in_dim + 3*ffn_dim + out_dim)*sizeof(float) + 128*1024*1024;
    struct ggml_init_params params = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return NULL;
    struct gp_mlp_graph * g = (struct gp_mlp_graph *)calloc(1, sizeof(struct gp_mlp_graph));
    if (!g) { ggml_free(ctx); return NULL; }
    g->ctx = ctx; g->in_dim = in_dim; g->ffn_dim = ffn_dim; g->out_dim = out_dim; g->n_threads = n_threads;
    g->wg = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_g, in_dim, ffn_dim);
    g->wu = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_u, in_dim, ffn_dim);
    g->wd = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_d, ffn_dim, out_dim);
    g->x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, 1);
    if (!g->wg || !g->wu || !g->wd || !g->x) { ggml_free(ctx); free(g); return NULL; }
    memcpy(ggml_get_data(g->wg), raw_g, raw_g_bytes);
    memcpy(ggml_get_data(g->wu), raw_u, raw_u_bytes);
    memcpy(ggml_get_data(g->wd), raw_d, raw_d_bytes);
    struct ggml_tensor * gate = ggml_mul_mat(ctx, g->wg, g->x);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, g->wu, g->x);
    struct ggml_tensor * act  = ggml_silu(ctx, gate);
    struct ggml_tensor * prod = ggml_mul(ctx, act, up);
    g->y = ggml_mul_mat(ctx, g->wd, prod);
    ggml_set_name(g->y, "mlp_y");
    g->graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(g->graph, g->y);
    return g;
}

static int gp_mlp_run(struct gp_mlp_graph * g, const float * x, float * y) {
    memcpy(ggml_get_data(g->x), x, (size_t)g->in_dim*sizeof(float));
    int rc = ggml_graph_compute_with_ctx(g->ctx, g->graph, g->n_threads);
    if (rc != 0) return rc;
    memcpy(y, ggml_get_data(g->y), (size_t)g->out_dim*sizeof(float));
    return 0;
}

static void gp_mlp_free(struct gp_mlp_graph * g) {
    if (!g) return;
    if (g->ctx) ggml_free(g->ctx);
    free(g);
}



struct gp_backend_mulmat {
    struct ggml_context * ctx;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_cgraph * graph;
    struct ggml_tensor * w;
    struct ggml_tensor * x;
    struct ggml_tensor * y;
    int in_dim;
    int out_dim;
};

static struct gp_backend_mulmat * gp_backend_new_mulmat(int typ, const void * raw, size_t raw_bytes, int in_dim, int out_dim) {
    gp_backend_init_all();
    struct ggml_init_params params = { 32*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return NULL;
    struct gp_backend_mulmat * g = (struct gp_backend_mulmat *)calloc(1, sizeof(struct gp_backend_mulmat));
    if (!g) { ggml_free(ctx); return NULL; }
    g->ctx = ctx; g->in_dim = in_dim; g->out_dim = out_dim;
    g->backend = ggml_backend_cpu_init();
    if (!g->backend) { ggml_free(ctx); free(g); return NULL; }
    g->w = ggml_new_tensor_2d(ctx, (enum ggml_type)typ, in_dim, out_dim);
    g->x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, 1);
    g->y = ggml_mul_mat(ctx, g->w, g->x);
    g->graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(g->graph, g->y);
    g->buffer = ggml_backend_alloc_ctx_tensors(ctx, g->backend);
    if (!g->buffer) { ggml_backend_free(g->backend); ggml_free(ctx); free(g); return NULL; }
    ggml_backend_tensor_set(g->w, raw, 0, raw_bytes);
    return g;
}

static int gp_backend_mulmat_run(struct gp_backend_mulmat * g, const float * x, float * out) {
    ggml_backend_tensor_set(g->x, x, 0, (size_t)g->in_dim*sizeof(float));
    enum ggml_status rc = ggml_backend_graph_compute(g->backend, g->graph);
    if (rc != GGML_STATUS_SUCCESS) return (int)rc;
    ggml_backend_tensor_get(g->y, out, 0, (size_t)g->out_dim*sizeof(float));
    return 0;
}

static void gp_backend_mulmat_free(struct gp_backend_mulmat * g) {
    if (!g) return;
    if (g->buffer) ggml_backend_buffer_free(g->buffer);
    if (g->backend) ggml_backend_free(g->backend);
    if (g->ctx) ggml_free(g->ctx);
    free(g);
}



struct gp_ffn_block_graph {
    struct ggml_context * ctx;
    struct ggml_cgraph * graph;
    struct ggml_tensor * norm;
    struct ggml_tensor * wg;
    struct ggml_tensor * wu;
    struct ggml_tensor * wd;
    struct ggml_tensor * x;
    struct ggml_tensor * y;
    int hidden;
    int ffn;
    int n_threads;
};

static struct gp_ffn_block_graph * gp_new_ffn_block(const float * norm_data,
                                        int typ_g, const void * raw_g, size_t raw_g_bytes,
                                        int typ_u, const void * raw_u, size_t raw_u_bytes,
                                        int typ_d, const void * raw_d, size_t raw_d_bytes,
                                        int hidden, int ffn, float eps, int n_threads) {
    gp_backend_init_all();
    size_t mem = raw_g_bytes + raw_u_bytes + raw_d_bytes + (size_t)(hidden*4 + hidden + 4*ffn)*sizeof(float) + 160*1024*1024;
    struct ggml_init_params params = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return NULL;
    struct gp_ffn_block_graph * g = (struct gp_ffn_block_graph *)calloc(1, sizeof(struct gp_ffn_block_graph));
    if (!g) { ggml_free(ctx); return NULL; }
    g->ctx=ctx; g->hidden=hidden; g->ffn=ffn; g->n_threads=n_threads;
    g->x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    g->norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    g->wg = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_g, hidden, ffn);
    g->wu = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_u, hidden, ffn);
    g->wd = ggml_new_tensor_2d(ctx, (enum ggml_type)typ_d, ffn, hidden);
    if (!g->x || !g->norm || !g->wg || !g->wu || !g->wd) { ggml_free(ctx); free(g); return NULL; }
    memcpy(ggml_get_data(g->norm), norm_data, (size_t)hidden*sizeof(float));
    memcpy(ggml_get_data(g->wg), raw_g, raw_g_bytes);
    memcpy(ggml_get_data(g->wu), raw_u, raw_u_bytes);
    memcpy(ggml_get_data(g->wd), raw_d, raw_d_bytes);
    struct ggml_tensor * nx = ggml_rms_norm(ctx, g->x, eps);
    nx = ggml_mul(ctx, nx, g->norm);
    struct ggml_tensor * gate = ggml_mul_mat(ctx, g->wg, nx);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, g->wu, nx);
    struct ggml_tensor * act  = ggml_silu(ctx, gate);
    struct ggml_tensor * prod = ggml_mul(ctx, act, up);
    struct ggml_tensor * down = ggml_mul_mat(ctx, g->wd, prod);
    g->y = ggml_add(ctx, g->x, down);
    ggml_set_name(g->y, "ffn_block_y");
    g->graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(g->graph, g->y);
    return g;
}

static int gp_ffn_block_run(struct gp_ffn_block_graph * g, const float * x, float * y) {
    memcpy(ggml_get_data(g->x), x, (size_t)g->hidden*sizeof(float));
    int rc = ggml_graph_compute_with_ctx(g->ctx, g->graph, g->n_threads);
    if (rc != 0) return rc;
    memcpy(y, ggml_get_data(g->y), (size_t)g->hidden*sizeof(float));
    return 0;
}

static void gp_ffn_block_free(struct gp_ffn_block_graph * g) {
    if (!g) return;
    if (g->ctx) ggml_free(g->ctx);
    free(g);
}

*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

type MulMat struct {
	p             *C.struct_gp_graph
	inDim, outDim int
}

func NewMulMat(qtype int, raw []byte, inDim, outDim, threads int) (*MulMat, error) {
	if len(raw) == 0 || inDim <= 0 || outDim <= 0 {
		return nil, fmt.Errorf("bad MulMat args")
	}
	p := C.gp_new_mulmat(C.int(qtype), unsafe.Pointer(&raw[0]), C.size_t(len(raw)), C.int(inDim), C.int(outDim), C.int(threads))
	if p == nil {
		return nil, fmt.Errorf("ggml graph allocation failed")
	}
	return &MulMat{p: p, inDim: inDim, outDim: outDim}, nil
}

func (m *MulMat) Close() {
	if m != nil && m.p != nil {
		C.gp_free(m.p)
		m.p = nil
	}
}
func (m *MulMat) Run(x []float32, out []float32) error {
	if len(x) < m.inDim || len(out) < m.outDim {
		return fmt.Errorf("bad Run sizes")
	}
	rc := C.gp_mulmat_run(m.p, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])))
	if rc != 0 {
		return fmt.Errorf("ggml_graph_compute rc=%d", int(rc))
	}
	return nil
}

type QKV struct {
	p                  *C.struct_gp_qkv_graph
	inDim, qDim, kvDim int
}

func NewQKV(wq, wk, wv *gguf.QuantMatrix, threads int) (*QKV, error) {
	if wq == nil || wk == nil || wv == nil || len(wq.Raw) == 0 || len(wk.Raw) == 0 || len(wv.Raw) == 0 {
		return nil, fmt.Errorf("bad QKV args")
	}
	p := C.gp_new_qkv(C.int(wq.QType), unsafe.Pointer(&wq.Raw[0]), C.size_t(len(wq.Raw)), C.int(wk.QType), unsafe.Pointer(&wk.Raw[0]), C.size_t(len(wk.Raw)), C.int(wv.QType), unsafe.Pointer(&wv.Raw[0]), C.size_t(len(wv.Raw)), C.int(wq.InDim), C.int(wq.OutDim), C.int(wk.OutDim), C.int(threads))
	if p == nil {
		return nil, fmt.Errorf("ggml qkv graph allocation failed")
	}
	return &QKV{p: p, inDim: wq.InDim, qDim: wq.OutDim, kvDim: wk.OutDim}, nil
}
func (g *QKV) Close() {
	if g != nil && g.p != nil {
		C.gp_qkv_free(g.p)
		g.p = nil
	}
}
func (g *QKV) Run(x, q, k, v []float32) error {
	if len(x) < g.inDim || len(q) < g.qDim || len(k) < g.kvDim || len(v) < g.kvDim {
		return fmt.Errorf("bad qkv Run sizes")
	}
	rc := C.gp_qkv_run(g.p, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&q[0])), (*C.float)(unsafe.Pointer(&k[0])), (*C.float)(unsafe.Pointer(&v[0])))
	if rc != 0 {
		return fmt.Errorf("ggml qkv compute rc=%d", int(rc))
	}
	return nil
}

type MLP struct {
	p                     *C.struct_gp_mlp_graph
	inDim, ffnDim, outDim int
}

func NewMLP(wg, wu, wd *gguf.QuantMatrix, threads int) (*MLP, error) {
	if wg == nil || wu == nil || wd == nil || len(wg.Raw) == 0 || len(wu.Raw) == 0 || len(wd.Raw) == 0 {
		return nil, fmt.Errorf("bad MLP args")
	}
	p := C.gp_new_mlp(C.int(wg.QType), unsafe.Pointer(&wg.Raw[0]), C.size_t(len(wg.Raw)), C.int(wu.QType), unsafe.Pointer(&wu.Raw[0]), C.size_t(len(wu.Raw)), C.int(wd.QType), unsafe.Pointer(&wd.Raw[0]), C.size_t(len(wd.Raw)), C.int(wg.InDim), C.int(wg.OutDim), C.int(wd.OutDim), C.int(threads))
	if p == nil {
		return nil, fmt.Errorf("ggml mlp graph allocation failed")
	}
	return &MLP{p: p, inDim: wg.InDim, ffnDim: wg.OutDim, outDim: wd.OutDim}, nil
}
func (g *MLP) Close() {
	if g != nil && g.p != nil {
		C.gp_mlp_free(g.p)
		g.p = nil
	}
}
func (g *MLP) Run(x, y []float32) error {
	if len(x) < g.inDim || len(y) < g.outDim {
		return fmt.Errorf("bad mlp Run sizes")
	}
	rc := C.gp_mlp_run(g.p, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
	if rc != 0 {
		return fmt.Errorf("ggml mlp compute rc=%d", int(rc))
	}
	return nil
}

type BackendMulMat struct {
	p             *C.struct_gp_backend_mulmat
	inDim, outDim int
}

func NewBackendMulMat(qtype int, raw []byte, inDim, outDim int) (*BackendMulMat, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("empty raw")
	}
	p := C.gp_backend_new_mulmat(C.int(qtype), unsafe.Pointer(&raw[0]), C.size_t(len(raw)), C.int(inDim), C.int(outDim))
	if p == nil {
		return nil, fmt.Errorf("backend mulmat allocation failed")
	}
	return &BackendMulMat{p: p, inDim: inDim, outDim: outDim}, nil
}
func (m *BackendMulMat) Close() {
	if m != nil && m.p != nil {
		C.gp_backend_mulmat_free(m.p)
		m.p = nil
	}
}
func (m *BackendMulMat) Run(x, out []float32) error {
	if len(x) < m.inDim || len(out) < m.outDim {
		return fmt.Errorf("bad BackendMulMat sizes")
	}
	rc := C.gp_backend_mulmat_run(m.p, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])))
	if rc != 0 {
		return fmt.Errorf("backend mulmat rc=%d", int(rc))
	}
	return nil
}

type FFNBlock struct {
	p      *C.struct_gp_ffn_block_graph
	hidden int
}

func NewFFNBlock(norm []float32, wg, wu, wd *gguf.QuantMatrix, eps float32, threads int) (*FFNBlock, error) {
	if len(norm) == 0 || wg == nil || wu == nil || wd == nil {
		return nil, fmt.Errorf("bad FFNBlock args")
	}
	p := C.gp_new_ffn_block((*C.float)(unsafe.Pointer(&norm[0])), C.int(wg.QType), unsafe.Pointer(&wg.Raw[0]), C.size_t(len(wg.Raw)), C.int(wu.QType), unsafe.Pointer(&wu.Raw[0]), C.size_t(len(wu.Raw)), C.int(wd.QType), unsafe.Pointer(&wd.Raw[0]), C.size_t(len(wd.Raw)), C.int(wg.InDim), C.int(wg.OutDim), C.float(eps), C.int(threads))
	if p == nil {
		return nil, fmt.Errorf("ggml ffn block allocation failed")
	}
	return &FFNBlock{p: p, hidden: wg.InDim}, nil
}
func (g *FFNBlock) Close() {
	if g != nil && g.p != nil {
		C.gp_ffn_block_free(g.p)
		g.p = nil
	}
}
func (g *FFNBlock) Run(x, y []float32) error {
	if len(x) < g.hidden || len(y) < g.hidden {
		return fmt.Errorf("bad FFNBlock sizes")
	}
	rc := C.gp_ffn_block_run(g.p, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
	if rc != 0 {
		return fmt.Errorf("ffn block rc=%d", int(rc))
	}
	return nil
}
