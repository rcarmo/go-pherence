//go:build cgo && linux

package ggmlgraph

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -lggml -lggml-base -lggml-cpu -lm -lstdc++
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ggml.h>
#include <ggml-cpu.h>

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
*/
import "C"

import (
	"fmt"
	"unsafe"
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
