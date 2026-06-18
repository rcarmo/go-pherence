//go:build ggml && cgo && linux

// Package ggmlcompute binds directly to the SpacemiT/GGML quantized dot kernels.
package ggmlcompute

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -lggml -lggml-base -lggml-cpu -lm -lstdc++
#include <stdint.h>
#include <stddef.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <stdlib.h>
#include <string.h>

// These are exported by libggml-cpu.so but not declared in public headers.
extern void quantize_row_q8_K(const float * x, void * vy, int64_t k);
extern void ggml_vec_dot_q2_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q3_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q6_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static size_t gp_type_size(int typ) { return ggml_type_size((enum ggml_type)typ); }
static int gp_blck_size(int typ) { return ggml_blck_size((enum ggml_type)typ); }
static const char * gp_type_name(int typ) { return ggml_type_name((enum ggml_type)typ); }

static void gp_quantize_q8k(const float * x, void * y, int64_t k) {
    ggml_cpu_init();
    quantize_row_q8_K(x, y, k);
}

static int gp_vecdot_rows_direct(int typ, int n, float * out, const void * rows, size_t row_bytes, const void * q8, int nrows) {
    ggml_cpu_init();
    for (int r = 0; r < nrows; r++) {
        const char * row = (const char *) rows + (size_t) r * row_bytes;
        float s = 0.0f;
        switch (typ) {
        case GGML_TYPE_Q2_K:
            ggml_vec_dot_q2_K_q8_K(n, &s, 0, row, 0, q8, 0, 1);
            break;
        case GGML_TYPE_Q3_K:
            ggml_vec_dot_q3_K_q8_K(n, &s, 0, row, 0, q8, 0, 1);
            break;
        case GGML_TYPE_Q6_K:
            ggml_vec_dot_q6_K_q8_K(n, &s, 0, row, 0, q8, 0, 1);
            break;
        default:
            return -1;
        }
        out[r] = s;
    }
    return 0;
}

static int gp_mul_mat_f16_f32(const uint16_t * w_f16, const float * x_f32, float * out, int in_dim, int out_dim) {
    if (!w_f16 || !x_f32 || !out || in_dim <= 0 || out_dim <= 0) {
        return -1;
    }
    ggml_cpu_init();
    size_t mem_size = (size_t) in_dim * (size_t) out_dim * sizeof(uint16_t) +
                      (size_t) in_dim * sizeof(float) +
                      (size_t) out_dim * sizeof(float) +
                      32*1024*1024;
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return -2;
    }
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, in_dim, out_dim);
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_dim);
    memcpy(w->data, w_f16, (size_t) in_dim * (size_t) out_dim * sizeof(uint16_t));
    memcpy(x->data, x_f32, (size_t) in_dim * sizeof(float));
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    struct ggml_backend * backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(ctx);
        return -3;
    }
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return -4;
    }
    memcpy(out, y->data, (size_t) out_dim * sizeof(float));
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}

static int gp_get_row_to_f32(int typ, const void * raw, size_t raw_bytes, int in_dim, int out_dim, int row, float * out) {
    if (!raw || !out || in_dim <= 0 || out_dim <= 0 || row < 0 || row >= out_dim) {
        return -1;
    }
    ggml_cpu_init();
    enum ggml_type gt = (enum ggml_type) typ;
    size_t tensor_bytes = ggml_row_size(gt, in_dim) * (size_t) out_dim;
    if (raw_bytes < tensor_bytes) {
        return -2;
    }
    size_t mem_size = tensor_bytes + (size_t) in_dim * sizeof(float) + 16*1024*1024;
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return -3;
    }
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, gt, in_dim, out_dim);
    struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    memcpy(w->data, raw, tensor_bytes);
    ((int32_t *) ids->data)[0] = row;
    struct ggml_tensor * y = ggml_get_rows(ctx, w, ids);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    struct ggml_backend * backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(ctx);
        return -4;
    }
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return -5;
    }
    memcpy(out, y->data, (size_t) in_dim * sizeof(float));
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/internal/ggmlutil"
)

const (
	F16 = 1
	Q2K = 10
	Q3K = 11
	Q6K = 14
	Q8K = 15
)

func TypeName(t int) string     { return C.GoString(C.gp_type_name(C.int(t))) }
func TypeSize(t int) int        { return int(C.gp_type_size(C.int(t))) }
func BlockSize(t int) int       { return int(C.gp_blck_size(C.int(t))) }
func RawBytes(t int, n int) int { return ggmlutil.RawBytes(n, BlockSize(t), TypeSize(t)) }

func QuantizeQ8K(x []float32) ([]byte, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("empty x")
	}
	raw := make([]byte, RawBytes(Q8K, len(x)))
	if len(raw) == 0 {
		return nil, fmt.Errorf("q8_K raw size zero for n=%d", len(x))
	}
	C.gp_quantize_q8k((*C.float)(unsafe.Pointer(&x[0])), unsafe.Pointer(&raw[0]), C.int64_t(len(x)))
	return raw, nil
}

func VecDotRowsDirect(qtype int, out []float32, rows []byte, rowBytes int, q8 []byte, n int, nrows int) error {
	if len(out) < nrows || len(rows) < rowBytes*nrows || len(q8) == 0 {
		return fmt.Errorf("bad VecDotRowsDirect sizes")
	}
	rc := C.gp_vecdot_rows_direct(C.int(qtype), C.int(n), (*C.float)(unsafe.Pointer(&out[0])), unsafe.Pointer(&rows[0]), C.size_t(rowBytes), unsafe.Pointer(&q8[0]), C.int(nrows))
	if rc != 0 {
		return fmt.Errorf("unsupported qtype %d", qtype)
	}
	return nil
}

func MulMatF16F32(out []float32, wF16 []uint16, x []float32, inDim, outDim int) error {
	if inDim <= 0 || outDim <= 0 || len(out) < outDim || len(wF16) < inDim*outDim || len(x) < inDim {
		return fmt.Errorf("bad MulMatF16F32 sizes")
	}
	rc := C.gp_mul_mat_f16_f32((*C.uint16_t)(unsafe.Pointer(&wF16[0])), (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(inDim), C.int(outDim))
	if rc != 0 {
		return fmt.Errorf("ggml F16 mul_mat failed rc=%d", int(rc))
	}
	return nil
}

func GetRowToF32(qtype int, out []float32, raw []byte, inDim, outDim, row int) error {
	if inDim <= 0 || outDim <= 0 || row < 0 || row >= outDim || len(out) < inDim || len(raw) == 0 {
		return fmt.Errorf("bad GetRowToF32 sizes")
	}
	rc := C.gp_get_row_to_f32(C.int(qtype), unsafe.Pointer(&raw[0]), C.size_t(len(raw)), C.int(inDim), C.int(outDim), C.int(row), (*C.float)(unsafe.Pointer(&out[0])))
	if rc != 0 {
		return fmt.Errorf("ggml get_rows failed rc=%d", int(rc))
	}
	return nil
}
