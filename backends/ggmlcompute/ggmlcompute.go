//go:build ggml && cgo && linux

// Package ggmlcompute binds directly to the SpacemiT/GGML quantized dot kernels.
package ggmlcompute

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -lggml -lggml-base -lggml-cpu -lm -lstdc++
#include <stdint.h>
#include <stddef.h>
#include <ggml.h>

// These are exported by libggml-cpu.so but not declared in public headers.
extern void quantize_row_q8_K(const float * x, void * vy, int64_t k);
extern void ggml_vec_dot_q2_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q3_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q6_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static size_t gp_type_size(int typ) { return ggml_type_size((enum ggml_type)typ); }
static int gp_blck_size(int typ) { return ggml_blck_size((enum ggml_type)typ); }
static const char * gp_type_name(int typ) { return ggml_type_name((enum ggml_type)typ); }

static void gp_quantize_q8k(const float * x, void * y, int64_t k) {
    quantize_row_q8_K(x, y, k);
}

static int gp_vecdot_rows_direct(int typ, int n, float * out, const void * rows, size_t row_bytes, const void * q8, int nrows) {
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
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const (
	Q2K = 10
	Q3K = 11
	Q6K = 14
	Q8K = 15
)

func TypeName(t int) string     { return C.GoString(C.gp_type_name(C.int(t))) }
func TypeSize(t int) int        { return int(C.gp_type_size(C.int(t))) }
func BlockSize(t int) int       { return int(C.gp_blck_size(C.int(t))) }
func RawBytes(t int, n int) int { return (n / BlockSize(t)) * TypeSize(t) }

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
