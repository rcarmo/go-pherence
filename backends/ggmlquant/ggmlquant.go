//go:build ggml && cgo && linux

package ggmlquant

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -lggml -lggml-base -lggml-cpu -lm -lstdc++
#include <stdint.h>
#include <stdlib.h>
#include <ggml.h>
#include <ggml-cpu.h>

static int type_size(int typ) { return (int)ggml_type_size((enum ggml_type)typ); }
static int blck_size(int typ) { return (int)ggml_blck_size((enum ggml_type)typ); }
static const char* type_name(int typ) { return ggml_type_name((enum ggml_type)typ); }
static void dequant_row(int typ, const void * x, float * y, int64_t k) {
    const struct ggml_type_traits * tr = ggml_get_type_traits((enum ggml_type)typ);
    tr->to_float(x, y, k);
}
static int vecdot_type(int typ) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu((enum ggml_type)typ);
    return (int)tr->vec_dot_type;
}
static int has_vecdot(int typ) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu((enum ggml_type)typ);
    return tr->vec_dot != 0;
}
static int nrows_for_type(int typ) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu((enum ggml_type)typ);
    return (int)tr->nrows;
}
static int has_from_float(int typ) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu((enum ggml_type)typ);
    return tr->from_float != 0;
}
static void from_float_cpu(int typ, const float * x, void * y, int64_t k) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu((enum ggml_type)typ);
    tr->from_float(x, y, k);
}
static void vecdot(int typ, int n, float * s, const void * x, const void * y) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu((enum ggml_type)typ);
    tr->vec_dot(n, s, 0, x, 0, y, 0, 1);
}

static void vecdot_rows(int typ, int n, float * out, const void * x_rows, size_t row_bytes, const void * y, int nrows) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu((enum ggml_type)typ);
    for (int r = 0; r < nrows; r++) {
        const char * xr = (const char *)x_rows + (size_t)r * row_bytes;
        float s = 0.0f;
        tr->vec_dot(n, &s, 0, xr, 0, y, 0, 1);
        out[r] = s;
    }
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/internal/ggmlutil"
)

const (
	F32  = 0
	F16  = 1
	Q8_0 = 8
	Q8_K = 15
	Q2_K = 10
	Q3_K = 11
	Q4_K = 12
	Q6_K = 14
)

func TypeName(t int) string     { return C.GoString(C.type_name(C.int(t))) }
func TypeSize(t int) int        { return int(C.type_size(C.int(t))) }
func BlockSize(t int) int       { return int(C.blck_size(C.int(t))) }
func VecDotType(t int) int      { return int(C.vecdot_type(C.int(t))) }
func NRows(t int) int           { return int(C.nrows_for_type(C.int(t))) }
func HasVecDot(t int) bool      { return C.has_vecdot(C.int(t)) != 0 }
func HasFromFloat(t int) bool   { return C.has_from_float(C.int(t)) != 0 }
func RawBytes(t int, n int) int { return ggmlutil.RawBytes(n, BlockSize(t), TypeSize(t)) }

func QuantizeFromFloat(t int, x []float32) ([]byte, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("empty input")
	}
	if !HasFromFloat(t) {
		return nil, fmt.Errorf("type %d no from_float", t)
	}
	raw := make([]byte, RawBytes(t, len(x)))
	if len(raw) == 0 {
		return nil, fmt.Errorf("type %d raw size zero for n=%d", t, len(x))
	}
	C.from_float_cpu(C.int(t), (*C.float)(unsafe.Pointer(&x[0])), unsafe.Pointer(&raw[0]), C.int64_t(len(x)))
	return raw, nil
}

func DequantRow(t int, raw []byte, out []float32) error {
	if len(raw) == 0 || len(out) == 0 {
		return fmt.Errorf("empty raw/out")
	}
	C.dequant_row(C.int(t), unsafe.Pointer(&raw[0]), (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(len(out)))
	return nil
}

// VecDot calls GGML's quantized vec_dot callback. y must be encoded as VecDotType(t).
func VecDot(t int, xRaw []byte, yRaw []byte, n int) (float32, error) {
	if !HasVecDot(t) {
		return 0, fmt.Errorf("type %d no vecdot", t)
	}
	if len(xRaw) == 0 || len(yRaw) == 0 {
		return 0, fmt.Errorf("empty raw")
	}
	var s C.float
	C.vecdot(C.int(t), C.int(n), &s, unsafe.Pointer(&xRaw[0]), unsafe.Pointer(&yRaw[0]))
	return float32(s), nil
}

// VecDotRows computes out[r] = dot(raw row r, yRaw) for nrows rows in one C call.
// yRaw must be encoded as VecDotType(t), usually q8_K for K-quants.
func VecDotRows(t int, out []float32, xRows []byte, rowBytes int, yRaw []byte, n int, nrows int) error {
	if !HasVecDot(t) {
		return fmt.Errorf("type %d no vecdot", t)
	}
	if len(out) < nrows || len(xRows) < rowBytes*nrows || len(yRaw) == 0 {
		return fmt.Errorf("bad VecDotRows sizes")
	}
	C.vecdot_rows(C.int(t), C.int(n), (*C.float)(unsafe.Pointer(&out[0])), unsafe.Pointer(&xRows[0]), C.size_t(rowBytes), unsafe.Pointer(&yRaw[0]), C.int(nrows))
	return nil
}
