//go:build amd64 && cgo

package llamaq4

/*
#cgo CFLAGS: -O3
#include <stdint.h>
void go_llama_q4_0_q8_0_8x4(const uint8_t *q4, const uint8_t *q8, int blocks, float *out);
void go_llama_q4_0_q8_0_projection(const uint8_t *q4, const uint8_t *q8, int rows, int tokens, int blocks, float *out);
void go_llama_q4_0_q8_0_projection_rows(const uint8_t *q4, const uint8_t *q8, int row_base, int row_groups, int rows, int tokens, int blocks, float *out);
*/
import "C"

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/cpu"
)

// Available reports whether the fused kernels can execute on this build and CPU.
func Available() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasAVXVNNI && cpu.X86.HasFMA
}

func DotQ4_0x8Q8_0x4VNNI(q4, q8 []byte, blocks int, out *[32]float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI and FMA")
	}
	if blocks <= 0 || len(q4) != blocks*144 || len(q8) != blocks*136 {
		return fmt.Errorf("llama tile size: q4=%d q8=%d blocks=%d", len(q4), len(q8), blocks)
	}
	C.go_llama_q4_0_q8_0_8x4(
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(q4))),
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(q8))),
		C.int(blocks),
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	return nil
}

func ProjectQ4_0x8Q8_0x4VNNI(q4, q8 []byte, rows, tokens, blocks int, out []float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI and FMA")
	}
	rowGroups, tokenGroups := (rows+7)/8, (tokens+3)/4
	if rows <= 0 || tokens <= 0 || blocks <= 0 || len(q4) != rowGroups*blocks*144 || len(q8) != tokenGroups*blocks*136 || len(out) != rows*tokens {
		return fmt.Errorf("llama projection size: q4=%d q8=%d out=%d rows=%d tokens=%d blocks=%d", len(q4), len(q8), len(out), rows, tokens, blocks)
	}
	C.go_llama_q4_0_q8_0_projection(
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(q4))),
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(q8))),
		C.int(rows), C.int(tokens), C.int(blocks),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(out))),
	)
	return nil
}

func ProjectQ4_0x8Q8_0x4RowsVNNI(q4, q8 []byte, rowBase, rowGroups, rows, tokens, blocks int, out []float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI and FMA")
	}
	tokenGroups, roundedRows := (tokens+3)/4, (rows+7)/8*8
	if rowBase < 0 || rowBase%8 != 0 || rowGroups <= 0 || rowBase >= rows || rowBase+rowGroups*8 > roundedRows || tokens <= 0 || blocks <= 0 || len(q4) != rowGroups*blocks*144 || len(q8) != tokenGroups*blocks*136 || len(out) != rows*tokens {
		return fmt.Errorf("llama projection rows size: q4=%d q8=%d out=%d base=%d groups=%d rows=%d tokens=%d blocks=%d", len(q4), len(q8), len(out), rowBase, rowGroups, rows, tokens, blocks)
	}
	C.go_llama_q4_0_q8_0_projection_rows(
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(q4))),
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(q8))),
		C.int(rowBase), C.int(rowGroups), C.int(rows), C.int(tokens), C.int(blocks),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(out))),
	)
	return nil
}
