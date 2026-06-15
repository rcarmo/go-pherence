//go:build riscv64

package rvv

// f32ToF16RVV converts n float32 values to IEEE-754 fp16 bits using RVV/Zvfh.
//
//go:noescape
func f32ToF16RVV(src *float32, dst *uint16, n int)

// F32ToF16RVV converts src float32 values to IEEE-754 fp16 bits in dst. dst must
// have at least len(src) elements. It uses RVV/Zvfh narrowing conversion and is
// intended for fast activation packing before FP16 GEMMs.
func F32ToF16RVV(dst []uint16, src []float32) {
	if len(src) == 0 {
		return
	}
	if len(dst) < len(src) {
		panic("rvv.F32ToF16RVV: len(dst) < len(src)")
	}
	f32ToF16RVV(&src[0], &dst[0], len(src))
}
