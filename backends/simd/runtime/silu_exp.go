package simd

import "unsafe"

// SiLUMulExpTo computes dst[i] = gate[i] / (1 + exp(-gate[i])) * up[i] using
// caller-owned scratch for the exp(-gate) temporary.
//
// Contract:
//   - len(dst), len(gate), len(up), and len(scratch) must be equal and non-zero
//   - dst may be exactly the same slice as gate and/or up for in-place use
//   - partial overlap between dst and gate/up is rejected and returns false
//   - scratch must be fully disjoint from dst, gate, and up
//
// The exp() stage uses ExpF32To so bounded SIMD kernels are reused when
// available. The reciprocal/divide stage remains scalar because there is no
// checked vector-divide helper yet, and this preserves the current SiLU
// exceptional-value behaviour used by the scalar reference kernels.
func SiLUMulExpTo(dst, gate, up, scratch []float32) bool {
	n := len(dst)
	if n == 0 || len(gate) != n || len(up) != n || len(scratch) != n {
		return false
	}
	if !expF32AliasOK(dst, gate) || !expF32AliasOK(dst, up) {
		return false
	}
	if !float32SlicesDisjoint(scratch, dst) || !float32SlicesDisjoint(scratch, gate) || !float32SlicesDisjoint(scratch, up) {
		return false
	}
	if !VecScaleTo(scratch, gate, -1) {
		return false
	}
	if !ExpF32To(scratch, scratch) {
		return false
	}
	for i, x := range gate {
		scratch[i] = x / (1 + scratch[i])
	}
	return VecMulTo(dst, up, scratch)
}

func float32SlicesDisjoint(a, b []float32) bool {
	ap := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bp := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if ap == bp {
		return false
	}
	asz, okA := checkedFloat32ByteOffset(len(a))
	bsz, okB := checkedFloat32ByteOffset(len(b))
	if !okA || !okB {
		return false
	}
	aEnd := ap + asz
	bEnd := bp + bsz
	return aEnd <= bp || bEnd <= ap
}
