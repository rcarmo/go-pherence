package nvidia

import "testing"

func TestUploadMLXWeightValidation(t *testing.T) {
	if _, err := UploadMLXWeight(nil, nil, nil, 0, 8, 64, false); err == nil {
		t.Fatalf("expected invalid dimension error")
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := UploadMLXWeight(nil, nil, nil, 16, maxInt/2+1, 8, false); err == nil {
		t.Fatalf("expected MLX weight size overflow error")
	}
	if validGPUMLXWeight(&GPUMLXWeight{InDim: 16, OutDim: 8, Groups: 99, GroupSz: 8, QWeight: &Buffer{Size: 64}, Scales: &Buffer{Size: 64}, Biases: &Buffer{Size: 64}}) {
		t.Fatalf("validGPUMLXWeight accepted inconsistent groups")
	}
	if !SgemmReady() {
		t.Skip("GPU not available; dimension validation checked before readiness")
	}
	if _, err := UploadMLXWeight(make([]uint32, 1), make([]float32, 8), make([]float32, 8), 16, 8, 8, false); err == nil {
		t.Fatalf("expected short weight error")
	}
	if _, err := UploadMLXWeight(make([]uint32, 16), make([]float32, 1), make([]float32, 8), 16, 8, 8, false); err == nil {
		t.Fatalf("expected short scale error")
	}
}

func TestMLXDispatchMalformedDoesNotPanic(t *testing.T) {
	out := NewDevBuf(1)
	x := NewDevBufFrom([]float32{1})
	GemvMLX(out, x, nil)
	GemmMLX(out, x, nil, 2)
	GemvMLXDirect(out, x, nil)
	cand := NewMLXSelectedExpertPersistentCandidate()
	defer cand.Free()
	_ = cand.Run(out, x, nil, nil)
	validShape := &GPUMLXWeight{InDim: 8, OutDim: 2, Groups: 1, GroupSz: 8, QWeight: &Buffer{Ptr: 1, Size: 4}, Scales: &Buffer{Ptr: 1, Size: 8}, Biases: &Buffer{Ptr: 1, Size: 8}}
	GemvMLX(NewDevBuf(1), NewDevBuf(8), validShape)
	GemmMLX(NewDevBuf(3), NewDevBuf(16), validShape, 2)
	GemmMLX(NewDevBuf(4), NewDevBuf(15), validShape, 2)
	GemvMLXDirect(NewDevBuf(1), NewDevBuf(8), validShape)
	_ = cand.Run(NewDevBuf(2), NewDevBuf(8), []*GPUMLXWeight{validShape}, []uint32{0})
}
