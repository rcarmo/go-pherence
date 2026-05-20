package quant

import "testing"

func TestCheckedCompatibilityWrappersRejectMalformedInputs(t *testing.T) {
	if GemvQ4To(make([]float32, 1), make([]float32, 1), nil, nil, nil, nil, 8, 8, true) {
		t.Fatal("GemvQ4To accepted malformed input")
	}
	if GemvQ4SymTo(make([]float32, 1), make([]float32, 1), nil, nil, nil, 8, 8) {
		t.Fatal("GemvQ4SymTo accepted malformed input")
	}
	if GemmQ4(make([]float32, 1), make([]float32, 1), 1, nil, nil, nil, nil, 8, 8, true) {
		t.Fatal("GemmQ4 accepted malformed input")
	}
	if GemmQ4Sym(make([]float32, 1), make([]float32, 1), 1, nil, nil, nil, 8, 8) {
		t.Fatal("GemmQ4Sym accepted malformed input")
	}
	if GemvMLQTo(make([]float32, 1), make([]float32, 1), nil) {
		t.Fatal("GemvMLQTo accepted malformed input")
	}
	if GemmMLQ(make([]float32, 1), make([]float32, 1), 1, nil) {
		t.Fatal("GemmMLQ accepted malformed input")
	}
	if GemvNVFP4To(make([]float32, 1), make([]float32, 1), nil) {
		t.Fatal("GemvNVFP4To accepted malformed input")
	}
	if GemmNVFP4(make([]float32, 1), make([]float32, 1), 1, nil) {
		t.Fatal("GemmNVFP4 accepted malformed input")
	}
}

func TestCheckedCompatibilityWrappersAcceptTinyValidInputs(t *testing.T) {
	q4Out := make([]float32, 1)
	q4X := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	q4Weight := []int32{0x76543210}
	q4GIdx := make([]int32, 8)
	q4Scales := []float32{0.5}
	if !GemvQ4SymTo(q4Out, q4X, q4Weight, q4GIdx, q4Scales, 8, 1) {
		t.Fatal("GemvQ4SymTo rejected valid tiny input")
	}
	if !GemvQ4To(q4Out, q4X, q4Weight, nil, q4GIdx, q4Scales, 8, 1, true) {
		t.Fatal("GemvQ4To rejected valid tiny symmetric input")
	}
	if !GemmQ4Sym(make([]float32, 2), append(q4X, q4X...), 2, q4Weight, q4GIdx, q4Scales, 8, 1) {
		t.Fatal("GemmQ4Sym rejected valid tiny batched input")
	}
	if !GemmQ4(make([]float32, 2), append(q4X, q4X...), 2, q4Weight, nil, q4GIdx, q4Scales, 8, 1, true) {
		t.Fatal("GemmQ4 rejected valid tiny symmetric batched input")
	}

	mlxWeight := &MLXQuantWeight{
		Weight:    []uint32{0x76543210},
		Scales:    []float32{0.5},
		Biases:    []float32{0},
		OutDim:    1,
		InDim:     8,
		Groups:    1,
		GroupSize: 8,
		Bits:      4,
	}
	if !GemvMLQTo(make([]float32, 1), q4X, mlxWeight) {
		t.Fatal("GemvMLQTo rejected valid tiny input")
	}
	if !GemmMLQ(make([]float32, 2), append(q4X, q4X...), 2, mlxWeight) {
		t.Fatal("GemmMLQ rejected valid tiny batched input")
	}

	nvWeight := &NVFP4Weight{
		Weight:       []byte{0x10, 0x32, 0x54, 0x76},
		WeightScale:  []byte{0x38},
		WeightScale2: 1,
		OutDim:       1,
		InDim:        8,
		Groups:       1,
		GroupSize:    8,
	}
	if !GemvNVFP4To(make([]float32, 1), q4X, nvWeight) {
		t.Fatal("GemvNVFP4To rejected valid tiny input")
	}
	if !GemmNVFP4(make([]float32, 2), append(q4X, q4X...), 2, nvWeight) {
		t.Fatal("GemmNVFP4 rejected valid tiny batched input")
	}
}
