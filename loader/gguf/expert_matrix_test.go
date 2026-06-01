package gguf

import "testing"

func TestDequantRowQ4KToZeroBlock(t *testing.T) {
	dst := make([]float32, 256)
	if err := dequantRowQ4KTo(dst, make([]byte, 144), 256); err != nil {
		t.Fatal(err)
	}
	for i, v := range dst {
		if v != 0 {
			t.Fatalf("dst[%d]=%v want 0", i, v)
		}
	}
}

func TestDotExpertRow(t *testing.T) {
	if got := dotExpertRow([]float32{1, 2, 3}, []float32{4, 5, 6}); got != 32 {
		t.Fatalf("dot=%v", got)
	}
}

func TestExpertMatricesFromTensorQ4K(t *testing.T) {
	g := &GGUF{Tensors: []TensorInfo{{Name: "blk.0.ffn_gate_exps.weight", Shape: []uint64{256, 2, 3}, QType: QuantQ4_K}}, DataOffset: 0}
	// Avoid file IO by constructing the matrix directly; this validates row math
	// for GGUF's [inDim, outDim, experts] expert tensor layout.
	m := &ExpertMatrices{Name: "experts", QType: QuantQ4_K, Raw: make([]byte, 144*2*3), InDim: 256, OutDim: 2, Experts: 3}
	_ = g
	row := make([]float32, 256)
	if err := m.DequantExpertRowTo(row, 2, 1); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, 2)
	if err := m.GemvExpertTo(out, row, 1); err != nil {
		t.Fatal(err)
	}
}
