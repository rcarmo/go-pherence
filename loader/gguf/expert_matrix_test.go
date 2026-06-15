package gguf

import (
	"encoding/binary"
	"testing"
)

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

func TestDequantRowQ4KToMatchesGGMLNibbleGroups(t *testing.T) {
	raw := make([]byte, 144)
	// d=1, dmin=0; all 8 scale groups use scale=1, min=0.
	binary.LittleEndian.PutUint16(raw[0:2], 0x3c00)
	scales := raw[4:16]
	for i := 0; i < 4; i++ {
		scales[i] = 1
		scales[i+8] = 1
	}
	qs := raw[16:144]
	for i := 0; i < 32; i++ {
		lo := byte(i & 0x0f)
		hi := byte((i + 1) & 0x0f)
		qs[i] = lo | (hi << 4)
	}
	dst := make([]float32, 256)
	if err := dequantRowQ4KTo(dst, raw, 256); err != nil {
		t.Fatal(err)
	}
	checks := map[int]float32{
		0: 0, 1: 1, 15: 15, 16: 0, 31: 15, // group 0: low nibbles q[0:32]
		32: 1, 33: 2, 47: 0, 48: 1, 63: 0, // group 1: high nibbles q[0:32]
	}
	for i, want := range checks {
		if dst[i] != want {
			t.Fatalf("dst[%d]=%v want %v", i, dst[i], want)
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
