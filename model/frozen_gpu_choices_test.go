package model

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestSelectedBF16HeadRowsAndBias(t *testing.T) {
	// Rows differ deliberately: using the wrong embedding/head matrix cannot
	// produce the expected logits. The caller selects which actual tensor.
	weights := []float32{1, 2, 3, 4, -1, 2, 5, -2}
	raw := make([]byte, len(weights)*2)
	for i, v := range weights {
		binary.LittleEndian.PutUint16(raw[i*2:], uint16(math.Float32bits(v)>>16))
	}
	got, err := selectedBF16Logits([]float32{2, 3}, raw, 4, []int{3, 1}, []float32{0, 0.5, 0, -1})
	if err != nil || len(got) != 2 || got[0] != 3 || got[1] != 18.5 {
		t.Fatal(got, err)
	}
	for _, ids := range [][]int{{-1}, {4}} {
		if _, err := selectedBF16Logits([]float32{2, 3}, raw, 4, ids, nil); err == nil {
			t.Fatal("accepted invalid ID")
		}
	}
	if _, err := selectedBF16Logits([]float32{float32(math.NaN()), 1}, raw, 4, []int{0}, nil); err == nil {
		t.Fatal("nonfinite hidden accepted")
	}
	if _, err := selectedBF16Logits([]float32{1, 2}, raw, 4, []int{0}, []float32{1}); err == nil {
		t.Fatal("bias mismatch accepted")
	}
}
