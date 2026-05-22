package qwen

import (
	"math"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func TestSplitQwen35FullQGatePerHead(t *testing.T) {
	qFull := []float32{
		1, 2, 3, 4, // head 0 q, gate
		5, 6, 7, 8, // head 1 q, gate
	}
	q, gate, err := splitQwen35FullQGate(qFull, 2, 2)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	wantQ := []float32{1, 2, 5, 6}
	wantGate := []float32{3, 4, 7, 8}
	for i := range wantQ {
		if q[i] != wantQ[i] || gate[i] != wantGate[i] {
			t.Fatalf("split mismatch q=%v gate=%v", q, gate)
		}
	}
}

func TestSplitQwen35LinearQKVRawThenRepeatToValueHeads(t *testing.T) {
	meta := loaderconfig.QwenNativeMTPMetadata{
		LinearNumKeyHeads:   2,
		LinearNumValueHeads: 4,
		LinearKeyHeadDim:    2,
		LinearValueHeadDim:  2,
	}
	shapes := loaderconfig.Qwen35LinearAttentionShapes{KeyDim: 4, ValueDim: 8, HeadVDim: 2}
	projected := []float32{
		1, 2, // q key-head 0
		3, 4, // q key-head 1
		5, 6, // k key-head 0
		7, 8, // k key-head 1
		9, 10, // v head 0
		11, 12, // v head 1
		13, 14, // v head 2
		15, 16, // v head 3
	}
	parts, err := splitQwen35LinearQKVRaw(projected, shapes)
	if err != nil {
		t.Fatalf("raw split: %v", err)
	}
	q, k, err := repeatQwen35LinearQKToValueHeads(parts.Q, parts.K, meta)
	if err != nil {
		t.Fatalf("repeat: %v", err)
	}
	wantQ := []float32{1, 2, 1, 2, 3, 4, 3, 4}
	wantK := []float32{5, 6, 5, 6, 7, 8, 7, 8}
	for i := range wantQ {
		if q[i] != wantQ[i] || k[i] != wantK[i] {
			t.Fatalf("repeat mismatch q=%v k=%v", q, k)
		}
	}
	if len(parts.V) != 8 || parts.V[0] != 9 || parts.V[7] != 16 {
		t.Fatalf("unexpected value split: %v", parts.V)
	}
}

func TestQwen35LinearAttentionStateShapeMatchesValueHeads(t *testing.T) {
	meta := loaderconfig.QwenNativeMTPMetadata{
		HiddenSize:          16,
		LinearConvKernelDim: 4,
		LinearNumKeyHeads:   2,
		LinearNumValueHeads: 4,
		LinearKeyHeadDim:    2,
		LinearValueHeadDim:  3,
	}
	state, err := NewQwen35LinearAttentionState(meta)
	if err != nil {
		t.Fatalf("state: %v", err)
	}
	wantSSM := meta.LinearNumValueHeads * meta.LinearValueHeadDim * meta.LinearKeyHeadDim
	if len(state.SSM) != wantSSM {
		t.Fatalf("SSM len=%d want %d", len(state.SSM), wantSSM)
	}
}

func TestQwen35GatedRMSNormValueHeads(t *testing.T) {
	x := []float32{3, 4, 0, 5}
	gate := []float32{1, 1, 2, 2}
	weight := []float32{2, 3}
	if err := qwen35GatedRMSNormValueHeads(x, gate, weight, 2, 2, 1e-6); err != nil {
		t.Fatalf("gated norm: %v", err)
	}
	// head 0 scale ~= 1/sqrt((9+16)/2), then per-dim weight and silu(gate)
	silu1 := float32(1 / (1 + math.Exp(-1)))
	scale0 := float32(1 / math.Sqrt((9+16)/2.0+1e-6))
	want0 := float32(3) * scale0 * 2 * silu1
	want1 := float32(4) * scale0 * 3 * silu1
	if math.Abs(float64(x[0]-want0)) > 1e-5 || math.Abs(float64(x[1]-want1)) > 1e-5 {
		t.Fatalf("head0 got %v want [%v %v]", x[:2], want0, want1)
	}
}
