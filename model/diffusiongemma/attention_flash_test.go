package diffusiongemma

import (
	"math"
	"testing"
)

func TestFlashAttentionContextMatchesMaterialized(t *testing.T) {
	const (
		positions     = 5
		heads         = 4
		kvHeads       = 2
		headDim       = 8
		encSeq        = 3
		slidingWindow = 3
	)
	qRows := heads * headDim
	kRows := kvHeads * headDim
	vRows := kvHeads * headDim
	group := heads / kvHeads

	qAll := makePattern(positions*qRows, 0.017, -0.3)
	kAll := makePattern(positions*kRows, -0.023, 0.2)
	vAll := makePattern(positions*vRows, 0.031, -0.1)
	enc := EncoderKVLayer{
		Keys:    makePattern(encSeq*kRows, 0.019, 0.05),
		Values:  makePattern(encSeq*vRows, -0.029, 0.15),
		SeqLen:  encSeq,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	want := make([]float32, positions*qRows)
	got := make([]float32, positions*qRows)

	runMaterializedAttentionContextK3(want, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)
	runFlashAttentionContextK3(got, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)

	var maxDiff float64
	for i := range want {
		diff := math.Abs(float64(want[i] - got[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 2e-5 {
			t.Fatalf("flash attention mismatch at %d: want %.8f got %.8f diff %.8g", i, want[i], got[i], diff)
		}
	}
	if maxDiff == 0 {
		t.Fatalf("test pattern produced identical output; want a non-trivial comparison")
	}
}

func makePattern(n int, step, bias float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		v := math.Sin(float64(i+1)*step) + 0.25*math.Cos(float64(i+3)*step*1.7) + bias
		out[i] = float32(v)
	}
	return out
}
