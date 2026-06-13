package diffusiongemma

import (
	"math"
	"testing"
)

func TestFlashAttentionContextMatchesMaterialized(t *testing.T) {
	checkAttentionContext(t, 5, 4, 2, 8, 3, 3)
}

func TestBatchedFlashAttentionContextMatchesSerialReference(t *testing.T) {
	const (
		positions     = 2
		heads         = 16
		kvHeads       = 4
		headDim       = 4
		encSeq        = 5
		slidingWindow = 0
	)
	qRows := heads * headDim
	kRows := kvHeads * headDim
	vRows := kvHeads * headDim
	group := heads / kvHeads

	qAll := makePattern(positions*qRows, 0.013, -0.2)
	kAll := makePattern(positions*kRows, -0.021, 0.1)
	vAll := makePattern(positions*vRows, 0.027, -0.15)
	enc := EncoderKVLayer{
		Keys:    makePattern(encSeq*kRows, 0.017, 0.03),
		Values:  makePattern(encSeq*vRows, -0.023, 0.09),
		SeqLen:  encSeq,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	want := make([]float32, positions*qRows)
	got := make([]float32, positions*qRows)

	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3", "0")
	runFlashAttentionContextK3(want, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3", "1")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_THREADS", "8")
	runFlashAttentionContextK3(got, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)

	assertAttentionClose(t, want, got)
}

func checkAttentionContext(t *testing.T, positions, heads, kvHeads, headDim, encSeq, slidingWindow int) {
	t.Helper()
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

	assertAttentionClose(t, want, got)
}

func assertAttentionClose(t *testing.T, want, got []float32) {
	t.Helper()
	var maxDiff float64
	for i := range want {
		diff := math.Abs(float64(want[i] - got[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 2e-5 {
			t.Fatalf("attention mismatch at %d: want %.8f got %.8f diff %.8g", i, want[i], got[i], diff)
		}
	}
	if maxDiff == 0 {
		t.Fatalf("test pattern produced identical output; want a non-trivial comparison")
	}
}

func BenchmarkFlashAttentionContextSerial(b *testing.B) {
	benchmarkFlashAttentionContext(b, false)
}

func BenchmarkFlashAttentionContextBatched(b *testing.B) {
	benchmarkFlashAttentionContext(b, true)
}

func benchmarkFlashAttentionContext(b *testing.B, batched bool) {
	const (
		positions     = 16
		heads         = 16
		kvHeads       = 4
		headDim       = 16
		encSeq        = 64
		slidingWindow = 0
	)
	qRows := heads * headDim
	kRows := kvHeads * headDim
	vRows := kvHeads * headDim
	group := heads / kvHeads
	qAll := makePattern(positions*qRows, 0.013, -0.2)
	kAll := makePattern(positions*kRows, -0.021, 0.1)
	vAll := makePattern(positions*vRows, 0.027, -0.15)
	enc := EncoderKVLayer{
		Keys:    makePattern(encSeq*kRows, 0.017, 0.03),
		Values:  makePattern(encSeq*vRows, -0.023, 0.09),
		SeqLen:  encSeq,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	attnAll := make([]float32, positions*qRows)
	if batched {
		b.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3", "1")
		b.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_THREADS", "8")
	} else {
		b.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3", "0")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runFlashAttentionContextK3(attnAll, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)
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

func TestSlidingDecodePromptMaskMatchesLlamaRule(t *testing.T) {
	const (
		positions     = 1
		heads         = 1
		kvHeads       = 1
		headDim       = 1
		encSeq        = 5
		slidingWindow = 3
	)
	qRows, kRows, vRows, group := 1, 1, 1, 1
	qAll := []float32{0}
	kAll := []float32{0}
	vAll := []float32{100}
	enc := EncoderKVLayer{
		Keys:    []float32{0, 0, 0, 0, 0},
		Values:  []float32{10, 20, 30, 40, 50},
		SeqLen:  encSeq,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	gotMaterialized := make([]float32, 1)
	gotFlash := make([]float32, 1)
	runMaterializedAttentionContextK3(gotMaterialized, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)
	runFlashAttentionContextK3(gotFlash, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)

	// llama.cpp decode SWA rule: canvas_prompt_lo = P - n_swa + 1. With
	// P=5 and n_swa=3 only prompt keys 3 and 4 are visible, plus the canvas key.
	want := float32((40 + 50 + 100) / 3.0)
	for name, got := range map[string]float32{"materialized": gotMaterialized[0], "flash": gotFlash[0]} {
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("%s decode sliding output = %.8f, want %.8f", name, got, want)
		}
	}
}

func TestPromptAllowedForSlidingDecode(t *testing.T) {
	if !promptAllowedForSlidingDecode(0, 5, 0) || !promptAllowedForSlidingDecode(4, 5, 0) {
		t.Fatal("full attention should allow all prompt positions")
	}
	allowed := []bool{
		promptAllowedForSlidingDecode(0, 5, 3),
		promptAllowedForSlidingDecode(1, 5, 3),
		promptAllowedForSlidingDecode(2, 5, 3),
		promptAllowedForSlidingDecode(3, 5, 3),
		promptAllowedForSlidingDecode(4, 5, 3),
	}
	want := []bool{false, false, false, true, true}
	for i := range want {
		if allowed[i] != want[i] {
			t.Fatalf("allowed[%d]=%v want %v (all=%v)", i, allowed[i], want[i], allowed)
		}
	}
}
