package model

import (
	"math"
	"testing"
)

func gqaAttentionScaleReference(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	h := numHeads * headDim
	kvDim := numKVHeads * headDim
	headsPerKV := numHeads / numKVHeads
	out := make([]float32, h)
	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV
		scores := make([]float32, seqLen)
		for t := 0; t < seqLen; t++ {
			sum := float32(0)
			for d := 0; d < headDim; d++ {
				sum += q[head*headDim+d] * kCache[t*kvDim+kvHead*headDim+d]
			}
			scores[t] = sum * scale
		}
		mx := scores[0]
		for _, v := range scores[1:] {
			if v > mx {
				mx = v
			}
		}
		expSum := float32(0)
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - mx)))
			expSum += scores[i]
		}
		for i := range scores {
			scores[i] /= expSum
		}
		for d := 0; d < headDim; d++ {
			sum := float32(0)
			for t := 0; t < seqLen; t++ {
				sum += scores[t] * vCache[t*kvDim+kvHead*headDim+d]
			}
			out[head*headDim+d] = sum
		}
	}
	return out
}

func TestGQAAttentionScaleMatchesReference(t *testing.T) {
	seqLen := 17
	numHeads := 6
	numKVHeads := 2
	headDim := 32
	q := benchSeq(numHeads * headDim)
	k := benchSeq(seqLen * numKVHeads * headDim)
	v := benchSeq(seqLen * numKVHeads * headDim)
	scale := float32(0.37)
	got := gqaAttentionScale(q, k, v, seqLen, numHeads, numKVHeads, headDim, scale)
	want := gqaAttentionScaleReference(q, k, v, seqLen, numHeads, numKVHeads, headDim, scale)
	assertCloseFloat32Slice(t, "allocated", got, want, 2e-5)

	gotScratch := make([]float32, numHeads*headDim)
	scores := make([]float32, seqLen)
	for i := range gotScratch {
		gotScratch[i] = 123 // ensure gqaAttentionScaleInto clears reusable output
	}
	gqaAttentionScaleInto(gotScratch, scores, q, k, v, seqLen, numHeads, numKVHeads, headDim, scale)
	assertCloseFloat32Slice(t, "scratch", gotScratch, want, 2e-5)
}

func TestAttentionLogitSoftcapIsGemma2Only(t *testing.T) {
	cfg := LlamaConfig{ModelType: "gemma2", AttentionLogitSoftcapping: 50}
	if got := attentionLogitSoftcap(cfg); got != 50 {
		t.Fatalf("Gemma2 softcap=%g want 50", got)
	}
	cfg.ModelType = "qwen3moe"
	if got := attentionLogitSoftcap(cfg); got != 0 {
		t.Fatalf("non-Gemma2 softcap=%g want 0", got)
	}
}

func TestGemma2AttentionScaleByModelSize(t *testing.T) {
	cfg := LlamaConfig{ModelType: "gemma2", HiddenSize: 4608, NumHeads: 32, NumLayers: 46}
	if got, want := attentionScale(cfg, 256), float32(1.0/12.0); math.Abs(float64(got-want)) > 1e-7 {
		t.Fatalf("27B scale=%g want=%g", got, want)
	}
	cfg.NumLayers = 42
	if got, want := attentionScale(cfg, 256), float32(1.0/16.0); math.Abs(float64(got-want)) > 1e-7 {
		t.Fatalf("9B scale=%g want=%g", got, want)
	}
	cfg.NumLayers = 1
	if err := validateGemma2AttentionConfig(cfg); err == nil {
		t.Fatal("ambiguous Gemma2 layer count accepted")
	}
}

func TestGQAAttentionSoftcapAppliesBeforeSoftmax(t *testing.T) {
	q := []float32{1}
	k := []float32{-20, 0, 20}
	v := []float32{1, 3, 9}
	const cap = float32(2)
	got := gqaAttentionScaleSoftcap(q, k, v, 3, 1, 1, 1, 1, cap)
	if len(got) != 1 {
		t.Fatalf("softcap attention len=%d want 1", len(got))
	}
	scores := []float64{
		math.Tanh(-20/float64(cap)) * float64(cap),
		0,
		math.Tanh(20/float64(cap)) * float64(cap),
	}
	maxScore := scores[2]
	var weights [3]float64
	var sum float64
	for i, score := range scores {
		weights[i] = math.Exp(score - maxScore)
		sum += weights[i]
	}
	want := float32((weights[0]*1 + weights[1]*3 + weights[2]*9) / sum)
	if diff := math.Abs(float64(got[0] - want)); diff > 1e-6 {
		t.Fatalf("softcap attention=%g want=%g diff=%g", got[0], want, diff)
	}
	without := gqaAttentionScale(q, k, v, 3, 1, 1, 1, 1)
	if math.Abs(float64(got[0]-without[0])) < 0.1 {
		t.Fatalf("softcap had no material effect: capped=%g uncapped=%g", got[0], without[0])
	}
}

func TestGGUFAttentionSoftcapMatchesSharedCPUFormula(t *testing.T) {
	q := []float32{1}
	k := []float32{-20, 0, 20}
	v := []float32{1, 3, 9}
	m := &GGUFLlama{Config: GGUFLlamaConfig{AttentionLogitSoftcap: 2}}
	got := make([]float32, 1)
	m.gqaAttentionInto(got, make([]float32, 3), q, k, v, 3, 1, 1, 1)
	want := gqaAttentionScaleSoftcap(q, k, v, 3, 1, 1, 1, 1, 2)
	assertCloseFloat32Slice(t, "gguf-softcap", got, want, 1e-6)
}

func TestGGUFGemma2_27BAttentionScale(t *testing.T) {
	q := []float32{1}
	k := []float32{-2, 0, 2}
	v := []float32{1, 3, 9}
	m := &GGUFLlama{Config: GGUFLlamaConfig{Architecture: "gemma2", NumLayers: 46, HiddenSize: 4, NumHeads: 1}}
	got := make([]float32, 1)
	m.gqaAttentionInto(got, make([]float32, 3), q, k, v, 3, 1, 1, 1)
	want := gqaAttentionScale(q, k, v, 3, 1, 1, 1, 0.5)
	assertCloseFloat32Slice(t, "gguf-gemma2-27b-scale", got, want, 1e-6)
}

func assertCloseFloat32Slice(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("%s[%d]=%.8f want %.8f diff %.8g", name, i, got[i], want[i], diff)
		}
	}
}

func TestAttentionMalformedInputsDoNotPanic(t *testing.T) {
	if got := gqaAttention(nil, nil, nil, 1, 1, 0, 0); got != nil {
		t.Fatalf("gqaAttention malformed=%v, want nil", got)
	}
	out := []float32{99, 100}
	scores := []float32{1}
	gqaAttentionScaleInto(out, scores, nil, nil, nil, 1, 2, 0, 1, 1)
	if out[0] != 99 || out[1] != 100 {
		t.Fatalf("malformed attention modified out: %v", out)
	}
	got := gqaAttentionScale(nil, nil, nil, 0, 2, 1, 2, 1)
	if len(got) != 4 {
		t.Fatalf("zero-seq attention len=%d want 4", len(got))
	}
}

func TestRoPEPartialMalformedInputsDoNotPanic(t *testing.T) {
	x := []float32{1, 2}
	applyRoPEPartial(x, nil, 0, 1, 2, 1)
	applyRoPEPartial(x, []float32{1, 0}, -1, 1, 2, 1)
	applyRoPEPartial(x, []float32{1, 0}, 0, 4, 2, 99)
}
