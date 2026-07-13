package mosstranscribe

import (
	"math"
	"testing"
)

func TestTimeMergeIsTrimmedZeroCopyReshape(t *testing.T) {
	encoder := make([]float32, 9*AudioWidth)
	for i := range encoder {
		encoder[i] = float32(i)
	}
	merged, tokens, ok := TimeMerge(encoder, 9)
	if !ok || tokens != 2 || len(merged) != 2*AdaptorInputDim {
		t.Fatalf("TimeMerge len=%d tokens=%d ok=%v", len(merged), tokens, ok)
	}
	if &merged[0] != &encoder[0] || merged[AdaptorInputDim] != encoder[4*AudioWidth] {
		t.Fatal("TimeMerge did not preserve row-major reshape order/storage")
	}
	if _, _, ok := TimeMerge(encoder[:3*AudioWidth], 3); ok {
		t.Fatal("TimeMerge accepted fewer than four frames")
	}
}

func TestForwardAdaptorSparseOracle(t *testing.T) {
	w := AdaptorWeights{
		Linear1Weight: make([]uint16, AdaptorHiddenDim*AdaptorInputDim),
		Linear1Bias:   make([]float32, AdaptorHiddenDim),
		Linear2Weight: make([]uint16, AdaptorHiddenDim*AdaptorHiddenDim),
		Linear2Bias:   make([]float32, AdaptorHiddenDim),
		NormWeight:    make([]float32, AdaptorHiddenDim),
		NormBias:      make([]float32, AdaptorHiddenDim),
	}
	const bf16One = uint16(0x3f80)
	for i := 0; i < AdaptorHiddenDim; i++ {
		w.Linear1Weight[i*AdaptorInputDim+i] = bf16One
		w.Linear2Weight[i*AdaptorHiddenDim+i] = bf16One
		w.Linear1Bias[i] = float32(i%7-3) * 0.01
		w.Linear2Bias[i] = float32(i%5-2) * 0.02
		w.NormWeight[i] = 0.5 + float32(i%3)*0.25
		w.NormBias[i] = float32(i%11-5) * 0.01
	}
	merged := make([]float32, AdaptorInputDim)
	for i := 0; i < AdaptorHiddenDim; i++ {
		merged[i] = float32(i%23-11) * 0.125
	}
	out := make([]float32, AdaptorHiddenDim)
	scratch := make([]float32, AdaptorHiddenDim)
	if !ForwardAdaptorTo(out, scratch, merged, 1, w) {
		t.Fatal("ForwardAdaptorTo failed")
	}

	preNorm := make([]float32, AdaptorHiddenDim)
	var mean float32
	for i := range preNorm {
		x := merged[i] + w.Linear1Bias[i]
		x = x / (1 + float32(math.Exp(float64(-x))))
		preNorm[i] = x + w.Linear2Bias[i]
		mean += preNorm[i]
	}
	mean /= AdaptorHiddenDim
	var variance float32
	for _, x := range preNorm {
		d := x - mean
		variance += d * d
	}
	variance /= AdaptorHiddenDim
	invStd := float32(1 / math.Sqrt(float64(variance+AdaptorNormEps)))
	for i, x := range preNorm {
		want := (x-mean)*invStd*w.NormWeight[i] + w.NormBias[i]
		if diff := math.Abs(float64(out[i] - want)); diff > 5e-5 {
			t.Fatalf("out[%d]=%.9g want %.9g diff %.3g", i, out[i], want, diff)
		}
	}
}

func TestForwardAdaptorRejectsMalformedBuffers(t *testing.T) {
	if ForwardAdaptorTo(nil, nil, nil, 0, AdaptorWeights{}) {
		t.Fatal("accepted malformed buffers")
	}
}
