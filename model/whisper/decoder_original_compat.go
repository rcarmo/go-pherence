package whisper

import "math"

// originalCrossPadKeys returns the number of virtual zero keys in the original
// decoder cross-attention extent: whisper.cpp pads the encoder context to a
// multiple of 256 and passes no mask to flash attention.
func originalCrossPadKeys(encLen int) int {
	if encLen < 1 {
		return 0
	}
	return (encLen+255)/256*256 - encLen
}

// softmaxPadded normalises x together with pad zero logits. Only the real
// probabilities are written back; they sum to less than one.
func softmaxPadded(x []float32, pad int) {
	if len(x) == 0 {
		return
	}
	max := float32(0) // the zero logits always participate
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		e := fastExpF32(v - max)
		x[i] = e
		sum += e
	}
	sum += float32(pad) * fastExpF32(-max)
	if sum > 0 {
		for i := range x {
			x[i] /= sum
		}
	}
}

// geluOriginalTanh is ggml's tanh-form GELU in the original operation order:
// 0.5*x*(2-2/(exp(2*v)+1)), v=sqrt(2/pi)*x*(1+0.044715*x*x).
func geluOriginalTanh(x []float32) {
	const c = float32(0.7978845608028654)
	for i, v := range x {
		val := c * v * (1 + 0.044715*v*v)
		x[i] = 0.5 * v * (2 - 2/(float32(math.Exp(float64(2*val)))+1))
	}
}

// applyOriginalDecoderCompatibility marks a fresh state for the original's
// decoder semantics. Callers opt in explicitly; defaults are unchanged.
func (s *DecoderState) applyOriginalDecoderCompatibility(encLen int) {
	s.crossPadKeys = originalCrossPadKeys(encLen)
	s.tanhGELU = true
}
