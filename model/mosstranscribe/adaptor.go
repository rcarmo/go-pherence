package mosstranscribe

import (
	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

const (
	AudioWidth       = 1024
	AdaptorInputDim  = 4096
	AdaptorHiddenDim = 1024
	AdaptorNormEps   = float32(1e-6)
	AudioMergeFrames = 4
)

// AdaptorWeights are the checkpoint-native BF16 parameters for
// Linear(4096,1024) -> SiLU -> Linear(1024,1024) -> affine LayerNorm.
type AdaptorWeights struct {
	Linear1Weight []uint16
	Linear1Bias   []float32
	Linear2Weight []uint16
	Linear2Bias   []float32
	NormWeight    []float32
	NormBias      []float32
}

func (w AdaptorWeights) valid() bool {
	return len(w.Linear1Weight) == AdaptorHiddenDim*AdaptorInputDim &&
		len(w.Linear1Bias) == AdaptorHiddenDim &&
		len(w.Linear2Weight) == AdaptorHiddenDim*AdaptorHiddenDim &&
		len(w.Linear2Bias) == AdaptorHiddenDim &&
		len(w.NormWeight) == AdaptorHiddenDim &&
		len(w.NormBias) == AdaptorHiddenDim
}

// TimeMerge returns the largest prefix of encoder rows whose storage is the
// checkpoint's four-frame concatenation [tokens,4096]. No copy or arithmetic
// is required because the input is row-major and reshape preserves order.
func TimeMerge(encoder []float32, frames int) ([]float32, int, bool) {
	values, ok := checked.MulInt(frames, AudioWidth)
	if frames < AudioMergeFrames || !ok || len(encoder) < values {
		return nil, 0, false
	}
	tokens := frames / AudioMergeFrames
	mergedValues, ok := checked.MulInt(tokens, AdaptorInputDim)
	if !ok {
		return nil, 0, false
	}
	return encoder[:mergedValues], tokens, true
}

// ForwardAdaptorTo executes the VQ adaptor into caller-owned output. scratch
// must hold tokens*1024 float32 values. Checkpoint BF16 weights stay compressed;
// BF16 dot products and LayerNorm dispatch to SIMD implementations.
func ForwardAdaptorTo(out, scratch, merged []float32, tokens int, w AdaptorWeights) bool {
	inputValues, okIn := checked.MulInt(tokens, AdaptorInputDim)
	outputValues, okOut := checked.MulInt(tokens, AdaptorHiddenDim)
	if tokens <= 0 || !okIn || !okOut || len(merged) < inputValues || len(out) < outputValues || len(scratch) < outputValues || !w.valid() {
		return false
	}
	out = out[:outputValues]
	scratch = scratch[:outputValues]
	if !simd.GemmRowsBF16Parallel(scratch, merged[:inputValues], w.Linear1Weight, tokens, AdaptorHiddenDim, AdaptorInputDim) {
		return false
	}
	addRowBias(scratch, w.Linear1Bias, tokens, AdaptorHiddenDim)
	simd.SiLU(scratch, scratch)
	if !simd.GemmRowsBF16Parallel(out, scratch, w.Linear2Weight, tokens, AdaptorHiddenDim, AdaptorHiddenDim) {
		return false
	}
	addRowBias(out, w.Linear2Bias, tokens, AdaptorHiddenDim)
	return simd.LayerNormLastAxisTo(out, out, tokens, AdaptorHiddenDim, w.NormWeight, w.NormBias, AdaptorNormEps)
}

func addRowBias(values, bias []float32, rows, cols int) {
	for row := 0; row < rows; row++ {
		off := row * cols
		simd.VecAdd(values[off:off+cols], values[off:off+cols], bias[:cols])
	}
}
