package mosstranscribe

import (
	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/half"
)

// GPUAdaptor owns widened checkpoint weights for the small MOSS VQ adaptor.
// Checkpoint BF16 values widen exactly to F32 before their one-time upload.
type GPUAdaptor struct {
	linear1W, linear1B *nv.DevBuf
	linear2W, linear2B *nv.DevBuf
	normW, normB       *nv.DevBuf
	ready              bool
}

func NewGPUAdaptor(weights AdaptorWeights) *GPUAdaptor {
	ga := &GPUAdaptor{}
	if !weights.valid() || !nv.SgemmReady() {
		return ga
	}
	widen := func(values []uint16) []float32 {
		out := make([]float32, len(values))
		for i, value := range values {
			out[i] = half.BF16ToF32(value)
		}
		return out
	}
	upload := func(values []float32) *nv.DevBuf {
		buf := nv.NewDevBufFrom(values)
		if err := buf.ToGPU(); err != nil {
			buf.Free()
			return nil
		}
		return buf
	}
	ga.linear1W = upload(widen(weights.Linear1Weight))
	ga.linear1B = upload(weights.Linear1Bias)
	ga.linear2W = upload(widen(weights.Linear2Weight))
	ga.linear2B = upload(weights.Linear2Bias)
	ga.normW = upload(weights.NormWeight)
	ga.normB = upload(weights.NormBias)
	ga.ready = ga.linear1W != nil && ga.linear1B != nil && ga.linear2W != nil && ga.linear2B != nil && ga.normW != nil && ga.normB != nil
	if !ga.ready {
		ga.Close()
	}
	return ga
}

func (ga *GPUAdaptor) Ready() bool { return ga != nil && ga.ready }

func (ga *GPUAdaptor) Close() {
	if ga == nil {
		return
	}
	for _, buf := range []*nv.DevBuf{ga.linear1W, ga.linear1B, ga.linear2W, ga.linear2B, ga.normW, ga.normB} {
		if buf != nil {
			buf.Free()
		}
	}
	ga.linear1W, ga.linear1B, ga.linear2W, ga.linear2B, ga.normW, ga.normB = nil, nil, nil, nil, nil, nil
	ga.ready = false
}

// Forward runs Linear+SiLU+Linear+LayerNorm with one input upload and one final
// download. It returns false without modifying out when the GPU path fails.
func (ga *GPUAdaptor) Forward(out, merged []float32, tokens int) bool {
	inputN, outputN := tokens*AdaptorInputDim, tokens*AdaptorHiddenDim
	if !ga.Ready() || tokens <= 0 || len(merged) < inputN || len(out) < outputN {
		return false
	}
	input := nv.NewDevBufFrom(merged[:inputN])
	if err := input.ToGPU(); err != nil {
		return false
	}
	defer input.Free()
	scratch := make([]*nv.DevBuf, 0, 10)
	alloc := func(n int) (*nv.DevBuf, bool) {
		buf, err := nv.NewDevBufGPU(n)
		if err != nil {
			return nil, false
		}
		scratch = append(scratch, buf)
		return buf, true
	}
	defer func() {
		for _, buf := range scratch {
			buf.Free()
		}
	}()
	project := func(x, weight, bias *nv.DevBuf, rows, inDim, outDim int) (*nv.DevBuf, bool) {
		xT, ok := alloc(rows * inDim)
		if !ok || nv.WhisperTransposeBuffer(xT.GPUBuffer(), x.GPUBuffer(), rows, inDim) != nil {
			return nil, false
		}
		columnMajor, ok := alloc(rows * outDim)
		if !ok || nv.Sgemm(outDim, rows, inDim, 1, weight.GPUPtr(), xT.GPUBuffer(), columnMajor.GPUBuffer()) != nil {
			return nil, false
		}
		rowMajor, ok := alloc(rows * outDim)
		if !ok || nv.WhisperTransposeBuffer(rowMajor.GPUBuffer(), columnMajor.GPUBuffer(), outDim, rows) != nil {
			return nil, false
		}
		if nv.WhisperRowBiasBuffer(rowMajor.GPUBuffer(), bias.GPUPtr(), rows, outDim) != nil {
			return nil, false
		}
		return rowMajor, true
	}
	hidden, ok := project(input, ga.linear1W, ga.linear1B, tokens, AdaptorInputDim, AdaptorHiddenDim)
	if !ok {
		return false
	}
	nv.DevSiLU(hidden, hidden)
	projected, ok := project(hidden, ga.linear2W, ga.linear2B, tokens, AdaptorHiddenDim, AdaptorHiddenDim)
	if !ok {
		return false
	}
	normalized, ok := alloc(outputN)
	if !ok || nv.IdeogramLayerNormNoAffineBuffer(normalized.GPUBuffer(), projected.GPUBuffer(), tokens, AdaptorHiddenDim, AdaptorNormEps) != nil {
		return false
	}
	final, ok := alloc(outputN)
	if !ok || nv.WhisperRowAffineBuffer(final.GPUBuffer(), normalized.GPUBuffer(), ga.normW.GPUPtr(), ga.normB.GPUPtr(), tokens, AdaptorHiddenDim) != nil {
		return false
	}
	if err := final.GPUBuffer().Download(out[:outputN]); err != nil {
		return false
	}
	return true
}
