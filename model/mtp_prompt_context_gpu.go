package model

import "fmt"

// BuildMTPPromptContext runs the prompt through the hybrid GPU/CPU Generate
// path and returns the final activation plus float KV caches for MTP seeding.
// GPU-resident KV prefixes are copied back from device buffers for the returned
// context; CPU-resident layers reuse the CPU KV shadows maintained by Generate.
func (g *GPUModel) BuildMTPPromptContext(tokenIDs []int) (MTPPromptContext, error) {
	if g == nil || g.CPU == nil {
		return MTPPromptContext{}, fmt.Errorf("nil GPU model")
	}
	oldCapture := g.capturePromptContext
	g.capturePromptContext = true
	defer func() { g.capturePromptContext = oldCapture }()
	// Generate(..., 0) intentionally stops before the last prompt position.
	// Request one internal step; capturePromptContext intercepts the final prompt
	// activation before LM-head logits and prevents token append for this helper.
	out := g.Generate(tokenIDs, 1)
	_ = out
	if len(g.lastPromptTokens) == 0 || len(g.lastActivation) == 0 {
		return MTPPromptContext{}, fmt.Errorf("GPU prompt context was not captured")
	}
	seqLen := len(g.lastPromptTokens)
	kvK := make([][]float32, len(g.CPU.Layers))
	kvV := make([][]float32, len(g.CPU.Layers))
	for l := range g.CPU.Layers {
		dim, err := g.CPU.LayerKVDim(l)
		if err != nil {
			return MTPPromptContext{}, err
		}
		if dim == 0 {
			continue
		}
		want, ok := checkedProduct(seqLen, dim)
		if !ok {
			return MTPPromptContext{}, fmt.Errorf("GPU prompt KV length overflows layer=%d seq=%d dim=%d", l, seqLen, dim)
		}
		if l < len(g.kvGPU_K) && g.kvGPU_K[l] != nil && g.kvGPU_V[l] != nil {
			kd := g.kvGPU_K[l].Data()
			vd := g.kvGPU_V[l].Data()
			if len(kd) < want || len(vd) < want {
				return MTPPromptContext{}, fmt.Errorf("GPU KV layer %d len K/V=%d/%d want %d", l, len(kd), len(vd), want)
			}
			kvK[l] = append([]float32(nil), kd[:want]...)
			kvV[l] = append([]float32(nil), vd[:want]...)
			continue
		}
		if len(g.kvCacheK[l]) < want || len(g.kvCacheV[l]) < want {
			return MTPPromptContext{}, fmt.Errorf("CPU shadow KV layer %d len K/V=%d/%d want %d", l, len(g.kvCacheK[l]), len(g.kvCacheV[l]), want)
		}
		kvK[l] = append([]float32(nil), g.kvCacheK[l][:want]...)
		kvV[l] = append([]float32(nil), g.kvCacheV[l][:want]...)
	}
	return MTPPromptContext{
		Tokens:         append([]int(nil), g.lastPromptTokens...),
		PreviousToken:  g.lastPromptTokens[len(g.lastPromptTokens)-1],
		Activation:     append([]float32(nil), g.lastActivation...),
		KVCacheK:       kvK,
		KVCacheV:       kvV,
		SeqLen:         seqLen,
		FinalLogits:    append([]float32(nil), g.lastLogits...),
		FinalToken:     g.lastToken,
		LogitsComputed: len(g.lastLogits) > 0,
	}, nil
}
