package kv

// EstimateTurboQuantKVBytes estimates full-precision and TurboQuant-compressed
// K/V cache bytes for a model shape. The optional usesLayer callback can skip
// layers that do not use autoregressive K/V cache (for example QwenNext
// recurrent/SSM layers). The estimate mirrors CompressedKVCache's storage model:
// full residual tokens are float32 K+V, compressed tokens use packed payloads
// plus per-head min/scale float32 metadata for each of K and V.
func EstimateTurboQuantKVBytes(layers, kvHeads, headDim, seqLen int, cfg TurboQuantConfig, enabled bool, usesLayer func(int) bool) (fullBytes, estimatedBytes int64) {
	if layers <= 0 || kvHeads <= 0 || headDim <= 0 || seqLen <= 0 {
		return 0, 0
	}
	kvDim := kvHeads * headDim
	fullPerLayer := int64(seqLen) * int64(kvDim) * 2 * 4
	layerUsesKV := func(i int) bool {
		return usesLayer == nil || usesLayer(i)
	}
	for i := 0; i < layers; i++ {
		if layerUsesKV(i) {
			fullBytes += fullPerLayer
		}
	}
	if !enabled {
		return fullBytes, fullBytes
	}
	residual := cfg.ResidualWindow
	if residual < 0 {
		residual = 0
	}
	if residual > seqLen {
		residual = seqLen
	}
	compressedTokens := seqLen - residual
	bytesPerVec := func(bits int) int64 {
		if bits <= 0 {
			return int64(headDim * 4)
		}
		packed := (headDim*bits + 7) / 8
		return int64(kvHeads * (packed + 8))
	}
	compressedPerToken := bytesPerVec(cfg.KeyBits) + bytesPerVec(cfg.ValueBits)
	tq := NewTurboQuantState(headDim, layers, cfg)
	for i := 0; i < layers; i++ {
		if !layerUsesKV(i) {
			continue
		}
		if tq.IsProtectedLayer(i) {
			estimatedBytes += fullPerLayer
			continue
		}
		estimatedBytes += int64(residual)*int64(kvDim)*2*4 + int64(compressedTokens)*compressedPerToken
	}
	return fullBytes, estimatedBytes
}

func TurboQuantKVByteSavings(fullBytes, estimatedBytes int64) (savedBytes int64, ratio float64) {
	if fullBytes <= 0 {
		return 0, 0
	}
	savedBytes = fullBytes - estimatedBytes
	if savedBytes < 0 {
		savedBytes = 0
	}
	return savedBytes, float64(estimatedBytes) / float64(fullBytes)
}
