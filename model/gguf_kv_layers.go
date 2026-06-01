package model

import "github.com/rcarmo/go-pherence/runtime/kv"

// GGUFUsesCompressedKVLayer reports whether a layer uses autoregressive K/V
// attention cache in the native GGUF runtime. Plain LLaMA-family models use KV
// on every layer. QwenNext hybrid GGUFs use recurrent SSM state for most layers
// and full-attention K/V only at the configured interval (the local Qwen3.6 REAP
// checkpoint uses layers 3, 7, 11, ... with interval=4).
func (c GGUFLlamaConfig) GGUFUsesCompressedKVLayer(layer int) bool {
	if layer < 0 || layer >= c.NumLayers {
		return false
	}
	if !c.IsQwenNextHybridGGUF() {
		return true
	}
	if c.FullAttentionInterval <= 0 {
		return false
	}
	return (layer+1)%c.FullAttentionInterval == 0
}

func (c GGUFLlamaConfig) GGUFCompressedKVLayerCount() int {
	count := 0
	for i := 0; i < c.NumLayers; i++ {
		if c.GGUFUsesCompressedKVLayer(i) {
			count++
		}
	}
	return count
}

func (c GGUFLlamaConfig) GGUFTurboQuantKVBytes(tqCfg kv.TurboQuantConfig, enabled bool) (fullBytes, estimatedBytes int64) {
	return c.GGUFTurboQuantKVBytesForSeq(c.MaxSeqLen, tqCfg, enabled)
}

func ggufKVByteSavings(fullBytes, estimatedBytes int64) (savedBytes int64, ratio float64) {
	if fullBytes <= 0 {
		return 0, 0
	}
	savedBytes = fullBytes - estimatedBytes
	if savedBytes < 0 {
		savedBytes = 0
	}
	return savedBytes, float64(estimatedBytes) / float64(fullBytes)
}

func (c GGUFLlamaConfig) GGUFTurboQuantKVBytesForSeq(seqLen int, tqCfg kv.TurboQuantConfig, enabled bool) (fullBytes, estimatedBytes int64) {
	if c.NumLayers <= 0 || c.NumKVHeads <= 0 || c.HeadDim <= 0 || seqLen <= 0 {
		return 0, 0
	}
	kvDim := int64(c.NumKVHeads * c.HeadDim)
	fullPerLayer := int64(seqLen) * kvDim * 2 * 4
	fullBytes = int64(c.GGUFCompressedKVLayerCount()) * fullPerLayer
	if !enabled {
		return fullBytes, fullBytes
	}
	residual := tqCfg.ResidualWindow
	if residual < 0 {
		residual = 0
	}
	if residual > seqLen {
		residual = seqLen
	}
	compressedTokens := seqLen - residual
	bytesPerVec := func(bits int) int64 {
		if bits <= 0 {
			return int64(c.HeadDim * 4)
		}
		packed := (c.HeadDim*bits + 7) / 8
		return int64(c.NumKVHeads * (packed + 8)) // packed payload + per-head min/scale
	}
	compressedPerToken := bytesPerVec(tqCfg.KeyBits) + bytesPerVec(tqCfg.ValueBits)
	estimatedBytes = 0
	tq := kv.NewTurboQuantState(c.HeadDim, c.NumLayers, tqCfg)
	for i := 0; i < c.NumLayers; i++ {
		if !c.GGUFUsesCompressedKVLayer(i) {
			continue
		}
		if tq.IsProtectedLayer(i) {
			estimatedBytes += fullPerLayer
			continue
		}
		estimatedBytes += int64(residual)*kvDim*2*4 + int64(compressedTokens)*compressedPerToken
	}
	return fullBytes, estimatedBytes
}
