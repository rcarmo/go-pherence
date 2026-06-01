package model

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
