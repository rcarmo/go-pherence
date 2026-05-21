package model

import gemmacfg "github.com/rcarmo/go-pherence/model/gemma"

func normalizeGemma4TextConfig(data []byte, fallback LlamaConfig) (LlamaConfig, bool) {
	return gemmacfg.NormalizeTextConfig(data, fallback)
}
