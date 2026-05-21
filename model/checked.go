package model

import gemmacfg "github.com/rcarmo/go-pherence/model/gemma"

func layerKVHeadsForConfig(cfg LlamaConfig, layerIdx int) int {
	return gemmacfg.LayerKVHeads(cfg, layerIdx)
}

func checkedProduct(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}
