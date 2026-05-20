package model

import nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"

// chunkedGPULMHead computes logits using the NVIDIA backend in chunks.
// Returns true if the backend path was used.
func (g *GPUModel) chunkedGPULMHead(logits, hidden []float32, vocabSize, h int) bool {
	if g == nil {
		return false
	}
	return nvidia.ChunkedLMHead(logits, hidden, g.lmHead, vocabSize, h)
}
