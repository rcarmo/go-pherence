package diffusiongemma

import (
	"math"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestChunkedF32GPULMHeadMatchesCPU(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA SGEMM unavailable")
	}
	defer FreeGGUFChunkedLMHeadScratch()
	const vocab = 5
	const hidden = 3
	weights := &TextWeights{
		Globals: []TensorBinding{{TensorHandle: TensorHandle{Name: "model.decoder.embed_tokens.weight"}, DType: "F32", Shape: []int{vocab, hidden}}},
		floatCache: map[string]FloatTensor{
			"model.decoder.embed_tokens.weight": {
				Shape: []int{vocab, hidden},
				DType: "F32",
				Data: []float32{
					1, 0, 2,
					0, 1, -1,
					2, 1, 0,
					-1, 0.5, 1,
					0.25, -0.5, 0.75,
				},
			},
		},
	}
	scratch := ForwardScratch{
		Hidden: []float32{
			0.5, -1, 2,
			1.5, 0.25, -0.5,
		},
		Logits: [][]float32{make([]float32, vocab), make([]float32, vocab)},
	}
	for _, useCache := range []bool{false, true} {
		for i := range scratch.Logits {
			for j := range scratch.Logits[i] {
				scratch.Logits[i][j] = 0
			}
		}
		if err := runChunkedF32GPULMHead(weights, scratch, hidden, 2, useCache); err != nil {
			t.Fatalf("useCache=%v: %v", useCache, err)
		}
		assertChunkedF32LMHeadLogits(t, scratch, weights.floatCache["model.decoder.embed_tokens.weight"], vocab, hidden, useCache)
	}
}

func assertChunkedF32LMHeadLogits(t *testing.T, scratch ForwardScratch, embed FloatTensor, vocab, hidden int, useCache bool) {
	t.Helper()
	for pos := range scratch.Logits {
		for tok := 0; tok < vocab; tok++ {
			var want float32
			for h := 0; h < hidden; h++ {
				want += scratch.Hidden[pos*hidden+h] * embed.Data[tok*hidden+h]
			}
			if math.Abs(float64(scratch.Logits[pos][tok]-want)) > 1e-4 {
				t.Fatalf("useCache=%v logit[%d][%d]=%g want %g", useCache, pos, tok, scratch.Logits[pos][tok], want)
			}
		}
	}
}
