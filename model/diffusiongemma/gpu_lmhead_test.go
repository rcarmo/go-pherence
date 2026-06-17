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

func TestDenseF32GPULMHeadDeviceGraphSamplesAndBuildsSC(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA SGEMM unavailable")
	}
	const vocab = 5
	const hidden = 3
	const positions = 2
	// cached is [hidden,vocab] for row-major SGEMM: hidden[positions,hidden] * cached[hidden,vocab].
	cachedWeights := []float32{
		1, 0, 2, -1, 0.25,
		0, 1, 1, 0.5, -0.5,
		2, -1, 0, 1, 0.75,
	}
	cached, err := gpu.Malloc(len(cachedWeights))
	if err != nil {
		t.Fatal(err)
	}
	defer cached.Free()
	if err := cached.Upload(cachedWeights); err != nil {
		t.Fatal(err)
	}
	scEmbed, err := gpu.Malloc(len(cachedWeights))
	if err != nil {
		t.Fatal(err)
	}
	defer scEmbed.Free()
	// scEmbed is [vocab,hidden] for prob[positions,vocab] * scEmbed[vocab,hidden].
	scRows := []float32{
		1, 0, 2,
		0, 1, -1,
		2, 1, 0,
		-1, 0.5, 1,
		0.25, -0.5, 0.75,
	}
	if err := scEmbed.Upload(scRows); err != nil {
		t.Fatal(err)
	}
	scratch := ForwardScratch{Hidden: []float32{0.5, -1, 2, 1.5, 0.25, -0.5}, SCTempInv: 1, FinalLogitSoftcapping: 30}
	arg, ent, samp, sc, state, err := runDenseF32GPULMHeadDeviceGraph(scratch, hidden, cached, vocab, hidden, []float64{0.1, 0.9}, scEmbed, vocab, hidden, true)
	if err != nil {
		t.Fatal(err)
	}
	if state == nil || state.Logits == nil || state.Positions != positions || state.Vocab != vocab || state.Hidden != hidden {
		t.Fatalf("bad device SC state: %+v", state)
	}
	defer state.Free()
	if len(arg) != positions || len(ent) != positions || len(samp) != positions || len(sc) != positions*hidden {
		t.Fatalf("bad device graph lengths arg=%d ent=%d samp=%d sc=%d", len(arg), len(ent), len(samp), len(sc))
	}
	for i, v := range ent {
		if v <= 0 || math.IsNaN(v) {
			t.Fatalf("entropy[%d]=%g", i, v)
		}
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
