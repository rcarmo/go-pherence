package diffusiongemma

import (
	"math"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

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
	if len(arg) != positions || len(ent) != positions || len(samp) != positions || len(sc) != 0 {
		t.Fatalf("bad device graph lengths arg=%d ent=%d samp=%d sc=%d", len(arg), len(ent), len(samp), len(sc))
	}
	for i, v := range ent {
		if v <= 0 || math.IsNaN(v) {
			t.Fatalf("entropy[%d]=%g", i, v)
		}
	}
}
