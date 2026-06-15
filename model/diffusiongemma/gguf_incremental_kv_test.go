package diffusiongemma

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func maxMeanKVDiff(a, b []float32) (float64, float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var maxDiff, sum float64
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxDiff {
			maxDiff = d
		}
		sum += d
	}
	if n > 0 {
		sum /= float64(n)
	}
	return maxDiff, sum
}

func TestLocalGGUFIncrementalPromptKVMatchesFullEncode(t *testing.T) {
	runLocalGGUFIncrementalPromptKVTest(t, 0)
}

func TestLocalGGUFIncrementalPromptKVMatchesFullEncodeWithTinySWA(t *testing.T) {
	runLocalGGUFIncrementalPromptKVTest(t, 2)
}

func runLocalGGUFIncrementalPromptKVTest(t *testing.T, slidingWindowOverride int) {
	t.Helper()
	if os.Getenv("GO_PHERENCE_RUN_LOCAL_GGUF_CPU_KV") != "1" {
		t.Skip("set GO_PHERENCE_RUN_LOCAL_GGUF_CPU_KV=1 to run local heavy GGUF CPU incremental KV parity fixture")
	}
	ggufPath := filepath.Join("..", "..", "..", "llama.cpp", "models", "diffusiongemma-gguf", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skip("local DiffusionGemma GGUF Q4_K_M reference not present")
	}
	modelDir := filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8")
	meta, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	weights, err := OpenTextWeightsFromGGUF(g, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	idx, err := BuildGGUFExpertIndex(g, meta.Shape.TextLayers, meta.Shape.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	disp := CPUDispatcher{GGUFExpertIndex: idx, SkipEviction: true}
	ops := BuildForwardOpPlan(meta.Shape, nil)
	buffers := BuildForwardBufferPlan(meta.Shape)
	if slidingWindowOverride > 0 {
		buffers.SlidingWindow = slidingWindowOverride
	}
	prefix := []int{105, 2364, 107}
	suffix := []int{98357, 4142}
	prefixKV, err := disp.EncodePrompt(prefix, weights, ops, buffers)
	if err != nil {
		t.Fatal(err)
	}
	appendedKV, err := disp.EncodePromptSuffixGGUF(suffix, prefixKV, weights, ops, buffers)
	if err != nil {
		t.Fatal(err)
	}
	fullPrompt := append(append([]int(nil), prefix...), suffix...)
	fullKV, err := disp.EncodePrompt(fullPrompt, weights, ops, buffers)
	if err != nil {
		t.Fatal(err)
	}
	if len(appendedKV) != len(fullKV) {
		t.Fatalf("appended layers=%d full layers=%d", len(appendedKV), len(fullKV))
	}
	const kvTolerance = 1e-3
	for layer := range fullKV {
		if appendedKV[layer].SeqLen != fullKV[layer].SeqLen || appendedKV[layer].KVHeads != fullKV[layer].KVHeads || appendedKV[layer].HeadDim != fullKV[layer].HeadDim {
			t.Fatalf("layer %d shape mismatch appended=%+v full=%+v", layer, appendedKV[layer], fullKV[layer])
		}
		if maxDiff, meanDiff := maxMeanKVDiff(fullKV[layer].Keys, appendedKV[layer].Keys); maxDiff > kvTolerance {
			t.Fatalf("layer %d K diff max=%g mean=%g tolerance=%g sliding=%d", layer, maxDiff, meanDiff, kvTolerance, slidingWindowOverride)
		}
		if maxDiff, meanDiff := maxMeanKVDiff(fullKV[layer].Values, appendedKV[layer].Values); maxDiff > kvTolerance {
			t.Fatalf("layer %d V diff max=%g mean=%g tolerance=%g sliding=%d", layer, maxDiff, meanDiff, kvTolerance, slidingWindowOverride)
		}
	}
}
