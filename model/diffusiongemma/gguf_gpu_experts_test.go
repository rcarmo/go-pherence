package diffusiongemma

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestLocalGGUFGPUExpertsMatchCPUSelected(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB", "1024")
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
	hidden := meta.Shape.TextHiddenSize
	positions, topK := 2, 3
	residual := make([]float32, positions*hidden)
	for i := range residual {
		residual[i] = float32((i%17)-8) * 0.01
	}
	ids := []int{0, 7, 13, 13, 7, 1}
	vals := []float32{0.55, 0.30, 0.15, 0.50, 0.25, 0.25}
	mkScratch := func() ForwardScratch {
		return ForwardScratch{
			Residual:        append([]float32(nil), residual...),
			MoeOut:          make([]float32, len(residual)),
			TopKIDs:         append([]int(nil), ids...),
			TopKVals:        append([]float32(nil), vals...),
			TopKExperts:     topK,
			GGUFExpertIndex: idx,
		}
	}
	op := LayerOp{Layer: 0, Type: layerTypeAt(meta.Shape.LayerTypes, 0), Kind: OpExperts}
	cpuScratch := mkScratch()
	if err := runGGUFCPUExpertsIndexed(op, weights, cpuScratch, idx); err != nil {
		t.Fatal(err)
	}
	gpuScratch := mkScratch()
	used, _, err := runGGUFGPUExpertsIndexed(op, weights, gpuScratch, idx)
	if err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Fatal("GGUF GPU expert path was not used")
	}
	var maxAbs, meanAbs float64
	for i := range cpuScratch.MoeOut {
		d := math.Abs(float64(cpuScratch.MoeOut[i] - gpuScratch.MoeOut[i]))
		if d > maxAbs {
			maxAbs = d
		}
		meanAbs += d
	}
	meanAbs /= float64(len(cpuScratch.MoeOut))
	if maxAbs > 0.25 || meanAbs > 0.02 {
		t.Fatalf("GPU experts diverged from CPU: max_abs=%g mean_abs=%g", maxAbs, meanAbs)
	}
}
