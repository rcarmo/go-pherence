package diffusiongemma

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestLocalFP8ExpertLayerCPUvsGPUParity(t *testing.T) {
	modelDir := filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8")
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors.index.json")); err != nil {
		t.Skip("local FP8 DiffusionGemma model not present")
	}
	if !gpu.SgemmReady() {
		t.Skip("CUDA SGEMM unavailable")
	}
	m, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := OpenTextWeights(modelDir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	fp8w, err := OpenFP8TextWeights(modelDir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer fp8w.Close()
	idx, err := BuildFP8ExpertIndex(fp8w, m.Shape.TextLayers, m.Shape.NumExperts)
	if err != nil {
		t.Fatal(err)
	}

	layer := 4 // first layer outside the 4-layer pinned-prefix experiment that drifted
	positions := 16
	hidden := idx.HiddenSize
	topK := m.Shape.TopKExperts
	if topK <= 0 {
		topK = 8
	}

	mkScratch := func() ForwardScratch {
		s := ForwardScratch{
			Residual:       make([]float32, positions*hidden),
			MoeOut:         make([]float32, positions*hidden),
			TopKIDs:        make([]int, positions*topK),
			TopKVals:       make([]float32, positions*topK),
			TopKExperts:    topK,
			FP8ExpertIndex: idx,
		}
		rng := rand.New(rand.NewSource(1234))
		for i := range s.Residual {
			s.Residual[i] = float32(rng.NormFloat64()) * 0.02
		}
		for p := 0; p < positions; p++ {
			var sum float32
			for k := 0; k < topK; k++ {
				eid := (p*17 + k*13) % idx.NumExperts
				w := float32(topK-k) / float32(topK*(topK+1)/2)
				s.TopKIDs[p*topK+k] = eid
				s.TopKVals[p*topK+k] = w
				sum += w
			}
			// Keep top-k weights normalized like router output; both paths consume the same weights.
			for k := 0; k < topK; k++ {
				s.TopKVals[p*topK+k] /= sum
			}
		}
		return s
	}

	cpuScratch := mkScratch()
	gpuScratch := mkScratch()
	if err := runFP8CPUExpertsIndexed(LayerOp{Layer: layer}, weights, cpuScratch, idx); err != nil {
		t.Fatalf("CPU indexed experts: %v", err)
	}
	cache := NewExpertLRUCache(2 << 30)
	defer cache.ClearAll()
	if err := runLRUCachedExperts(LayerOp{Layer: layer}, weights, gpuScratch, fp8w, cache); err != nil {
		t.Fatalf("GPU LRU experts: %v", err)
	}
	max, mean, maxIdx := maxMeanDiff(cpuScratch.MoeOut, gpuScratch.MoeOut)
	if max > 2e-2 || mean > 2e-4 {
		pos := maxIdx / hidden
		dim := maxIdx % hidden
		t.Fatalf("MoE layer diff max=%g mean=%g pos=%d dim=%d cpu=%g gpu=%g", max, mean, pos, dim, cpuScratch.MoeOut[maxIdx], gpuScratch.MoeOut[maxIdx])
	}
	if math.IsNaN(max) || math.IsNaN(mean) {
		t.Fatalf("NaN diff max=%g mean=%g", max, mean)
	}
}
