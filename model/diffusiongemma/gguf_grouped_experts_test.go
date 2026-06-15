package diffusiongemma

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestLocalGGUFGroupedCPUExpertsMatchIndexed(t *testing.T) {
	ggufPath := filepath.Join("..", "..", "..", "llama.cpp", "models", "diffusiongemma-gguf", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skip("local DiffusionGemma GGUF Q4_K_M reference not present")
	}
	meta, err := LoadMetadata(filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8"))
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
	hidden, positions, topK := meta.Shape.TextHiddenSize, 2, 3
	residual := make([]float32, positions*hidden)
	for i := range residual {
		residual[i] = float32((i%19)-9) * 0.01
	}
	ids := []int{0, 7, 13, 13, 7, 1}
	vals := []float32{0.55, 0.30, 0.15, 0.50, 0.25, 0.25}
	mk := func() ForwardScratch {
		return ForwardScratch{Residual: append([]float32(nil), residual...), MoeOut: make([]float32, len(residual)), TopKIDs: append([]int(nil), ids...), TopKVals: append([]float32(nil), vals...), TopKExperts: topK, GGUFExpertIndex: idx}
	}
	op := LayerOp{Layer: 0, Type: layerTypeAt(meta.Shape.LayerTypes, 0), Kind: OpExperts}
	base := mk()
	if err := runGGUFCPUExpertsIndexed(op, weights, base, idx); err != nil {
		t.Fatal(err)
	}
	work, err := FlattenSelectedExperts(ids, vals, positions, topK, idx.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	arr := BuildSelectedExpertWorkArrays(work)
	grouped, err := BuildSelectedExpertGroupedWork(arr, idx.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	ga, err := BuildSelectedExpertGroupedArrays(arr, grouped)
	if err != nil {
		t.Fatal(err)
	}
	if idx.entries[0].downScale != nil {
		if err := ga.ApplyDownScalesByExpert(idx.entries[0].downScale); err != nil {
			t.Fatal(err)
		}
	}
	got := mk()
	if err := runGGUFCPUExpertsGrouped(op, weights, got, idx, ga); err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i := range base.MoeOut {
		if d := math.Abs(float64(base.MoeOut[i] - got.MoeOut[i])); d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > 1e-4 {
		t.Fatalf("grouped experts maxAbs=%g", maxAbs)
	}
}

func TestLocalGGUFCPUExpertsWithNormedRowsMatchIndexed(t *testing.T) {
	ggufPath := filepath.Join("..", "..", "..", "llama.cpp", "models", "diffusiongemma-gguf", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skip("local DiffusionGemma GGUF Q4_K_M reference not present")
	}
	meta, err := LoadMetadata(filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8"))
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
	hidden, positions, topK := meta.Shape.TextHiddenSize, 3, 4
	residual := make([]float32, positions*hidden)
	for i := range residual {
		residual[i] = float32((i%23)-11) * 0.007
	}
	ids := []int{0, 7, 13, 21, 13, 7, 1, 3, 5, 9, 17, 29}
	vals := []float32{0.40, 0.30, 0.20, 0.10, 0.50, 0.25, 0.15, 0.10, 0.35, 0.30, 0.20, 0.15}
	mk := func() ForwardScratch {
		return ForwardScratch{Residual: append([]float32(nil), residual...), MoeOut: make([]float32, len(residual)), TopKIDs: append([]int(nil), ids...), TopKVals: append([]float32(nil), vals...), TopKExperts: topK, GGUFExpertIndex: idx}
	}
	op := LayerOp{Layer: 0, Type: layerTypeAt(meta.Shape.LayerTypes, 0), Kind: OpExperts}
	base := mk()
	if err := runGGUFCPUExpertsIndexed(op, weights, base, idx); err != nil {
		t.Fatal(err)
	}
	preNorm2, err := loadFloatVector(weights, weights.ForwardPlan().Layers[0].PreFFNLayerNorm2)
	if err != nil {
		t.Fatal(err)
	}
	normedRows := make([]float32, len(residual))
	for pos := 0; pos < positions; pos++ {
		row := normedRows[pos*hidden : (pos+1)*hidden]
		copy(row, residual[pos*hidden:(pos+1)*hidden])
		if !simd.RMSNormTo(row, preNorm2, 1e-6) {
			t.Fatalf("pre_norm_2 failed pos=%d", pos)
		}
	}
	got := mk()
	ResetGGUFCPUExpertTimingStats()
	if err := runGGUFCPUExpertsIndexedWithNormedRows(op, weights, got, idx, normedRows); err != nil {
		t.Fatal(err)
	}
	stats := ggufCPUExpertTimingSnapshot()
	if stats.NormNS != 0 {
		t.Fatalf("normed-row helper recorded norm time %d, want 0", stats.NormNS)
	}
	var maxAbs float64
	for i := range base.MoeOut {
		if d := math.Abs(float64(base.MoeOut[i] - got.MoeOut[i])); d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > 1e-4 {
		t.Fatalf("normed-row experts maxAbs=%g", maxAbs)
	}
}
