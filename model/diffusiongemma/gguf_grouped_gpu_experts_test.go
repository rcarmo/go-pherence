package diffusiongemma

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestLocalGGUFGroupedGPUExpertsMatchCPUSelected(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB", "1024")
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
		residual[i] = float32((i%23)-11) * 0.007
	}
	ids := []int{0, 7, 13, 13, 7, 1}
	vals := []float32{0.55, 0.30, 0.15, 0.50, 0.25, 0.25}
	mk := func() ForwardScratch {
		return ForwardScratch{Residual: append([]float32(nil), residual...), MoeOut: make([]float32, len(residual)), TopKIDs: append([]int(nil), ids...), TopKVals: append([]float32(nil), vals...), TopKExperts: topK, GGUFExpertIndex: idx}
	}
	op := LayerOp{Layer: 0, Type: layerTypeAt(meta.Shape.LayerTypes, 0), Kind: OpExperts}
	cpuScratch := mk()
	if err := runGGUFCPUExpertsIndexed(op, weights, cpuScratch, idx); err != nil {
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
	var bufs SelectedExpertGroupedArraysGPUBuffers
	defer bufs.Free()
	if err := bufs.Upload(ga); err != nil {
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
		if !rmsNormForGroupedGPUTest(row, preNorm2) {
			t.Fatal("rms norm failed")
		}
	}
	gpuScratch := mk()
	used, err := runGGUFGPUExpertsGrouped(op, weights, gpuScratch, idx, normedRows, ga, &bufs)
	if err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Fatal("grouped GPU executor not used")
	}
	postNorm2, err := loadFloatVector(weights, weights.ForwardPlan().Layers[0].PostFFNLayerNorm2)
	if err != nil {
		t.Fatal(err)
	}
	for off := 0; off < len(gpuScratch.MoeOut); off += hidden {
		if !simd.RMSNormTo(gpuScratch.MoeOut[off:off+hidden], postNorm2, 1e-6) {
			t.Fatal("post norm failed")
		}
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
		t.Fatalf("grouped GPU experts diverged: max=%g mean=%g", maxAbs, meanAbs)
	}
}

func rmsNormForGroupedGPUTest(row, w []float32) bool { return simd.RMSNormTo(row, w, 1e-6) }
