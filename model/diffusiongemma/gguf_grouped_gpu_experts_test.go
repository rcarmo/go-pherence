package diffusiongemma

import (
	"errors"
	"fmt"
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

func TestLocalGGUFPartialGroupedGPUCPUExpertsMatchCPUSelected(t *testing.T) {
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
	defer FreeGGUFGPUExpertCaches()
	hidden, positions, topK := meta.Shape.TextHiddenSize, 3, 3
	residual := make([]float32, positions*hidden)
	for i := range residual {
		residual[i] = float32((i%29)-14) * 0.005
	}
	ids := []int{0, 7, 13, 13, 7, 1, 0, 1, 13}
	vals := []float32{0.55, 0.30, 0.15, 0.50, 0.25, 0.25, 0.40, 0.35, 0.25}
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
	kept, dropped, err := SplitSelectedExpertGroupedArrays(ga, func(expert int) bool { return expert == 0 || expert == 7 })
	if err != nil {
		t.Fatal(err)
	}
	for _, expert := range kept.ActiveExperts {
		if _, err := residentQ4KGateUpExpertMatrix(idx, 0, expert); err != nil {
			t.Fatal(err)
		}
		if _, err := residentQ8DownExpertMatrix(idx, 0, expert); err != nil {
			t.Fatal(err)
		}
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
	partialScratch := mk()
	var bufs SelectedExpertGroupedArraysGPUBuffers
	defer bufs.Free()
	if err := bufs.Upload(kept); err != nil {
		t.Fatal(err)
	}
	used, err := runGGUFGPUExpertsGroupedFused(op, partialScratch, idx, normedRows, kept, &bufs)
	if err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Fatal("partial GPU kept executor not used")
	}
	if err := runGGUFCPUExpertsGroupedNoPostNorm(op, partialScratch, idx, normedRows, dropped); err != nil {
		t.Fatal(err)
	}
	postNorm2, err := loadFloatVector(weights, weights.ForwardPlan().Layers[0].PostFFNLayerNorm2)
	if err != nil {
		t.Fatal(err)
	}
	for off := 0; off < len(partialScratch.MoeOut); off += hidden {
		if !simd.RMSNormTo(partialScratch.MoeOut[off:off+hidden], postNorm2, 1e-6) {
			t.Fatal("post norm failed")
		}
	}
	var maxAbs, meanAbs float64
	for i := range cpuScratch.MoeOut {
		d := math.Abs(float64(cpuScratch.MoeOut[i] - partialScratch.MoeOut[i]))
		if d > maxAbs {
			maxAbs = d
		}
		meanAbs += d
	}
	meanAbs /= float64(len(cpuScratch.MoeOut))
	if maxAbs > 0.25 || meanAbs > 0.02 {
		t.Fatalf("partial grouped GPU+CPU experts diverged: max=%g mean=%g", maxAbs, meanAbs)
	}
}

func TestLocalGGUFPartialGroupedGPUCPUExpertsPromptScale(t *testing.T) {
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
	defer FreeGGUFGPUExpertCaches()
	hidden, positions, topK := meta.Shape.TextHiddenSize, 92, 8
	residual := make([]float32, positions*hidden)
	for i := range residual {
		residual[i] = float32((i%31)-15) * 0.004
	}
	ids := make([]int, positions*topK)
	vals := make([]float32, positions*topK)
	for pos := 0; pos < positions; pos++ {
		var sum float32
		for k := 0; k < topK; k++ {
			ids[pos*topK+k] = (pos*17 + k*23 + (pos/3)*5) % idx.NumExperts
			v := float32(topK-k) / float32(topK)
			vals[pos*topK+k] = v
			sum += v
		}
		for k := 0; k < topK; k++ {
			vals[pos*topK+k] /= sum
		}
	}
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
	keep := map[int]bool{0: true, 4: true, 7: true, 15: true, 35: true, 38: true, 69: true, 84: true, 89: true}
	kept, dropped, err := SplitSelectedExpertGroupedArrays(ga, func(expert int) bool { return keep[expert] })
	if err != nil {
		t.Fatal(err)
	}
	if len(kept.ActiveExperts) == 0 || len(dropped.ActiveExperts) == 0 {
		t.Fatalf("unexpected split kept=%d dropped=%d", len(kept.ActiveExperts), len(dropped.ActiveExperts))
	}
	for _, expert := range kept.ActiveExperts {
		if _, err := residentQ4KGateUpExpertMatrix(idx, 0, expert); err != nil {
			t.Fatal(err)
		}
		if _, err := residentQ8DownExpertMatrix(idx, 0, expert); err != nil {
			t.Fatal(err)
		}
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
	partialScratch := mk()
	var bufs SelectedExpertGroupedArraysGPUBuffers
	defer bufs.Free()
	if err := bufs.Upload(kept); err != nil {
		t.Fatal(err)
	}
	used, err := runGGUFGPUExpertsGroupedFused(op, partialScratch, idx, normedRows, kept, &bufs)
	if err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Fatal("prompt-scale partial GPU kept executor not used")
	}
	if err := runGGUFCPUExpertsGroupedNoPostNorm(op, partialScratch, idx, normedRows, dropped); err != nil {
		t.Fatal(err)
	}
	postNorm2, err := loadFloatVector(weights, weights.ForwardPlan().Layers[0].PostFFNLayerNorm2)
	if err != nil {
		t.Fatal(err)
	}
	for off := 0; off < len(partialScratch.MoeOut); off += hidden {
		if !simd.RMSNormTo(partialScratch.MoeOut[off:off+hidden], postNorm2, 1e-6) {
			t.Fatal("post norm failed")
		}
	}
	var maxAbs, meanAbs float64
	for i := range cpuScratch.MoeOut {
		d := math.Abs(float64(cpuScratch.MoeOut[i] - partialScratch.MoeOut[i]))
		if d > maxAbs {
			maxAbs = d
		}
		meanAbs += d
	}
	meanAbs /= float64(len(cpuScratch.MoeOut))
	if maxAbs > 0.25 || meanAbs > 0.02 {
		t.Fatalf("prompt-scale partial grouped GPU+CPU experts diverged: max=%g mean=%g kept=%d dropped=%d", maxAbs, meanAbs, len(kept.ActiveExperts), len(dropped.ActiveExperts))
	}
}

func TestLocalGGUFPartialGroupedGPUCPUExpertsPromptScaleLayer5(t *testing.T) {
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
	defer FreeGGUFGPUExpertCaches()
	layer := 5
	if idx.entries[layer].down.QType != gguf.QuantQ8_0 {
		t.Fatalf("layer %d down qtype=%s, want Q8_0", layer, idx.entries[layer].down.QType)
	}
	hidden, positions, topK := meta.Shape.TextHiddenSize, 92, 8
	residual := make([]float32, positions*hidden)
	for i := range residual {
		residual[i] = float32((i%37)-18) * 0.0035
	}
	hot := []int{99, 126, 58, 73, 100, 75, 28, 63}
	cold := []int{0, 1, 2, 3, 4, 5, 6, 7}
	ids := make([]int, positions*topK)
	vals := make([]float32, positions*topK)
	for pos := 0; pos < positions; pos++ {
		var sum float32
		for k := 0; k < topK; k++ {
			if k < 4 {
				ids[pos*topK+k] = hot[(pos+k)%len(hot)]
			} else {
				ids[pos*topK+k] = cold[(pos+k)%len(cold)]
			}
			v := float32(topK-k) / float32(topK)
			vals[pos*topK+k] = v
			sum += v
		}
		for k := 0; k < topK; k++ {
			vals[pos*topK+k] /= sum
		}
	}
	mk := func() ForwardScratch {
		return ForwardScratch{Residual: append([]float32(nil), residual...), MoeOut: make([]float32, len(residual)), TopKIDs: append([]int(nil), ids...), TopKVals: append([]float32(nil), vals...), TopKExperts: topK, GGUFExpertIndex: idx}
	}
	op := LayerOp{Layer: layer, Type: layerTypeAt(meta.Shape.LayerTypes, layer), Kind: OpExperts}
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
	if idx.entries[layer].downScale != nil {
		if err := ga.ApplyDownScalesByExpert(idx.entries[layer].downScale); err != nil {
			t.Fatal(err)
		}
	}
	keep := map[int]bool{}
	for _, expert := range hot {
		keep[expert] = true
	}
	kept, dropped, err := SplitSelectedExpertGroupedArrays(ga, func(expert int) bool { return keep[expert] })
	if err != nil {
		t.Fatal(err)
	}
	if len(kept.ActiveExperts) == 0 || len(dropped.ActiveExperts) == 0 {
		t.Fatalf("unexpected split kept=%d dropped=%d", len(kept.ActiveExperts), len(dropped.ActiveExperts))
	}
	for _, expert := range kept.ActiveExperts {
		if _, err := residentQ4KGateUpExpertMatrix(idx, layer, expert); err != nil {
			t.Fatal(err)
		}
		if _, err := residentQ8DownExpertMatrix(idx, layer, expert); err != nil {
			t.Fatal(err)
		}
	}
	preNorm2, err := loadFloatVector(weights, weights.ForwardPlan().Layers[layer].PreFFNLayerNorm2)
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
	partialScratch := mk()
	var bufs SelectedExpertGroupedArraysGPUBuffers
	defer bufs.Free()
	if err := bufs.Upload(kept); err != nil {
		t.Fatal(err)
	}
	used, err := runGGUFGPUExpertsGroupedFused(op, partialScratch, idx, normedRows, kept, &bufs)
	if err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Fatal("layer-5 prompt-scale partial GPU kept executor not used")
	}
	if err := runGGUFCPUExpertsGroupedNoPostNorm(op, partialScratch, idx, normedRows, dropped); err != nil {
		t.Fatal(err)
	}
	postNorm2, err := loadFloatVector(weights, weights.ForwardPlan().Layers[layer].PostFFNLayerNorm2)
	if err != nil {
		t.Fatal(err)
	}
	for off := 0; off < len(partialScratch.MoeOut); off += hidden {
		if !simd.RMSNormTo(partialScratch.MoeOut[off:off+hidden], postNorm2, 1e-6) {
			t.Fatal("post norm failed")
		}
	}
	var maxAbs, meanAbs float64
	for i := range cpuScratch.MoeOut {
		d := math.Abs(float64(cpuScratch.MoeOut[i] - partialScratch.MoeOut[i]))
		if d > maxAbs {
			maxAbs = d
		}
		meanAbs += d
	}
	meanAbs /= float64(len(cpuScratch.MoeOut))
	if maxAbs > 0.25 || meanAbs > 0.02 {
		t.Fatalf("layer-5 prompt-scale partial grouped GPU+CPU experts diverged: max=%g mean=%g kept=%d dropped=%d", maxAbs, meanAbs, len(kept.ActiveExperts), len(dropped.ActiveExperts))
	}
}

func TestLocalGGUFPartialLayer5ExactEncoderInputDelta(t *testing.T) {
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
	defer FreeGGUFGPUExpertCaches()
	layer := 5
	hot := []int{99, 126, 58, 73, 100, 75, 28, 63}
	keep := map[int]bool{}
	for _, expert := range hot {
		keep[expert] = true
		if _, err := residentQ4KGateUpExpertMatrix(idx, layer, expert); err != nil {
			t.Fatal(err)
		}
		if _, err := residentQ8DownExpertMatrix(idx, layer, expert); err != nil {
			t.Fatal(err)
		}
	}
	stop := errors.New("captured layer 5 exact input")
	oldHook := encoderGGUFMoEHook
	defer func() { encoderGGUFMoEHook = oldHook }()
	compared := false
	var capturedPreMaxAbs, capturedPreMeanAbs, capturedPostMaxAbs, capturedPostMeanAbs float64
	var capturedPreMaxIndex, capturedPostMaxIndex int
	encoderGGUFMoEHook = func(hookLayer int, lt string, weights *TextWeights, scratch ForwardScratch, idx *GGUFExpertIndex) error {
		if hookLayer != layer {
			return nil
		}
		compared = true
		positions := len(scratch.Residual) / idx.HiddenSize
		topK := scratch.TopKExperts
		if positions != 92 || topK != 8 {
			return fmt.Errorf("unexpected layer-5 capture shape positions=%d topK=%d", positions, topK)
		}
		op := LayerOp{Layer: hookLayer, Type: lt, Kind: OpExperts}
		cpuScratch := ForwardScratch{Residual: append([]float32(nil), scratch.Residual...), MoeOut: make([]float32, len(scratch.MoeOut)), TopKIDs: append([]int(nil), scratch.TopKIDs...), TopKVals: append([]float32(nil), scratch.TopKVals...), TopKExperts: topK, GGUFExpertIndex: idx}
		if err := runGGUFCPUExpertsIndexed(op, weights, cpuScratch, idx); err != nil {
			return err
		}
		work, err := FlattenSelectedExperts(scratch.TopKIDs, scratch.TopKVals, positions, topK, idx.NumExperts)
		if err != nil {
			return err
		}
		arr := BuildSelectedExpertWorkArrays(work)
		grouped, err := BuildSelectedExpertGroupedWork(arr, idx.NumExperts)
		if err != nil {
			return err
		}
		ga, err := BuildSelectedExpertGroupedArrays(arr, grouped)
		if err != nil {
			return err
		}
		if idx.entries[hookLayer].downScale != nil {
			if err := ga.ApplyDownScalesByExpert(idx.entries[hookLayer].downScale); err != nil {
				return err
			}
		}
		kept, dropped, err := SplitSelectedExpertGroupedArrays(ga, func(expert int) bool { return keep[expert] })
		if err != nil {
			return err
		}
		if len(kept.ActiveExperts) == 0 || len(dropped.ActiveExperts) == 0 {
			return fmt.Errorf("unexpected split kept=%d dropped=%d", len(kept.ActiveExperts), len(dropped.ActiveExperts))
		}
		preNorm2, err := loadFloatVector(weights, weights.ForwardPlan().Layers[hookLayer].PreFFNLayerNorm2)
		if err != nil {
			return err
		}
		normedRows := make([]float32, len(scratch.Residual))
		for pos := 0; pos < positions; pos++ {
			row := normedRows[pos*idx.HiddenSize : (pos+1)*idx.HiddenSize]
			copy(row, scratch.Residual[pos*idx.HiddenSize:(pos+1)*idx.HiddenSize])
			if !rmsNormForGroupedGPUTest(row, preNorm2) {
				return fmt.Errorf("rms norm failed")
			}
		}
		cpuNoPostScratch := ForwardScratch{Residual: append([]float32(nil), scratch.Residual...), MoeOut: make([]float32, len(scratch.MoeOut)), TopKIDs: append([]int(nil), scratch.TopKIDs...), TopKVals: append([]float32(nil), scratch.TopKVals...), TopKExperts: topK, GGUFExpertIndex: idx}
		if err := runGGUFCPUExpertsGroupedNoPostNorm(op, cpuNoPostScratch, idx, normedRows, ga); err != nil {
			return err
		}
		partialScratch := ForwardScratch{Residual: append([]float32(nil), scratch.Residual...), MoeOut: make([]float32, len(scratch.MoeOut)), TopKIDs: append([]int(nil), scratch.TopKIDs...), TopKVals: append([]float32(nil), scratch.TopKVals...), TopKExperts: topK, GGUFExpertIndex: idx}
		var bufs SelectedExpertGroupedArraysGPUBuffers
		defer bufs.Free()
		if err := bufs.Upload(kept); err != nil {
			return err
		}
		used, err := runGGUFGPUExpertsGroupedFused(op, partialScratch, idx, normedRows, kept, &bufs)
		if err != nil {
			return err
		}
		if !used {
			return fmt.Errorf("layer-5 exact-input partial GPU kept executor not used")
		}
		if err := runGGUFCPUExpertsGroupedNoPostNorm(op, partialScratch, idx, normedRows, dropped); err != nil {
			return err
		}
		compare := func(a, b []float32) (maxAbs float64, meanAbs float64, maxIndex int) {
			maxIndex = -1
			for i := range a {
				d := math.Abs(float64(a[i] - b[i]))
				if d > maxAbs {
					maxAbs = d
					maxIndex = i
				}
				meanAbs += d
			}
			meanAbs /= float64(len(a))
			return maxAbs, meanAbs, maxIndex
		}
		capturedPreMaxAbs, capturedPreMeanAbs, capturedPreMaxIndex = compare(cpuNoPostScratch.MoeOut, partialScratch.MoeOut)
		postNorm2, err := loadFloatVector(weights, weights.ForwardPlan().Layers[hookLayer].PostFFNLayerNorm2)
		if err != nil {
			return err
		}
		for off := 0; off < len(partialScratch.MoeOut); off += idx.HiddenSize {
			if !simd.RMSNormTo(cpuNoPostScratch.MoeOut[off:off+idx.HiddenSize], postNorm2, 1e-6) {
				return fmt.Errorf("cpu post norm failed")
			}
			if !simd.RMSNormTo(partialScratch.MoeOut[off:off+idx.HiddenSize], postNorm2, 1e-6) {
				return fmt.Errorf("partial post norm failed")
			}
		}
		capturedPostMaxAbs, capturedPostMeanAbs, capturedPostMaxIndex = compare(cpuNoPostScratch.MoeOut, partialScratch.MoeOut)
		if capturedPostMaxAbs > 0.35 || capturedPostMeanAbs > 0.02 {
			return fmt.Errorf("layer-5 exact-input partial grouped GPU+CPU experts unexpectedly diverged: pre_max=%g pre_mean=%g post_max=%g post_mean=%g kept=%d dropped=%d", capturedPreMaxAbs, capturedPreMeanAbs, capturedPostMaxAbs, capturedPostMeanAbs, len(kept.ActiveExperts), len(dropped.ActiveExperts))
		}
		return stop
	}
	prompt := make([]int, 92)
	for i := range prompt {
		prompt[i] = 100 + i
	}
	disp := CPUDispatcher{GGUFExpertIndex: idx, SkipEviction: true}
	ops := BuildForwardOpPlan(meta.Shape, nil)
	buffers := BuildForwardBufferPlan(meta.Shape)
	buffers.TopKExperts = 8
	_, err = disp.EncodePrompt(prompt, weights, ops, buffers)
	if !errors.Is(err, stop) {
		t.Fatalf("EncodePrompt err=%v want sentinel", err)
	}
	if !compared {
		t.Fatal("layer-5 hook did not run")
	}
	t.Logf("layer-5 exact-input partial grouped GPU+CPU delta: pre_max=%g pre_mean=%g pre_row=%d pre_dim=%d post_max=%g post_mean=%g post_row=%d post_dim=%d", capturedPreMaxAbs, capturedPreMeanAbs, capturedPreMaxIndex/idx.HiddenSize, capturedPreMaxIndex%idx.HiddenSize, capturedPostMaxAbs, capturedPostMeanAbs, capturedPostMaxIndex/idx.HiddenSize, capturedPostMaxIndex%idx.HiddenSize)
}

func rmsNormForGroupedGPUTest(row, w []float32) bool { return simd.RMSNormTo(row, w, 1e-6) }
