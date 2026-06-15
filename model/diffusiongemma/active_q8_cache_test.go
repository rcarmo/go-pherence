package diffusiongemma

import (
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestLocalActiveQ8DownPointerTableReusesResidentExperts(t *testing.T) {
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
	idx, err := BuildGGUFExpertIndex(g, meta.Shape.TextLayers, meta.Shape.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	active := []int{0, 7, 13}
	first, ok, err := activeQ8DownPointerTable(idx, 0, active)
	if err != nil || !ok {
		t.Fatalf("first pointer table ok=%v err=%v", ok, err)
	}
	residentFirst, err := residentQ8DownExpertMatrix(idx, 0, active[0])
	if err != nil {
		t.Fatal(err)
	}
	second, ok, err := activeQ8DownPointerTable(idx, 0, active)
	if err != nil || !ok {
		t.Fatalf("second pointer table ok=%v err=%v", ok, err)
	}
	residentSecond, err := residentQ8DownExpertMatrix(idx, 0, active[0])
	if err != nil {
		t.Fatal(err)
	}
	if first != second || first.QPtrs == nil || second.QPtrs == nil || first.QPtrs.Ptr != second.QPtrs.Ptr {
		t.Fatalf("active Q8 pointer table cache did not reuse table first=%p second=%p", first, second)
	}
	if residentFirst != residentSecond || residentFirst.Q == nil || residentSecond.Q == nil || residentFirst.Q.Ptr != residentSecond.Q.Ptr {
		t.Fatalf("resident Q8 expert cache did not reuse expert first=%p second=%p", residentFirst, residentSecond)
	}
}

func TestLocalActiveQ8DownPointerTableHonorsBudget(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB", "1")
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
	idx, err := BuildGGUFExpertIndex(g, meta.Shape.TextLayers, meta.Shape.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	_, ok, err := activeQ8DownPointerTable(idx, 0, []int{2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatalf("active Q8 pointer table unexpectedly fit inside tiny budget")
	}
}

func TestLocalResidentGGUFGPUExpertWeightsCacheReusesQ8Down(t *testing.T) {
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
	idx, err := BuildGGUFExpertIndex(g, meta.Shape.TextLayers, meta.Shape.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	first, ok, err := residentGGUFGPUExpertWeights(idx, 0, 7)
	if err != nil || !ok {
		t.Fatalf("first ok=%v err=%v", ok, err)
	}
	second, ok, err := residentGGUFGPUExpertWeights(idx, 0, 7)
	if err != nil || !ok {
		t.Fatalf("second ok=%v err=%v", ok, err)
	}
	if first != second || first.DownQ8 == nil || second.DownQ8 == nil || first.DownQ8.Q.Ptr != second.DownQ8.Q.Ptr {
		t.Fatalf("Q8 down cache did not reuse buffer first=%p second=%p", first, second)
	}
}
