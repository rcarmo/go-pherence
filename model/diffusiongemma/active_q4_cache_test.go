package diffusiongemma

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestLocalActiveQ4KGateUpMatrixCacheReusesBuffer(t *testing.T) {
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
	first, err := activeQ4KGateUpMatrix(idx, 0, active)
	if err != nil {
		t.Fatal(err)
	}
	second, err := activeQ4KGateUpMatrix(idx, 0, active)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || first.Q == nil || second.Q == nil || first.Q.Ptr != second.Q.Ptr {
		t.Fatalf("active Q4 cache did not reuse buffer first=%p second=%p", first, second)
	}
}

func TestLocalActiveQ4KGateUpPointerTableReusesResidentExperts(t *testing.T) {
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
	first, ok, err := activeQ4KGateUpPointerTable(idx, 0, active)
	if err != nil || !ok {
		t.Fatalf("first pointer table ok=%v err=%v", ok, err)
	}
	residentFirst, err := residentQ4KGateUpExpertMatrix(idx, 0, active[0])
	if err != nil {
		t.Fatal(err)
	}
	second, ok, err := activeQ4KGateUpPointerTable(idx, 0, active)
	if err != nil || !ok {
		t.Fatalf("second pointer table ok=%v err=%v", ok, err)
	}
	residentSecond, err := residentQ4KGateUpExpertMatrix(idx, 0, active[0])
	if err != nil {
		t.Fatal(err)
	}
	if first != second || first.QPtrs == nil || second.QPtrs == nil || first.QPtrs.Ptr != second.QPtrs.Ptr {
		t.Fatalf("active Q4 pointer table cache did not reuse table first=%p second=%p", first, second)
	}
	if residentFirst != residentSecond || residentFirst.Q == nil || residentSecond.Q == nil || residentFirst.Q.Ptr != residentSecond.Q.Ptr {
		t.Fatalf("resident Q4 expert cache did not reuse expert first=%p second=%p", residentFirst, residentSecond)
	}
}

func TestLocalActiveQ4KGateUpPointerTableHonorsBudget(t *testing.T) {
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
	_, ok, err := activeQ4KGateUpPointerTable(idx, 0, []int{2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatalf("active Q4 pointer table unexpectedly fit inside tiny budget")
	}
}

func TestLocalActiveQ4KGateUpMatrixHonorsBudget(t *testing.T) {
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
	_, err = activeQ4KGateUpMatrix(idx, 0, []int{2, 3, 4})
	if !errors.Is(err, errActiveExpertMatrixCacheBudget) {
		t.Fatalf("active Q4 cache err=%v want budget err", err)
	}
}
