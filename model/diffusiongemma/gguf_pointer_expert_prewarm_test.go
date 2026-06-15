package diffusiongemma

import (
	"sync"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func syntheticPointerExpertIndex(t *testing.T) *GGUFExpertIndex {
	t.Helper()
	gateUp := &gguf.ExpertMatrices{Name: "synthetic.gate_up", QType: gguf.QuantQ4_K, InDim: 256, OutDim: 2, Experts: 2}
	gateRowBytes, err := gateUp.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	gateUp.Raw = make([]byte, gateRowBytes*gateUp.OutDim*gateUp.Experts)
	down := &gguf.ExpertMatrices{Name: "synthetic.down", QType: gguf.QuantQ8_0, InDim: 32, OutDim: 4, Experts: 2}
	downRowBytes, err := down.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	down.Raw = make([]byte, downRowBytes*down.OutDim*down.Experts)
	return &GGUFExpertIndex{
		NumLayers:    1,
		NumExperts:   2,
		Intermediate: 1,
		HiddenSize:   4,
		entries: []ggufLayerExperts{{
			gateUp: gateUp,
			down:   down,
		}},
	}
}

func resetGGUFExpertResidencyTestState() { FreeGGUFGPUExpertCaches() }

func TestGGUFGPUExpertPrewarmCacheReserveBytes(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_CACHE_RESERVE_MB", "")
	if got := diffusionGemmaGGUFGPUExpertPrewarmCacheReserveBytes(); got != 0 {
		t.Fatalf("empty cache reserve=%d, want 0", got)
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_CACHE_RESERVE_MB", "256")
	if got := diffusionGemmaGGUFGPUExpertPrewarmCacheReserveBytes(); got != 256*1024*1024 {
		t.Fatalf("cache reserve=%d, want 256MiB", got)
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_CACHE_RESERVE_MB", "bad")
	if got := diffusionGemmaGGUFGPUExpertPrewarmCacheReserveBytes(); got != 0 {
		t.Fatalf("invalid cache reserve=%d, want 0", got)
	}
}

func TestPrewarmGGUFGPUPointerExpertCacheSynthetic(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	resetGGUFExpertResidencyTestState()
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB", "8192")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_RESERVE_MB", "0")
	idx := syntheticPointerExpertIndex(t)
	layers, experts, bytes, err := PrewarmGGUFGPUPointerExpertCache(idx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if layers != 1 || experts != 2 || bytes <= 0 {
		t.Fatalf("prewarm layers=%d experts=%d bytes=%d, want 1/2/>0", layers, experts, bytes)
	}
	if _, err := residentQ4KGateUpExpertMatrix(idx, 0, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := residentQ8DownExpertMatrix(idx, 0, 0); err != nil {
		t.Fatal(err)
	}
	FreeGGUFGPUExpertCaches()
	used, _ := activeExpertMatrixCacheUsageBytes()
	if used != 0 {
		t.Fatalf("expert cache bytes after free=%d, want 0", used)
	}
}

func countSyncMapEntries(m *sync.Map) int {
	count := 0
	m.Range(func(_, _ any) bool {
		count++
		return true
	})
	return count
}

func TestPrewarmGGUFGPUPointerExpertCacheQ4OnlySkipsQ8(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	resetGGUFExpertResidencyTestState()
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB", "8192")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_RESERVE_MB", "0")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_Q4_ONLY", "1")
	idx := syntheticPointerExpertIndex(t)
	layers, experts, bytes, err := PrewarmGGUFGPUPointerExpertCache(idx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if layers != 1 || experts != 2 || bytes <= 0 {
		t.Fatalf("q4-only prewarm layers=%d experts=%d bytes=%d, want 1/2/>0", layers, experts, bytes)
	}
	if got := countSyncMapEntries(&q4KGateUpExpertCache); got != 2 {
		t.Fatalf("Q4 resident entries=%d want 2", got)
	}
	if got := countSyncMapEntries(&q8DownExpertCache); got != 0 {
		t.Fatalf("Q8 resident entries=%d want 0 in q4-only mode", got)
	}
	FreeGGUFGPUExpertCaches()
}

func TestActivePointerTableBudgetFailureDoesNotPolluteResidentCaches(t *testing.T) {
	resetGGUFExpertResidencyTestState()
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB", "")
	idx := syntheticPointerExpertIndex(t)
	if table, ok, err := activeQ4KGateUpPointerTable(idx, 0, []int{0, 1}); err != nil || ok || table != nil {
		t.Fatalf("Q4 pointer table under tiny budget table=%v ok=%v err=%v, want nil/false/nil", table, ok, err)
	}
	if got := countSyncMapEntries(&q4KGateUpExpertCache); got != 0 {
		t.Fatalf("Q4 resident cache entries after failed table=%d, want 0", got)
	}
	if table, ok, err := activeQ8DownPointerTable(idx, 0, []int{0, 1}); err != nil || ok || table != nil {
		t.Fatalf("Q8 pointer table under tiny budget table=%v ok=%v err=%v, want nil/false/nil", table, ok, err)
	}
	if got := countSyncMapEntries(&q8DownExpertCache); got != 0 {
		t.Fatalf("Q8 resident cache entries after failed table=%d, want 0", got)
	}
	used, _ := activeExpertMatrixCacheUsageBytes()
	if used != 0 {
		t.Fatalf("budget bytes after failed active pointer tables=%d, want 0", used)
	}
}

func TestPrewarmGGUFGPUPointerExpertCacheHonorsCacheReserve(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	resetGGUFExpertResidencyTestState()
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB", "1")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_RESERVE_MB", "0")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_CACHE_RESERVE_MB", "1")
	idx := syntheticPointerExpertIndex(t)
	layers, experts, bytes, err := PrewarmGGUFGPUPointerExpertCache(idx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if layers != 0 || experts != 0 || bytes != 0 {
		t.Fatalf("cache-reserved prewarm layers=%d experts=%d bytes=%d, want zero", layers, experts, bytes)
	}
	if used, _ := activeExpertMatrixCacheUsageBytes(); used != 0 {
		t.Fatalf("cache-reserved prewarm used bytes=%d, want 0", used)
	}
}

func TestPrewarmGGUFGPUPointerExpertCacheDisabledWithoutBudget(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	resetGGUFExpertResidencyTestState()
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB", "")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_RESERVE_MB", "0")
	idx := syntheticPointerExpertIndex(t)
	layers, experts, bytes, err := PrewarmGGUFGPUPointerExpertCache(idx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if layers != 0 || experts != 0 || bytes != 0 {
		t.Fatalf("disabled prewarm layers=%d experts=%d bytes=%d, want zero", layers, experts, bytes)
	}
}
