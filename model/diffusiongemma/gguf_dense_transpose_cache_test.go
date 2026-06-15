package diffusiongemma

import "testing"

func TestCachedTransposedF32MatrixNoEvictPreservesExistingEntries(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_DENSE_TRANSPOSE_CACHE_MB", "1")
	FreeGGUFDenseTransposeCache()
	ggufDenseTransposeCache.Lock()
	ggufDenseTransposeCache.bytes = 1024*1024 - 16
	ggufDenseTransposeCache.Unlock()
	a := []float32{1, 2, 3, 4}
	if got, hit, err := cachedTransposedF32MatrixNoEvict(a, 2, 2); err != nil || hit || got == nil {
		t.Fatalf("first tiny cache insert got=%v hit=%v err=%v", got, hit, err)
	}
	b := []float32{5, 6, 7, 8}
	got, hit, err := cachedTransposedF32MatrixNoEvict(b, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got != nil || hit {
		t.Fatalf("no-evict insert should refuse over-budget matrix got=%v hit=%v", got, hit)
	}
	entries, bytes := GGUFDenseTransposeCacheStats()
	if entries != 1 || bytes <= 0 {
		t.Fatalf("cache stats entries=%d bytes=%d, want one positive entry", entries, bytes)
	}
	if got, hit, err := cachedTransposedF32Matrix(a, 2, 2); err != nil || !hit || got == nil {
		t.Fatalf("existing entry should remain after no-evict refusal got=%v hit=%v err=%v", got, hit, err)
	}
	got, hit, err = cachedTransposedF32Matrix(b, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got != nil || hit {
		t.Fatalf("default cache insert should also refuse over-budget matrix got=%v hit=%v", got, hit)
	}
	FreeGGUFDenseTransposeCache()
	entries, bytes = GGUFDenseTransposeCacheStats()
	if entries != 0 || bytes != 0 {
		t.Fatalf("cache stats after free entries=%d bytes=%d, want zero", entries, bytes)
	}
}
