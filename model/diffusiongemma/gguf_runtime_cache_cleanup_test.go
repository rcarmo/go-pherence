package diffusiongemma

import (
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestFreeGGUFGPURuntimeCachesResetsHostCachesAndStats(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_DENSE_TRANSPOSE_CACHE_MB", "1")
	FreeGGUFGPURuntimeCaches()

	// Populate dense transpose cache.
	w := []float32{1, 2, 3, 4}
	if got, _, err := cachedTransposedF32Matrix(w, 2, 2); err != nil || got == nil {
		t.Fatalf("populate dense transpose got=%v err=%v", got, err)
	}
	// Populate temp dense upload stats/counters (without requiring CUDA buffers).
	ggufTempDenseUploadCounters.calls.Add(1)
	ggufTempDenseUploadCounters.bytes.Add(16)
	// Populate chunked LM-head host cache.
	embed := []float32{1, 2, 3, 4, 5, 6}
	ggufChunkedLMHeadScratch.Lock()
	_ = cachedF32LMHeadChunk(embed, 3, 2, 2, 0, 0, 2)
	ggufChunkedLMHeadScratch.Unlock()
	// Populate resident dense GPU weight cache maps without requiring CUDA buffers.
	ggufGPUAttentionCache.Store(ggufGPUAttentionKey{layer: 99, qName: "q", kName: "k", oName: "o"}, &GGUFGPUAttentionWeights{QRows: 1, KRows: 1, VRows: 1, Hidden: 1})
	ggufGPUMLPCache.Store(ggufGPUMLPKey{layer: 99, gate: "g", up: "u", down: "d"}, &GGUFGPUMLPWeights{Hidden: 1, Intermediate: 1})
	residentF32MatrixCache.Store(residentF32MatrixKey{ptr: 1, m: 2, k: 2}, &gpu.Buffer{Size: 16})
	// Populate reusable GPU scratch maps with zero-pointer buffers so cleanup can
	// verify accounting without requiring CUDA allocation in this unit test.
	ggufGPUFusedExpertScratch.Lock()
	ggufGPUFusedExpertScratch.residual = &gpu.Buffer{Size: 16}
	ggufGPUFusedExpertScratch.residualN = 4
	ggufGPUFusedExpertScratch.Unlock()
	// Populate expert/attention diagnostic counters.
	ggufExpertDispatchCounters.fusedUsed.Add(1)
	ggufAttentionTimingCounters.calls.Add(1)
	ggufCPUExpertTimingCounters.calls.Add(1)

	attnEntries, mlpEntries, _ := GGUFGPUWeightCacheStats()
	if attnEntries != 1 || mlpEntries != 1 {
		t.Fatalf("resident dense cache entries attn=%d mlp=%d, want 1/1", attnEntries, mlpEntries)
	}
	f32Entries, f32Bytes := ResidentF32MatrixCacheStats()
	if f32Entries != 1 || f32Bytes != 16 {
		t.Fatalf("resident F32 cache entries=%d bytes=%d, want 1/16", f32Entries, f32Bytes)
	}
	entries, denseBytes := GGUFDenseTransposeCacheStats()
	if entries == 0 || denseBytes == 0 {
		t.Fatalf("dense cache was not populated entries=%d bytes=%d", entries, denseBytes)
	}
	chunks, chunkBytes := GGUFChunkedLMHeadScratchStats()
	if chunks == 0 || chunkBytes == 0 {
		t.Fatalf("lmhead cache was not populated chunks=%d bytes=%d", chunks, chunkBytes)
	}
	scratchBuffers, scratchBytes := GGUFGPUScratchStats()
	if scratchBuffers == 0 || scratchBytes == 0 {
		t.Fatalf("gpu scratch was not populated buffers=%d bytes=%d", scratchBuffers, scratchBytes)
	}

	FreeGGUFGPURuntimeCaches()
	attnEntries, mlpEntries, _ = GGUFGPUWeightCacheStats()
	if attnEntries != 0 || mlpEntries != 0 {
		t.Fatalf("resident dense cache after cleanup attn=%d mlp=%d, want zero", attnEntries, mlpEntries)
	}
	f32Entries, f32Bytes = ResidentF32MatrixCacheStats()
	if f32Entries != 0 || f32Bytes != 0 {
		t.Fatalf("resident F32 cache after cleanup entries=%d bytes=%d, want zero", f32Entries, f32Bytes)
	}
	entries, denseBytes = GGUFDenseTransposeCacheStats()
	if entries != 0 || denseBytes != 0 {
		t.Fatalf("dense cache after cleanup entries=%d bytes=%d", entries, denseBytes)
	}
	chunks, chunkBytes = GGUFChunkedLMHeadScratchStats()
	if chunks != 0 || chunkBytes != 0 {
		t.Fatalf("lmhead cache after cleanup chunks=%d bytes=%d", chunks, chunkBytes)
	}
	scratchBuffers, scratchBytes = GGUFGPUScratchStats()
	if scratchBuffers != 0 || scratchBytes != 0 {
		t.Fatalf("gpu scratch after cleanup buffers=%d bytes=%d", scratchBuffers, scratchBytes)
	}
	if s := ggufTempDenseUploadSnapshot(); s != (ggufTempDenseUploadStats{}) {
		t.Fatalf("temp dense stats after cleanup=%+v", s)
	}
	if s := ggufExpertDispatchStatsSnapshot(); s != (ggufExpertDispatchStats{}) {
		t.Fatalf("expert stats after cleanup=%+v", s)
	}
	if s := ggufAttentionTimingSnapshot(); s != (ggufAttentionTimingStats{}) {
		t.Fatalf("attention stats after cleanup=%+v", s)
	}
	if s := ggufCPUExpertTimingSnapshot(); s != (ggufCPUExpertTimingStats{}) {
		t.Fatalf("cpu expert stats after cleanup=%+v", s)
	}
	FreeGGUFGPURuntimeCaches()
	entries, denseBytes = GGUFDenseTransposeCacheStats()
	chunks, chunkBytes = GGUFChunkedLMHeadScratchStats()
	scratchBuffers, scratchBytes = GGUFGPUScratchStats()
	if entries != 0 || denseBytes != 0 || chunks != 0 || chunkBytes != 0 || scratchBuffers != 0 || scratchBytes != 0 {
		t.Fatalf("second cleanup left state dense=%d/%d chunks=%d/%d scratch=%d/%d", entries, denseBytes, chunks, chunkBytes, scratchBuffers, scratchBytes)
	}
}
