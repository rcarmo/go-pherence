package diffusiongemma

import "testing"

func TestGGUFChunkedLMHeadScratchStatsAndFree(t *testing.T) {
	FreeGGUFChunkedLMHeadScratch()
	embed := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
	}
	ggufChunkedLMHeadScratch.Lock()
	_ = cachedF32LMHeadChunk(embed, 5, 3, 2, 0, 0, 2)
	_ = cachedF32LMHeadChunk(embed, 5, 3, 2, 1, 2, 4)
	ggufChunkedLMHeadScratch.Unlock()
	chunks, bytes := GGUFChunkedLMHeadScratchStats()
	if chunks != 2 || bytes != int64(2*3*2*4) {
		t.Fatalf("stats chunks=%d bytes=%d", chunks, bytes)
	}
	FreeGGUFChunkedLMHeadScratch()
	chunks, bytes = GGUFChunkedLMHeadScratchStats()
	if chunks != 0 || bytes != 0 {
		t.Fatalf("after free chunks=%d bytes=%d, want zero", chunks, bytes)
	}
}
