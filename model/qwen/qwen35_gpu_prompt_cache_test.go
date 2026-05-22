package qwen

import (
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestEstimateQwen35ForwardStateBytes(t *testing.T) {
	got, err := EstimateQwen35ForwardStateBytes(sampleQwen35ForwardState())
	if err != nil {
		t.Fatal(err)
	}
	// sample: FullK 5 + FullV 5 + linear conv 2 + ssm 3 + ssm 1 = 16 floats
	if got != 16*4 {
		t.Fatalf("bytes=%d want %d", got, 16*4)
	}
}

func TestGPUPromptCacheBudgetRejectionStats(t *testing.T) {
	if !nvidia.SgemmReady() {
		t.Skip("NVIDIA backend not available")
	}
	c := NewGPUPromptCache(1)
	defer c.Free()
	if c.Put(kv.ChunkKey{ModelID: "m"}, sampleQwen35ForwardState()) {
		t.Fatal("tiny budget accepted state")
	}
	if c.Stats().BudgetRejections == 0 {
		t.Fatalf("missing budget rejection: %+v", c.Stats())
	}
}

func TestGPUPromptCacheHeadroomStatsDefault(t *testing.T) {
	c := NewGPUPromptCacheWithHeadroom(1024, 256)
	st := c.Stats()
	if st.MaxBytes != 1024 || st.HeadroomBytes != 256 {
		t.Fatalf("bad stats: %+v", st)
	}
}

func TestGPUPromptCacheDisabled(t *testing.T) {
	c := NewGPUPromptCache(0)
	if c.Put(kv.ChunkKey{ModelID: "m"}, sampleQwen35ForwardState()) {
		t.Fatal("disabled cache stored entry")
	}
	if c.Stats().Entries != 0 {
		t.Fatalf("unexpected entries: %+v", c.Stats())
	}
}

func TestGPUPromptCachePutGetDownload(t *testing.T) {
	if !nvidia.SgemmReady() {
		t.Skip("NVIDIA backend not available")
	}
	c := NewGPUPromptCache(1 << 20)
	defer c.Free()
	key := kv.ChunkKey{ModelID: "m", TokenHash: 1}
	want := sampleQwen35ForwardState()
	if !c.Put(key, want) {
		t.Fatal("put failed")
	}
	gotGPU, ok := c.Get(key)
	if !ok || gotGPU == nil {
		t.Fatal("missing gpu state")
	}
	got, err := DownloadQwen35ForwardStateGPU(gotGPU)
	if err != nil {
		t.Fatal(err)
	}
	if got.Pos != want.Pos || len(got.FullK) != len(want.FullK) || got.FullK[0][0] != want.FullK[0][0] {
		t.Fatalf("round trip mismatch got=%+v want=%+v", got, want)
	}
	if c.Stats().UsedBytes == 0 || c.Stats().Entries != 1 {
		t.Fatalf("bad stats: %+v", c.Stats())
	}
}
