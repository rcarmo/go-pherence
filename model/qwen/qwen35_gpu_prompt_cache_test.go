package qwen

import (
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

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
