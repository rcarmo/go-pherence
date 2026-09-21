package kv

import (
	"strings"
	"sync"
	"testing"
)

func TestChunkCacheRejectsBeforeCopyAndPreservesOldEntry(t *testing.T) {
	c := NewChunkCache(1024)
	key := ChunkKey{ModelID: "m", TokenHash: 1}
	if err := c.Put(key, []int{1}, Snapshot{Hidden: []float32{1}}); err != nil {
		t.Fatal(err)
	}
	big := Snapshot{Layers: make([]LayerKVSnapshot, 10000)}
	if err := c.Put(key, nil, big); err == nil {
		t.Fatal("free layer metadata")
	}
	if allocations := testing.AllocsPerRun(5, func() { _ = c.Put(key, nil, big) }); allocations > 4 {
		t.Fatal("rejected metadata copied", allocations)
	}
	if entry, ok := c.Get(key); !ok || entry.Snap.Hidden[0] != 1 {
		t.Fatal("rejection replaced cached value")
	}
	if err := c.Put(ChunkKey{ModelID: strings.Repeat("m", 2048)}, nil, Snapshot{}); err == nil {
		t.Fatal("uncharged identity")
	}
	if err := c.Put(key, make([]int, 1024), Snapshot{}); err == nil {
		t.Fatal("uncharged tokens")
	}
	if err := NewChunkCache(0).Put(key, nil, Snapshot{}); err == nil {
		t.Fatal("zero budget not disabled")
	}
}
func TestChunkCacheConcurrentOwnership(t *testing.T) {
	c := NewChunkCache(4096)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := ChunkKey{TokenHash: uint64(i)}
			for j := 0; j < 50; j++ {
				_ = c.Put(key, []int{i}, Snapshot{Hidden: []float32{float32(i)}})
				c.Get(key)
				c.Contains(key)
				c.Len()
				c.UsedBytes()
			}
		}(i)
	}
	wg.Wait()
	if c.UsedBytes() > c.MaxBytes() {
		t.Fatal("over budget")
	}
}
