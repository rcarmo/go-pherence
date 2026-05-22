package kv

import "testing"

func TestHashTokenChunkChainsPrefix(t *testing.T) {
	a := HashTokenChunk(0, []int{1, 2})
	b := HashTokenChunk(0, []int{3, 4})
	ab := HashTokenChunk(a, []int{3, 4})
	bb := HashTokenChunk(b, []int{3, 4})
	if a == b || ab == bb {
		t.Fatalf("hashes did not encode prefix chain a=%x b=%x ab=%x bb=%x", a, b, ab, bb)
	}
}

func TestChunkCacheCloneAndEvict(t *testing.T) {
	c := NewChunkCache(64)
	k1 := ChunkKey{ModelID: "m", TokenHash: 1, EndPos: 2}
	s1 := Snapshot{SeqLen: 2, Hidden: []float32{1, 2}, Layers: []LayerKVSnapshot{{K: []float32{1, 2}, V: []float32{3, 4}, SeqLen: 2, KVDim: 1}}}
	if err := c.Put(k1, []int{1, 2}, s1); err != nil {
		t.Fatal(err)
	}
	got, ok := c.Get(k1)
	if !ok {
		t.Fatal("missing entry")
	}
	got.Snap.Hidden[0] = 99
	got.Tokens[0] = 99
	got2, ok := c.Get(k1)
	if !ok || got2.Snap.Hidden[0] != 1 || got2.Tokens[0] != 1 {
		t.Fatalf("cache did not clone values: %+v", got2)
	}
	k2 := ChunkKey{ModelID: "m", TokenHash: 2, EndPos: 4}
	big := Snapshot{SeqLen: 4, Hidden: make([]float32, 32)}
	if err := c.Put(k2, []int{3, 4}, big); err != nil {
		t.Fatal(err)
	}
	if c.Len() != 0 || c.UsedBytes() != 0 {
		t.Fatalf("oversized entry should evict all, len=%d used=%d", c.Len(), c.UsedBytes())
	}
}
