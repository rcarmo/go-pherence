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

func TestChunkCacheContainsTracksEviction(t *testing.T) {
	// Payload + token/key/entry accounting admits exactly one small entry.
	c := NewChunkCache(273)
	k1 := ChunkKey{ModelID: "m", TokenHash: 1}
	k2 := ChunkKey{ModelID: "m", TokenHash: 2}
	if c.Contains(k1) {
		t.Fatal("unexpected pre-insert hit")
	}
	if err := c.Put(k1, []int{1}, Snapshot{Hidden: []float32{1, 2}}); err != nil {
		t.Fatal(err)
	}
	if !c.Contains(k1) {
		t.Fatal("missing inserted key")
	}
	if err := c.Put(k2, []int{2}, Snapshot{Hidden: []float32{3, 4}}); err != nil {
		t.Fatal(err)
	}
	if c.Contains(k1) == c.Contains(k2) {
		t.Fatalf("expected exactly one key after eviction, got k1=%v k2=%v", c.Contains(k1), c.Contains(k2))
	}
}

func TestChunkCacheCloneAndEvict(t *testing.T) {
	c := NewChunkCache(400)
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
	big := Snapshot{SeqLen: 4, Hidden: make([]float32, 128)}
	if err := c.Put(k2, []int{3, 4}, big); err == nil {
		t.Fatal("oversized entry accepted")
	}
	if c.Len() != 1 || !c.Contains(k1) || c.Contains(k2) {
		t.Fatal("rejected insert evicted valid entry")
	}
}
