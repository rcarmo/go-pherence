package kv

import (
	"container/list"
	"fmt"
	"hash/fnv"
)

type ChunkKey struct {
	ModelID     string
	Backend     string
	DType       string
	LayerLayout string
	TokenHash   uint64
	ChunkSize   int
	EndPos      int
}

type LayerKVSnapshot struct {
	K      []float32
	V      []float32
	SeqLen int
	KVDim  int
}

type Snapshot struct {
	Layers []LayerKVSnapshot
	SeqLen int
	Hidden []float32
}

type ChunkEntry struct {
	Key     ChunkKey
	Tokens  []int
	Snap    Snapshot
	Bytes   int64
	LastUse uint64
}

type ChunkCache struct {
	maxBytes int64
	used     int64
	tick     uint64
	ll       *list.List
	items    map[ChunkKey]*list.Element
}

func NewChunkCache(maxBytes int64) *ChunkCache {
	if maxBytes < 0 {
		maxBytes = 0
	}
	return &ChunkCache{maxBytes: maxBytes, ll: list.New(), items: map[ChunkKey]*list.Element{}}
}

func HashTokenChunk(prev uint64, tokens []int) uint64 {
	h := fnv.New64a()
	var buf [16]byte
	for i := 0; i < 8; i++ {
		buf[i] = byte(prev >> (8 * i))
	}
	_, _ = h.Write(buf[:8])
	for _, t := range tokens {
		u := uint64(uint(t))
		for i := 0; i < 8; i++ {
			buf[i] = byte(u >> (8 * i))
		}
		_, _ = h.Write(buf[:8])
	}
	return h.Sum64()
}

func EstimateSnapshotBytes(s Snapshot) (int64, error) {
	var n int64
	add := func(v int) error {
		if v < 0 {
			return fmt.Errorf("negative length %d", v)
		}
		if int64(v) > (1<<62-n)/4 {
			return fmt.Errorf("snapshot size overflow")
		}
		n += int64(v) * 4
		return nil
	}
	for _, l := range s.Layers {
		if err := add(len(l.K)); err != nil {
			return 0, err
		}
		if err := add(len(l.V)); err != nil {
			return 0, err
		}
	}
	if err := add(len(s.Hidden)); err != nil {
		return 0, err
	}
	return n, nil
}

func CloneSnapshot(s Snapshot) Snapshot {
	out := Snapshot{SeqLen: s.SeqLen, Hidden: append([]float32(nil), s.Hidden...), Layers: make([]LayerKVSnapshot, len(s.Layers))}
	for i, l := range s.Layers {
		out.Layers[i] = LayerKVSnapshot{K: append([]float32(nil), l.K...), V: append([]float32(nil), l.V...), SeqLen: l.SeqLen, KVDim: l.KVDim}
	}
	return out
}

func (c *ChunkCache) Put(key ChunkKey, tokens []int, snap Snapshot) error {
	if c == nil {
		return fmt.Errorf("nil chunk cache")
	}
	bytes, err := EstimateSnapshotBytes(snap)
	if err != nil {
		return err
	}
	if old := c.items[key]; old != nil {
		ent := old.Value.(*ChunkEntry)
		c.used -= ent.Bytes
		ent.Tokens = append(ent.Tokens[:0], tokens...)
		ent.Snap = CloneSnapshot(snap)
		ent.Bytes = bytes
		c.tick++
		ent.LastUse = c.tick
		c.used += bytes
		c.ll.MoveToFront(old)
		c.evict()
		return nil
	}
	c.tick++
	ent := &ChunkEntry{Key: key, Tokens: append([]int(nil), tokens...), Snap: CloneSnapshot(snap), Bytes: bytes, LastUse: c.tick}
	el := c.ll.PushFront(ent)
	c.items[key] = el
	c.used += bytes
	c.evict()
	return nil
}

func (c *ChunkCache) Get(key ChunkKey) (ChunkEntry, bool) {
	if c == nil {
		return ChunkEntry{}, false
	}
	el := c.items[key]
	if el == nil {
		return ChunkEntry{}, false
	}
	ent := el.Value.(*ChunkEntry)
	c.tick++
	ent.LastUse = c.tick
	c.ll.MoveToFront(el)
	return ChunkEntry{Key: ent.Key, Tokens: append([]int(nil), ent.Tokens...), Snap: CloneSnapshot(ent.Snap), Bytes: ent.Bytes, LastUse: ent.LastUse}, true
}

func (c *ChunkCache) UsedBytes() int64 {
	if c == nil {
		return 0
	}
	return c.used
}
func (c *ChunkCache) Len() int {
	if c == nil {
		return 0
	}
	return len(c.items)
}

func (c *ChunkCache) evict() {
	if c.maxBytes <= 0 {
		return
	}
	for c.used > c.maxBytes && c.ll.Len() > 0 {
		el := c.ll.Back()
		ent := el.Value.(*ChunkEntry)
		delete(c.items, ent.Key)
		c.used -= ent.Bytes
		if c.used < 0 {
			c.used = 0
		}
		c.ll.Remove(el)
	}
}
