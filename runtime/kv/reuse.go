package kv

import (
	"container/list"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"sync"
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

// ChunkCache owns cloned tokens/snapshots. Inputs must remain immutable during
// calls. A zero (or negative) byte budget disables storage; accounting includes
// payload and conservative metadata allowances, not allocator/RSS guarantees.
type ChunkCache struct {
	mu       sync.Mutex
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
	return c.PutWithRetainedBytes(key, tokens, snap, 0)
}

// PutWithRetainedBytes charges additional caller-owned payload tied to an entry
// (e.g. a model sidecar) without cloning a duplicate snapshot. The caller must
// serialise its sidecar publication and prune it on cache eviction. Rejection
// occurs before any copy and preserves existing entries.
func (c *ChunkCache) PutWithRetainedBytes(key ChunkKey, tokens []int, snap Snapshot, retained int64) error {
	if c == nil {
		return fmt.Errorf("nil chunk cache")
	}
	bytes, err := EstimateChunkEntryBytes(key, len(tokens), snap, retained)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.maxBytes <= 0 || bytes > c.maxBytes {
		return fmt.Errorf("chunk cache entry exceeds budget")
	}
	// Remove old/evicted references before copying; subtraction avoids sum wrap.
	if old := c.items[key]; old != nil {
		c.remove(old)
	}
	for c.used > c.maxBytes-bytes {
		c.remove(c.ll.Back())
	}
	key.ModelID = strings.Clone(key.ModelID)
	key.Backend = strings.Clone(key.Backend)
	key.DType = strings.Clone(key.DType)
	key.LayerLayout = strings.Clone(key.LayerLayout)
	c.tick++
	ent := &ChunkEntry{Key: key, Tokens: append([]int(nil), tokens...), Snap: CloneSnapshot(snap), Bytes: bytes, LastUse: c.tick}
	el := c.ll.PushFront(ent)
	c.items[key] = el
	c.used += bytes
	return nil
}

func (c *ChunkCache) Get(key ChunkKey) (ChunkEntry, bool) {
	if c == nil {
		return ChunkEntry{}, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
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

func (c *ChunkCache) Contains(key ChunkKey) bool {
	if c == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.items[key] != nil
}

func (c *ChunkCache) UsedBytes() int64 {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.used
}

func (c *ChunkCache) MaxBytes() int64 {
	if c == nil {
		return 0
	}
	return c.maxBytes
}

func (c *ChunkCache) Len() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.items)
}

// remove requires c.mu.
func (c *ChunkCache) remove(el *list.Element) {
	ent := el.Value.(*ChunkEntry)
	delete(c.items, ent.Key)
	c.used -= ent.Bytes
	c.ll.Remove(el)
}

// EstimateChunkEntryBytes includes copied identity/tokens, layer headers and a
// fixed entry allowance. Additional retained storage must already be checked.
func EstimateChunkEntryBytes(key ChunkKey, tokens int, snap Snapshot, retained int64) (int64, error) {
	payload, err := EstimateSnapshotBytes(snap)
	if err != nil {
		return 0, err
	}
	if tokens < 0 || retained < 0 {
		return 0, fmt.Errorf("negative retained size")
	}
	total := payload
	add := func(n int64) bool {
		if n < 0 || n > math.MaxInt64-total {
			return false
		}
		total += n
		return true
	}
	if int64(tokens) > math.MaxInt64/8 || int64(len(snap.Layers)) > math.MaxInt64/64 {
		return 0, fmt.Errorf("entry size overflow")
	}
	for _, n := range []int64{256, int64(tokens) * 8, int64(len(snap.Layers)) * 64, int64(len(key.ModelID)), int64(len(key.Backend)), int64(len(key.DType)), int64(len(key.LayerLayout)), retained} {
		if !add(n) {
			return 0, fmt.Errorf("entry size overflow")
		}
	}
	return total, nil
}
