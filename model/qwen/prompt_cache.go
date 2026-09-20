package qwen

import (
	"fmt"
	"github.com/rcarmo/go-pherence/runtime/kv"
	"math"
	"slices"
	"sync"
)

const PromptCacheVersion = "qwen35-prompt-state-v1"

type PromptSnapshot struct {
	State   Qwen35BaseForwardState
	Next    int
	Logit   float32
	Hidden  []float32
	PreNorm []float32
	EndPos  int
}

type PromptCacheStats struct {
	MaxBytes       int64 `json:"max_bytes"`
	UsedBytes      int64 `json:"used_bytes"`
	Entries        int   `json:"entries"`
	SidecarEntries int   `json:"sidecar_entries"`
}

// PromptCache serialises its index and sidecar. Zero budget disables storage.
// Callers must not mutate input tokens/state during Store or lookup.
type PromptCache struct {
	mu      sync.Mutex
	cache   *kv.ChunkCache
	sidecar map[kv.ChunkKey]PromptSnapshot
}

func NewPromptCache(maxBytes int64) *PromptCache {
	return &PromptCache{cache: kv.NewChunkCache(maxBytes), sidecar: map[kv.ChunkKey]PromptSnapshot{}}
}

// PromptPrefixKey returns a zero key for an invalid chunk size.
func PromptPrefixKey(modelID, layout, dtype string, tokens []int, chunkSize int) kv.ChunkKey {
	if chunkSize <= 0 {
		return kv.ChunkKey{}
	}
	prev := uint64(0)
	for start := 0; start < len(tokens); {
		end := start + min(chunkSize, len(tokens)-start)
		prev = kv.HashTokenChunk(prev, tokens[start:end])
		start = end
	}
	return kv.ChunkKey{ModelID: modelID, Backend: "qwen36run", DType: dtype, LayerLayout: layout, TokenHash: prev, ChunkSize: chunkSize, EndPos: len(tokens)}
}

func ClonePromptSnapshot(s PromptSnapshot) PromptSnapshot {
	// Cache only live rows, not spare capacity reserved by generation state.
	state := Qwen35BaseForwardState{Pos: s.State.Pos, FullK: make([][]float32, len(s.State.FullK)), FullV: make([][]float32, len(s.State.FullV)), Linear: make([]Qwen35LinearAttentionState, len(s.State.Linear))}
	for i, v := range s.State.FullK {
		state.FullK[i] = append([]float32(nil), v...)
	}
	for i, v := range s.State.FullV {
		state.FullV[i] = append([]float32(nil), v...)
	}
	for i, v := range s.State.Linear {
		state.Linear[i] = CloneQwen35LinearAttentionState(v)
	}
	return PromptSnapshot{State: state, Next: s.Next, Logit: s.Logit, Hidden: append([]float32(nil), s.Hidden...), PreNorm: append([]float32(nil), s.PreNorm...), EndPos: s.EndPos}
}

// PromptSnapshotForBudget is a legacy payload-only reporting adapter. Store
// uses promptSnapshotBytes instead, including metadata and sidecar ownership.
// Returned layer rows borrow the input; Hidden is a copy. No admission uses it.
func PromptSnapshotForBudget(s PromptSnapshot) kv.Snapshot {
	layers := make([]kv.LayerKVSnapshot, 0, max(len(s.State.FullK), len(s.State.FullV))+len(s.State.Linear))
	for i := 0; i < max(len(s.State.FullK), len(s.State.FullV)); i++ {
		var k, v []float32
		if i < len(s.State.FullK) {
			k = s.State.FullK[i]
		}
		if i < len(s.State.FullV) {
			v = s.State.FullV[i]
		}
		layers = append(layers, kv.LayerKVSnapshot{K: k, V: v, SeqLen: s.State.Pos})
	}
	for _, lin := range s.State.Linear {
		layers = append(layers, kv.LayerKVSnapshot{K: lin.Conv, V: lin.SSM, SeqLen: lin.Pos})
	}
	hidden := make([]float32, 0, len(s.Hidden)+len(s.PreNorm))
	hidden = append(hidden, s.Hidden...)
	hidden = append(hidden, s.PreNorm...)
	return kv.Snapshot{SeqLen: s.State.Pos, Hidden: hidden, Layers: layers}
}

func (c *PromptCache) PruneSidecar() {
	if c == nil || c.cache == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.pruneSidecarLocked()
}

func (c *PromptCache) pruneSidecarLocked() {
	for key := range c.sidecar {
		if !c.cache.Contains(key) {
			delete(c.sidecar, key)
		}
	}
}

func (c *PromptCache) Store(modelID, layout, dtype string, tokens []int, chunkSize int, snap PromptSnapshot) bool {
	if c == nil || c.cache == nil || chunkSize <= 0 {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cache.MaxBytes() <= 0 {
		return false
	}
	key := PromptPrefixKey(modelID, layout, dtype, tokens, chunkSize)
	retained, err := promptSnapshotBytes(snap)
	if err != nil {
		return false
	}
	// Index retains tokens and charges the sidecar, not a second full KV copy.
	if err := c.cache.PutWithRetainedBytes(key, tokens, kv.Snapshot{}, retained); err != nil {
		return false
	}
	c.pruneSidecarLocked()
	entry, ok := c.cache.Get(key)
	if !ok {
		return false
	}
	delete(c.sidecar, key) // replacement drops old payload/key before copying
	c.sidecar[entry.Key] = ClonePromptSnapshot(snap)
	return true
}

func (c *PromptCache) FindLongest(modelID, layout, dtype string, tokens []int, chunkSize int) (PromptSnapshot, bool) {
	snap, _, ok := c.FindLongestWithKey(modelID, layout, dtype, tokens, chunkSize)
	return snap, ok
}

func (c *PromptCache) FindLongestWithKey(modelID, layout, dtype string, tokens []int, chunkSize int) (PromptSnapshot, kv.ChunkKey, bool) {
	if c == nil || c.cache == nil || chunkSize <= 0 {
		return PromptSnapshot{}, kv.ChunkKey{}, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for end := len(tokens); end > 0; {
		key := PromptPrefixKey(modelID, layout, dtype, tokens[:end], chunkSize)
		if entry, ok := c.cache.Get(key); ok && slices.Equal(entry.Tokens, tokens[:end]) {
			if snap, ok := c.sidecar[key]; ok {
				return ClonePromptSnapshot(snap), key, true
			}
		}
		if end%chunkSize != 0 {
			end = (end / chunkSize) * chunkSize
		} else {
			end -= chunkSize
		}
	}
	return PromptSnapshot{}, kv.ChunkKey{}, false
}

func (c *PromptCache) Stats() PromptCacheStats {
	if c == nil || c.cache == nil {
		return PromptCacheStats{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return PromptCacheStats{MaxBytes: c.cache.MaxBytes(), UsedBytes: c.cache.UsedBytes(), Entries: c.cache.Len(), SidecarEntries: len(c.sidecar)}
}

// Includes independently cloned K/V rows (including unmatched V), linear state,
// vector payload, row/slice headers and a conservative sidecar entry allowance.
func promptSnapshotBytes(s PromptSnapshot) (int64, error) {
	total := int64(256)
	add := func(n, unit int) bool {
		if n < 0 || int64(n) > (math.MaxInt64-total)/int64(unit) {
			return false
		}
		total += int64(n) * int64(unit)
		return true
	}
	if !add(len(s.State.FullK), 24) || !add(len(s.State.FullV), 24) || !add(len(s.State.Linear), 56) || !add(len(s.Hidden), 4) || !add(len(s.PreNorm), 4) {
		return 0, fmt.Errorf("prompt metadata overflow")
	}
	for _, rows := range [][][]float32{s.State.FullK, s.State.FullV} {
		for _, row := range rows {
			if !add(len(row), 4) {
				return 0, fmt.Errorf("prompt KV overflow")
			}
		}
	}
	for _, lin := range s.State.Linear {
		if !add(len(lin.Conv), 4) || !add(len(lin.SSM), 4) {
			return 0, fmt.Errorf("prompt linear state overflow")
		}
	}
	return total, nil
}
