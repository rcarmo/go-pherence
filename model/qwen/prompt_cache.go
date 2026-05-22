package qwen

import "github.com/rcarmo/go-pherence/runtime/kv"

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

type PromptCache struct {
	cache   *kv.ChunkCache
	sidecar map[kv.ChunkKey]PromptSnapshot
}

func NewPromptCache(maxBytes int64) *PromptCache {
	return &PromptCache{cache: kv.NewChunkCache(maxBytes), sidecar: map[kv.ChunkKey]PromptSnapshot{}}
}

func PromptPrefixKey(modelID, layout, dtype string, tokens []int, chunkSize int) kv.ChunkKey {
	prev := uint64(0)
	for start := 0; start < len(tokens); start += chunkSize {
		end := start + chunkSize
		if end > len(tokens) {
			end = len(tokens)
		}
		prev = kv.HashTokenChunk(prev, tokens[start:end])
	}
	return kv.ChunkKey{ModelID: modelID, Backend: "qwen36run", DType: dtype, LayerLayout: layout, TokenHash: prev, ChunkSize: chunkSize, EndPos: len(tokens)}
}

func ClonePromptSnapshot(s PromptSnapshot) PromptSnapshot {
	return PromptSnapshot{State: CloneQwen35BaseForwardState(s.State), Next: s.Next, Logit: s.Logit, Hidden: append([]float32(nil), s.Hidden...), PreNorm: append([]float32(nil), s.PreNorm...), EndPos: s.EndPos}
}

func PromptSnapshotForBudget(s PromptSnapshot) kv.Snapshot {
	layers := make([]kv.LayerKVSnapshot, 0, len(s.State.FullK)+len(s.State.Linear))
	for i := range s.State.FullK {
		var v []float32
		if i < len(s.State.FullV) {
			v = s.State.FullV[i]
		}
		layers = append(layers, kv.LayerKVSnapshot{K: s.State.FullK[i], V: v, SeqLen: s.State.Pos})
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
	for key := range c.sidecar {
		if !c.cache.Contains(key) {
			delete(c.sidecar, key)
		}
	}
}

func (c *PromptCache) Store(modelID, layout, dtype string, tokens []int, chunkSize int, snap PromptSnapshot) bool {
	if c == nil || c.cache == nil {
		return false
	}
	key := PromptPrefixKey(modelID, layout, dtype, tokens, chunkSize)
	stored := ClonePromptSnapshot(snap)
	if err := c.cache.Put(key, tokens, PromptSnapshotForBudget(stored)); err != nil {
		c.PruneSidecar()
		return false
	}
	if !c.cache.Contains(key) {
		delete(c.sidecar, key)
		c.PruneSidecar()
		return false
	}
	c.sidecar[key] = stored
	c.PruneSidecar()
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
	for end := len(tokens); end > 0; {
		key := PromptPrefixKey(modelID, layout, dtype, tokens[:end], chunkSize)
		if _, ok := c.cache.Get(key); ok {
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
	return PromptCacheStats{MaxBytes: c.cache.MaxBytes(), UsedBytes: c.cache.UsedBytes(), Entries: c.cache.Len(), SidecarEntries: len(c.sidecar)}
}
