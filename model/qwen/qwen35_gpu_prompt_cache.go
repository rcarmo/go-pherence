package qwen

import (
	"container/list"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

type GPUPromptCacheEntry struct {
	Key   kv.ChunkKey
	State *Qwen35GPUForwardState
	Bytes int64
}

type GPUPromptCacheStats struct {
	MaxBytes  int64 `json:"max_bytes"`
	UsedBytes int64 `json:"used_bytes"`
	Entries   int   `json:"entries"`
}

type GPUPromptCache struct {
	maxBytes  int64
	usedBytes int64
	ll        *list.List
	items     map[kv.ChunkKey]*list.Element
}

func NewGPUPromptCache(maxBytes int64) *GPUPromptCache {
	if maxBytes < 0 {
		maxBytes = 0
	}
	return &GPUPromptCache{maxBytes: maxBytes, ll: list.New(), items: map[kv.ChunkKey]*list.Element{}}
}

func (c *GPUPromptCache) Put(key kv.ChunkKey, state Qwen35BaseForwardState) bool {
	if c == nil || c.maxBytes <= 0 {
		return false
	}
	g, err := UploadQwen35ForwardStateGPU(state)
	if err != nil {
		return false
	}
	if g.Bytes > c.maxBytes {
		g.Free()
		return false
	}
	if old := c.items[key]; old != nil {
		ent := old.Value.(*GPUPromptCacheEntry)
		if ent.State != nil {
			ent.State.Free()
		}
		c.usedBytes -= ent.Bytes
		ent.State = g
		ent.Bytes = g.Bytes
		c.usedBytes += g.Bytes
		c.ll.MoveToFront(old)
		c.evict()
		return c.Contains(key)
	}
	ent := &GPUPromptCacheEntry{Key: key, State: g, Bytes: g.Bytes}
	el := c.ll.PushFront(ent)
	c.items[key] = el
	c.usedBytes += ent.Bytes
	c.evict()
	return c.Contains(key)
}

func (c *GPUPromptCache) Get(key kv.ChunkKey) (*Qwen35GPUForwardState, bool) {
	if c == nil {
		return nil, false
	}
	el := c.items[key]
	if el == nil {
		return nil, false
	}
	c.ll.MoveToFront(el)
	return el.Value.(*GPUPromptCacheEntry).State, true
}

func (c *GPUPromptCache) Contains(key kv.ChunkKey) bool {
	if c == nil {
		return false
	}
	return c.items[key] != nil
}

func (c *GPUPromptCache) Stats() GPUPromptCacheStats {
	if c == nil {
		return GPUPromptCacheStats{}
	}
	return GPUPromptCacheStats{MaxBytes: c.maxBytes, UsedBytes: c.usedBytes, Entries: len(c.items)}
}

func (c *GPUPromptCache) Free() {
	if c == nil {
		return
	}
	for _, el := range c.items {
		ent := el.Value.(*GPUPromptCacheEntry)
		if ent.State != nil {
			ent.State.Free()
		}
	}
	c.items = map[kv.ChunkKey]*list.Element{}
	c.ll.Init()
	c.usedBytes = 0
}

func (c *GPUPromptCache) evict() {
	for c.maxBytes > 0 && c.usedBytes > c.maxBytes && c.ll.Len() > 0 {
		el := c.ll.Back()
		ent := el.Value.(*GPUPromptCacheEntry)
		delete(c.items, ent.Key)
		if ent.State != nil {
			ent.State.Free()
		}
		c.usedBytes -= ent.Bytes
		if c.usedBytes < 0 {
			c.usedBytes = 0
		}
		c.ll.Remove(el)
	}
}
