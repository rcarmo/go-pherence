package qwen

import (
	"container/list"
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

type GPUPromptCacheEntry struct {
	Key       kv.ChunkKey
	State     *Qwen35GPUForwardState
	StateBF16 *Qwen35GPUForwardStateBF16
	Bytes     int64
}

type GPUPromptCacheStats struct {
	MaxBytes           int64  `json:"max_bytes"`
	UsedBytes          int64  `json:"used_bytes"`
	Entries            int    `json:"entries"`
	HeadroomBytes      uint64 `json:"headroom_bytes,omitempty"`
	LastEstimateBytes  int64  `json:"last_estimate_bytes,omitempty"`
	LastFreeBytes      uint64 `json:"last_free_bytes,omitempty"`
	Compressed         bool   `json:"compressed,omitempty"`
	UploadFailures     int64  `json:"upload_failures,omitempty"`
	BudgetRejections   int64  `json:"budget_rejections,omitempty"`
	HeadroomRejections int64  `json:"headroom_rejections,omitempty"`
}

type GPUPromptCache struct {
	maxBytes           int64
	usedBytes          int64
	headroomBytes      uint64
	compressed         bool
	lastEstimateBytes  int64
	lastFreeBytes      uint64
	uploadFailures     int64
	budgetRejections   int64
	headroomRejections int64
	ll                 *list.List
	items              map[kv.ChunkKey]*list.Element
}

func EstimateQwen35ForwardStateBytes(s Qwen35BaseForwardState) (int64, error) {
	var n int64
	add := func(length int) error {
		if length < 0 {
			return fmt.Errorf("negative length %d", length)
		}
		if int64(length) > (1<<62-n)/4 {
			return fmt.Errorf("Qwen forward-state byte size overflow")
		}
		n += int64(length) * 4
		return nil
	}
	for _, row := range s.FullK {
		if err := add(len(row)); err != nil {
			return 0, err
		}
	}
	for _, row := range s.FullV {
		if err := add(len(row)); err != nil {
			return 0, err
		}
	}
	for _, lin := range s.Linear {
		if err := add(len(lin.Conv)); err != nil {
			return 0, err
		}
		if err := add(len(lin.SSM)); err != nil {
			return 0, err
		}
	}
	return n, nil
}

func NewGPUPromptCache(maxBytes int64) *GPUPromptCache {
	return NewGPUPromptCacheWithHeadroom(maxBytes, 0)
}

func NewGPUPromptCacheWithHeadroom(maxBytes int64, headroomBytes uint64) *GPUPromptCache {
	return NewGPUPromptCacheWithOptions(maxBytes, headroomBytes, false)
}

func NewGPUPromptCacheWithOptions(maxBytes int64, headroomBytes uint64, compressed bool) *GPUPromptCache {
	if maxBytes < 0 {
		maxBytes = 0
	}
	return &GPUPromptCache{maxBytes: maxBytes, headroomBytes: headroomBytes, compressed: compressed, ll: list.New(), items: map[kv.ChunkKey]*list.Element{}}
}

func (c *GPUPromptCache) Put(key kv.ChunkKey, state Qwen35BaseForwardState) bool {
	if c == nil || c.maxBytes <= 0 {
		return false
	}
	estBytes, err := EstimateQwen35ForwardStateBytes(state)
	if c.compressed {
		estBytes = (estBytes + 1) / 2
	}
	c.lastEstimateBytes = estBytes
	if err != nil {
		c.uploadFailures++
		return false
	}
	if estBytes > c.maxBytes {
		c.budgetRejections++
		return false
	}
	free, _ := nvidia.MemInfo()
	c.lastFreeBytes = free
	if free > 0 && uint64(estBytes)+c.headroomBytes > free {
		c.headroomRejections++
		return false
	}
	var g *Qwen35GPUForwardState
	var gb *Qwen35GPUForwardStateBF16
	if c.compressed {
		gb, err = UploadQwen35ForwardStateGPUBF16(state)
	} else {
		g, err = UploadQwen35ForwardStateGPU(state)
	}
	if err != nil {
		c.uploadFailures++
		return false
	}
	actualBytes := estBytes
	if g != nil {
		actualBytes = g.Bytes
	}
	if gb != nil {
		actualBytes = gb.Bytes
	}
	if old := c.items[key]; old != nil {
		ent := old.Value.(*GPUPromptCacheEntry)
		if ent.State != nil {
			ent.State.Free()
		}
		if ent.StateBF16 != nil {
			ent.StateBF16.Free()
		}
		c.usedBytes -= ent.Bytes
		ent.State = g
		ent.StateBF16 = gb
		ent.Bytes = actualBytes
		c.usedBytes += actualBytes
		c.ll.MoveToFront(old)
		c.evict()
		return c.Contains(key)
	}
	ent := &GPUPromptCacheEntry{Key: key, State: g, StateBF16: gb, Bytes: actualBytes}
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
	return GPUPromptCacheStats{MaxBytes: c.maxBytes, UsedBytes: c.usedBytes, Entries: len(c.items), HeadroomBytes: c.headroomBytes, LastEstimateBytes: c.lastEstimateBytes, LastFreeBytes: c.lastFreeBytes, Compressed: c.compressed, UploadFailures: c.uploadFailures, BudgetRejections: c.budgetRejections, HeadroomRejections: c.headroomRejections}
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
		if ent.StateBF16 != nil {
			ent.StateBF16.Free()
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
		if ent.StateBF16 != nil {
			ent.StateBF16.Free()
		}
		c.usedBytes -= ent.Bytes
		if c.usedBytes < 0 {
			c.usedBytes = 0
		}
		c.ll.Remove(el)
	}
}
