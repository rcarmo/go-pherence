package model

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/rcarmo/go-pherence/gpu"
)

type Qwen35GPUCacheStats struct {
	Enabled        bool  `json:"enabled"`
	RequestedBytes int64 `json:"requested_bytes"`
	BudgetBytes    int64 `json:"budget_bytes"`
	Clamped        bool  `json:"clamped"`
	UsedBytes      int64 `json:"used_bytes"`
	Entries        int   `json:"entries"`
	Hits           int64 `json:"hits"`
	Misses         int64 `json:"misses"`
	Evictions      int64 `json:"evictions"`
	Uploads        int64 `json:"uploads"`
	Transient      int64 `json:"transient_uploads"`
}

type qwen35GPUCacheState struct {
	sync.Mutex
	requestedBytes int64
	budgetBytes    int64
	clamped        bool
	usedBytes      int64
	entries        map[*Qwen35NVFP4Weight]bool
	tick           uint64
	hits           int64
	misses         int64
	evictions      int64
	uploads        int64
	transient      int64
}

var qwen35GPUCache = qwen35GPUCacheState{entries: map[*Qwen35NVFP4Weight]bool{}}

func ConfigureQwen35GPUCache(budgetBytes int64) {
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	qwen35GPUCache.requestedBytes = budgetBytes
	qwen35GPUCache.budgetBytes = qwen35SafeGPUCacheBudget(budgetBytes)
	qwen35GPUCache.clamped = qwen35GPUCache.budgetBytes != budgetBytes
	qwen35GPUCache.evictUntilLocked(0, nil)
}

func qwen35SafeGPUCacheBudget(requested int64) int64 {
	if requested <= 0 || !qwen35GPUEnabled || !gpu.SgemmReady() {
		return requested
	}
	free, _ := gpu.MemInfo()
	if free == 0 {
		return requested
	}
	const headroom = int64(1536 * 1024 * 1024)
	usable := int64(free)
	if usable > headroom {
		usable -= headroom
	} else {
		usable = usable / 2
	}
	if usable > 0 && requested > usable {
		return usable
	}
	return requested
}

func ResetQwen35GPUCache() {
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	for q := range qwen35GPUCache.entries {
		q.FreeGPU()
		delete(qwen35GPUCache.entries, q)
	}
	qwen35GPUCache.usedBytes = 0
	qwen35GPUCache.hits = 0
	qwen35GPUCache.misses = 0
	qwen35GPUCache.evictions = 0
	qwen35GPUCache.uploads = 0
	qwen35GPUCache.transient = 0
}

func Qwen35GPUCacheStatsSnapshot() Qwen35GPUCacheStats {
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	return Qwen35GPUCacheStats{
		Enabled:        qwen35GPUEnabled,
		RequestedBytes: qwen35GPUCache.requestedBytes,
		BudgetBytes:    qwen35GPUCache.budgetBytes,
		Clamped:        qwen35GPUCache.clamped,
		UsedBytes:      qwen35GPUCache.usedBytes,
		Entries:        len(qwen35GPUCache.entries),
		Hits:           qwen35GPUCache.hits,
		Misses:         qwen35GPUCache.misses,
		Evictions:      qwen35GPUCache.evictions,
		Uploads:        qwen35GPUCache.uploads,
		Transient:      qwen35GPUCache.transient,
	}
}

func qwen35CachedGPUWeight(q *Qwen35NVFP4Weight) (*gpu.GPUNVFP4Weight, bool, error) {
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	if q.GPU != nil {
		q.LastUse = atomic.AddUint64(&qwen35GPUCache.tick, 1)
		qwen35GPUCache.hits++
		return q.GPU, false, nil
	}
	qwen35GPUCache.misses++
	need := qwen35GPUWeightBytes(q)
	if qwen35GPUCache.budgetBytes > 0 && need > qwen35GPUCache.budgetBytes {
		return nil, false, fmt.Errorf("%s needs %.1f MB, larger than GPU cache budget %.1f MB", q.Name, float64(need)/1e6, float64(qwen35GPUCache.budgetBytes)/1e6)
	}
	if qwen35GPUCache.budgetBytes > 0 && qwen35GPUCache.usedBytes+need > qwen35GPUCache.budgetBytes {
		gw, err := gpu.UploadNVFP4Weight(q.W)
		if err != nil {
			return nil, false, err
		}
		qwen35GPUCache.transient++
		qwen35GPUCache.uploads++
		return gw, true, nil
	}
	gw, err := gpu.UploadNVFP4Weight(q.W)
	if err != nil {
		qwen35GPUCache.evictAllLocked()
		gw, err = gpu.UploadNVFP4Weight(q.W)
	}
	if err != nil {
		return nil, false, err
	}
	q.GPU = gw
	q.GPUBytes = need
	q.LastUse = atomic.AddUint64(&qwen35GPUCache.tick, 1)
	qwen35GPUCache.entries[q] = true
	qwen35GPUCache.usedBytes += need
	qwen35GPUCache.uploads++
	return gw, false, nil
}

func qwen35GPUWeightBytes(q *Qwen35NVFP4Weight) int64 {
	if q == nil || q.W == nil {
		return 0
	}
	weight := int64(len(q.W.Weight))
	scale := int64(len(q.W.WeightScale))
	// GPU buffers are float32-slot padded byte uploads.
	padded := func(n int64) int64 {
		if n <= 0 {
			return 0
		}
		return ((n + 3) / 4) * 4
	}
	return padded(weight) + padded(scale)
}

func (c *qwen35GPUCacheState) evictUntilLocked(need int64, keep *Qwen35NVFP4Weight) {
	if c.budgetBytes <= 0 {
		return
	}
	for c.usedBytes+need > c.budgetBytes && len(c.entries) > 0 {
		var victim *Qwen35NVFP4Weight
		for q := range c.entries {
			if q == keep {
				continue
			}
			if victim == nil || q.LastUse < victim.LastUse {
				victim = q
			}
		}
		if victim == nil {
			return
		}
		c.freeEntryLocked(victim)
	}
}

func (c *qwen35GPUCacheState) evictAllLocked() {
	for q := range c.entries {
		c.freeEntryLocked(q)
	}
}

func (c *qwen35GPUCacheState) freeEntryLocked(q *Qwen35NVFP4Weight) {
	if q == nil {
		return
	}
	bytes := q.GPUBytes
	q.FreeGPU()
	delete(c.entries, q)
	c.usedBytes -= bytes
	if c.usedBytes < 0 {
		c.usedBytes = 0
	}
	c.evictions++
}
