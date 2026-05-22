package qwen

import (
	"fmt"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

type Qwen35GPUPrewarmStats struct {
	Considered int   `json:"considered"`
	Uploaded   int   `json:"uploaded"`
	Skipped    int   `json:"skipped"`
	Bytes      int64 `json:"bytes"`
}

type Qwen35GPUCacheStats struct {
	Enabled        bool                     `json:"enabled"`
	RequestedBytes int64                    `json:"requested_bytes"`
	BudgetBytes    int64                    `json:"budget_bytes"`
	Clamped        bool                     `json:"clamped"`
	UsedBytes      int64                    `json:"used_bytes"`
	Entries        int                      `json:"entries"`
	Hits           int64                    `json:"hits"`
	Misses         int64                    `json:"misses"`
	Evictions      int64                    `json:"evictions"`
	Uploads        int64                    `json:"uploads"`
	UploadBytes    int64                    `json:"upload_bytes,omitempty"`
	Transient      int64                    `json:"transient_uploads"`
	TransientBytes int64                    `json:"transient_bytes,omitempty"`
	TopTransient   []Qwen35GPUTransientStat `json:"top_transient,omitempty"`
}

type Qwen35GPUTransientStat struct {
	Name  string `json:"name"`
	Count int64  `json:"count"`
	Bytes int64  `json:"bytes"`
}

type qwen35GPUMXEntry struct {
	GPU     *nvidia.GPUMLXWeight
	Bytes   int64
	LastUse uint64
}

type qwen35GPUCacheState struct {
	sync.Mutex
	requestedBytes    int64
	budgetBytes       int64
	clamped           bool
	usedBytes         int64
	entries           map[*Qwen35NVFP4Weight]bool
	mlxEntries        map[*mlx.QuantWeight]*qwen35GPUMXEntry
	transientGPU      *nvidia.GPUNVFP4Weight
	transientByName   map[string]Qwen35GPUTransientStat
	transientDetailed bool
	tick              uint64
	hits              int64
	misses            int64
	evictions         int64
	uploads           int64
	uploadBytes       int64
	transient         int64
	transientBytes    int64
}

var qwen35GPUCache = qwen35GPUCacheState{entries: map[*Qwen35NVFP4Weight]bool{}, mlxEntries: map[*mlx.QuantWeight]*qwen35GPUMXEntry{}, transientByName: map[string]Qwen35GPUTransientStat{}}
var qwen35GPUCacheHeadroomBytes int64 = 512 * 1024 * 1024

func SetQwen35GPUCacheHeadroom(bytes int64) {
	if bytes < 0 {
		bytes = 0
	}
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	qwen35GPUCacheHeadroomBytes = bytes
}

func SetQwen35GPUTransientDetail(enabled bool) {
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	qwen35GPUCache.transientDetailed = enabled
	if !enabled {
		qwen35GPUCache.transientByName = map[string]Qwen35GPUTransientStat{}
	}
}

func ConfigureQwen35GPUCache(budgetBytes int64) {
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	qwen35GPUCache.requestedBytes = budgetBytes
	qwen35GPUCache.budgetBytes = qwen35SafeGPUCacheBudget(budgetBytes)
	qwen35GPUCache.clamped = qwen35GPUCache.budgetBytes != budgetBytes
	qwen35GPUCache.evictUntilLocked(0, nil)
}

func qwen35SafeGPUCacheBudget(requested int64) int64 {
	if requested <= 0 || !qwen35GPUReady {
		return requested
	}
	free, _ := nvidia.MemInfo()
	if free == 0 {
		return requested
	}
	headroom := qwen35GPUCacheHeadroomBytes
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
	resetQwen35MLXGPUScratch()
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	for q := range qwen35GPUCache.entries {
		q.FreeGPU()
		delete(qwen35GPUCache.entries, q)
	}
	for q, e := range qwen35GPUCache.mlxEntries {
		if e != nil && e.GPU != nil {
			e.GPU.Free()
		}
		delete(qwen35GPUCache.mlxEntries, q)
	}
	if qwen35GPUCache.transientGPU != nil {
		qwen35GPUCache.transientGPU.Free()
		qwen35GPUCache.transientGPU = nil
	}
	qwen35GPUCache.usedBytes = 0
	qwen35GPUCache.hits = 0
	qwen35GPUCache.misses = 0
	qwen35GPUCache.evictions = 0
	qwen35GPUCache.uploads = 0
	qwen35GPUCache.uploadBytes = 0
	qwen35GPUCache.transient = 0
	qwen35GPUCache.transientBytes = 0
	qwen35GPUCache.transientByName = map[string]Qwen35GPUTransientStat{}
}

func PrewarmQwen35GPUCache(base *Qwen35BaseModel) Qwen35GPUPrewarmStats {
	stats := Qwen35GPUPrewarmStats{}
	if !qwen35GPUReady || base == nil {
		return stats
	}
	mlxWeights := qwen35BaseMLXWeights(base)
	// Placement policy: preserve a decode-hot layer prefix by default. Qwen3.6
	// repeats the same projection sequence every token; keeping an intact prefix
	// resident gives deterministic hits on subsequent decode steps. Size-only
	// sorting maximizes entry count but was slower because it scattered partial
	// layers and left large hot projections on CPU.
	for _, m := range mlxWeights {
		stats.Considered++
		if m == nil {
			continue
		}
		need := qwen35MLXGPUWeightBytes(m)
		qwen35GPUCache.Lock()
		fits := qwen35GPUCache.budgetBytes <= 0 || qwen35GPUCache.usedBytes+need <= qwen35GPUCache.budgetBytes
		qwen35GPUCache.Unlock()
		if !fits {
			stats.Skipped++
			continue
		}
		if _, err := qwen35CachedGPUMXWeight(m); err == nil {
			stats.Uploaded++
			stats.Bytes += need
		} else {
			stats.Skipped++
		}
	}
	for _, q := range qwen35BaseNVFP4Weights(base) {
		stats.Considered++
		if q == nil || q.GPU != nil || q.W == nil {
			continue
		}
		need := qwen35GPUWeightBytes(q)
		qwen35GPUCache.Lock()
		fits := qwen35GPUCache.budgetBytes <= 0 || qwen35GPUCache.usedBytes+need <= qwen35GPUCache.budgetBytes
		qwen35GPUCache.Unlock()
		if !fits {
			stats.Skipped++
			continue
		}
		if _, transient, err := qwen35CachedGPUWeight(q); err == nil && !transient {
			stats.Uploaded++
			stats.Bytes += need
		} else {
			stats.Skipped++
		}
	}
	return stats
}

func qwen35BaseMLXWeights(base *Qwen35BaseModel) []*mlx.QuantWeight {
	var out []*mlx.QuantWeight
	for i := range base.Layers {
		layer := &base.Layers[i]
		if layer.Full != nil {
			l := layer.Full
			out = append(out, l.QWm, l.KWm, l.VWm, l.OWm, l.GateWm, l.UpWm, l.DownWm)
		}
		if layer.Linear != nil {
			l := layer.Linear
			out = append(out, l.QKVWm, l.GateWm, l.BetaWm, l.AlphaWm, l.OutWm, l.MLPGateWm, l.MLPUpWm, l.MLPDownWm)
		}
	}
	return out
}

func qwen35BaseNVFP4Weights(base *Qwen35BaseModel) []*Qwen35NVFP4Weight {
	var out []*Qwen35NVFP4Weight
	for i := range base.Layers {
		layer := &base.Layers[i]
		if layer.Full != nil {
			l := layer.Full
			out = append(out, l.QWQ, l.KWQ, l.VWQ, l.OWQ, l.GateWQ, l.UpWQ, l.DownWQ)
		}
		if layer.Linear != nil {
			l := layer.Linear
			out = append(out, l.QKVWQ, l.GateWQ, l.BetaWQ, l.AlphaWQ, l.OutWQ, l.MLPGateWQ, l.MLPUpWQ, l.MLPDownWQ)
		}
	}
	return out
}

func Qwen35GPUCacheStatsSnapshot() Qwen35GPUCacheStats {
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	top := make([]Qwen35GPUTransientStat, 0, len(qwen35GPUCache.transientByName))
	if qwen35GPUCache.transientDetailed {
		for _, stat := range qwen35GPUCache.transientByName {
			top = append(top, stat)
		}
	}
	sort.Slice(top, func(i, j int) bool {
		if top[i].Bytes == top[j].Bytes {
			return top[i].Name < top[j].Name
		}
		return top[i].Bytes > top[j].Bytes
	})
	if len(top) > 10 {
		top = top[:10]
	}
	return Qwen35GPUCacheStats{
		Enabled:        qwen35GPUEnabled,
		RequestedBytes: qwen35GPUCache.requestedBytes,
		BudgetBytes:    qwen35GPUCache.budgetBytes,
		Clamped:        qwen35GPUCache.clamped,
		UsedBytes:      qwen35GPUCache.usedBytes,
		Entries:        len(qwen35GPUCache.entries) + len(qwen35GPUCache.mlxEntries),
		Hits:           atomic.LoadInt64(&qwen35GPUCache.hits),
		Misses:         qwen35GPUCache.misses,
		Evictions:      qwen35GPUCache.evictions,
		Uploads:        qwen35GPUCache.uploads,
		UploadBytes:    qwen35GPUCache.uploadBytes,
		Transient:      qwen35GPUCache.transient,
		TransientBytes: qwen35GPUCache.transientBytes,
		TopTransient:   top,
	}
}

func qwen35CachedGPUWeight(q *Qwen35NVFP4Weight) (*nvidia.GPUNVFP4Weight, bool, error) {
	if q.GPU != nil {
		q.LastUse = atomic.AddUint64(&qwen35GPUCache.tick, 1)
		atomic.AddInt64(&qwen35GPUCache.hits, 1)
		return q.GPU, false, nil
	}
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
		if err := nvidia.UploadNVFP4WeightReuse(&qwen35GPUCache.transientGPU, q.W); err != nil {
			return nil, false, err
		}
		qwen35RecordTransientLocked(q, need)
		qwen35GPUCache.uploads++
		qwen35GPUCache.uploadBytes += need
		return qwen35GPUCache.transientGPU, false, nil
	}
	gw, err := nvidia.UploadNVFP4Weight(q.W)
	if err != nil {
		if reuseErr := nvidia.UploadNVFP4WeightReuse(&qwen35GPUCache.transientGPU, q.W); reuseErr == nil {
			qwen35RecordTransientLocked(q, need)
			qwen35GPUCache.uploads++
			qwen35GPUCache.uploadBytes += need
			return qwen35GPUCache.transientGPU, false, nil
		}
		qwen35GPUCache.evictUntilLocked(need, q)
		gw, err = nvidia.UploadNVFP4Weight(q.W)
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
	qwen35GPUCache.uploadBytes += need
	return gw, false, nil
}

func qwen35RecordTransientLocked(q *Qwen35NVFP4Weight, need int64) {
	qwen35GPUCache.transient++
	qwen35GPUCache.transientBytes += need
	if !qwen35GPUCache.transientDetailed {
		return
	}
	if qwen35GPUCache.transientByName == nil {
		qwen35GPUCache.transientByName = map[string]Qwen35GPUTransientStat{}
	}
	name := ""
	if q != nil {
		name = q.Name
	}
	stat := qwen35GPUCache.transientByName[name]
	stat.Name = name
	stat.Count++
	stat.Bytes += need
	qwen35GPUCache.transientByName[name] = stat
}

func CacheQwen35MLXWeight(q *mlx.QuantWeight) (*nvidia.GPUMLXWeight, error) {
	return qwen35CachedGPUMXWeight(q)
}

func qwen35CachedGPUMXWeightIfResident(q *mlx.QuantWeight) (*nvidia.GPUMLXWeight, bool) {
	if q == nil {
		return nil, false
	}
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	e := qwen35GPUCache.mlxEntries[q]
	if e == nil || e.GPU == nil {
		return nil, false
	}
	e.LastUse = atomic.AddUint64(&qwen35GPUCache.tick, 1)
	qwen35GPUCache.hits++
	return e.GPU, true
}

func qwen35TransientGPUMXWeight(q *mlx.QuantWeight) (*nvidia.GPUMLXWeight, error) {
	if q == nil {
		return nil, fmt.Errorf("nil Qwen3.5 MLX weight")
	}
	gw, err := nvidia.UploadMLXWeightNative(q.Weight, q.Scales, q.Biases, q.InDim, q.OutDim, q.GroupSize)
	if err != nil {
		return nil, err
	}
	qwen35GPUCache.Lock()
	qwen35GPUCache.transient++
	qwen35GPUCache.transientBytes += qwen35MLXGPUWeightBytes(q)
	qwen35GPUCache.uploads++
	qwen35GPUCache.uploadBytes += qwen35MLXGPUWeightBytes(q)
	qwen35GPUCache.Unlock()
	return gw, nil
}

func qwen35CachedGPUMXWeight(q *mlx.QuantWeight) (*nvidia.GPUMLXWeight, error) {
	if q == nil {
		return nil, fmt.Errorf("nil Qwen3.5 MLX weight")
	}
	qwen35GPUCache.Lock()
	if e := qwen35GPUCache.mlxEntries[q]; e != nil && e.GPU != nil {
		e.LastUse = atomic.AddUint64(&qwen35GPUCache.tick, 1)
		qwen35GPUCache.hits++
		gw := e.GPU
		qwen35GPUCache.Unlock()
		return gw, nil
	}
	qwen35GPUCache.misses++
	need := qwen35MLXGPUWeightBytes(q)
	if qwen35GPUCache.budgetBytes > 0 && need > qwen35GPUCache.budgetBytes {
		qwen35GPUCache.Unlock()
		return nil, fmt.Errorf("MLX weight %dx%d needs %.1f MB, larger than GPU cache budget %.1f MB", q.OutDim, q.InDim, float64(need)/1e6, float64(qwen35GPUCache.budgetBytes)/1e6)
	}
	// For Qwen MLX the full model working set is larger than 12GB VRAM. Do not
	// evict resident MLX weights for one-off later-layer uploads: that thrashes
	// and produces zero hits on the next token. Instead, keep the resident prefix
	// stable and let overflow weights fall back to CPU until placement is smarter.
	if qwen35GPUCache.budgetBytes > 0 && qwen35GPUCache.usedBytes+need > qwen35GPUCache.budgetBytes {
		qwen35GPUCache.transient++
		qwen35GPUCache.transientBytes += need
		qwen35GPUCache.Unlock()
		return nil, fmt.Errorf("MLX weight cache full: need %.1f MB, used %.1f MB, budget %.1f MB", float64(need)/1e6, float64(qwen35GPUCache.usedBytes)/1e6, float64(qwen35GPUCache.budgetBytes)/1e6)
	}
	qwen35GPUCache.Unlock()

	gw, err := nvidia.UploadMLXWeightNative(q.Weight, q.Scales, q.Biases, q.InDim, q.OutDim, q.GroupSize)
	if err != nil {
		return nil, err
	}
	qwen35GPUCache.Lock()
	defer qwen35GPUCache.Unlock()
	if e := qwen35GPUCache.mlxEntries[q]; e != nil && e.GPU != nil {
		gw.Free()
		e.LastUse = atomic.AddUint64(&qwen35GPUCache.tick, 1)
		qwen35GPUCache.hits++
		return e.GPU, nil
	}
	qwen35GPUCache.mlxEntries[q] = &qwen35GPUMXEntry{GPU: gw, Bytes: need, LastUse: atomic.AddUint64(&qwen35GPUCache.tick, 1)}
	qwen35GPUCache.usedBytes += need
	qwen35GPUCache.uploads++
	qwen35GPUCache.uploadBytes += need
	return gw, nil
}

func qwen35MLXGPUWeightBytes(q *mlx.QuantWeight) int64 {
	if q == nil {
		return 0
	}
	// Qwen uses GemvMLXDirect, so upload/cache only the native MLX buffers:
	// packed uint32 weights plus F32 scales and biases.
	weight := int64(len(q.Weight)) * 4
	scales := int64(len(q.Scales)) * 4
	biases := int64(len(q.Biases)) * 4
	return weight + scales + biases
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
	for c.usedBytes+need > c.budgetBytes && (len(c.entries) > 0 || len(c.mlxEntries) > 0) {
		var victim *Qwen35NVFP4Weight
		var victimMLX *mlx.QuantWeight
		var victimUse uint64
		for q := range c.entries {
			if q == keep {
				continue
			}
			if victim == nil && victimMLX == nil || q.LastUse < victimUse {
				victim = q
				victimMLX = nil
				victimUse = q.LastUse
			}
		}
		for q, e := range c.mlxEntries {
			if e == nil {
				continue
			}
			if victim == nil && victimMLX == nil || e.LastUse < victimUse {
				victim = nil
				victimMLX = q
				victimUse = e.LastUse
			}
		}
		if victim != nil {
			c.freeEntryLocked(victim)
			continue
		}
		if victimMLX != nil {
			c.freeMLXEntryLocked(victimMLX)
			continue
		}
		return
	}
}

func (c *qwen35GPUCacheState) evictAllLocked() {
	for q := range c.entries {
		c.freeEntryLocked(q)
	}
	for q := range c.mlxEntries {
		c.freeMLXEntryLocked(q)
	}
}

func (c *qwen35GPUCacheState) freeMLXEntryLocked(q *mlx.QuantWeight) {
	if q == nil {
		return
	}
	e := c.mlxEntries[q]
	if e == nil {
		delete(c.mlxEntries, q)
		return
	}
	if e.GPU != nil {
		e.GPU.Free()
	}
	delete(c.mlxEntries, q)
	c.usedBytes -= e.Bytes
	if c.usedBytes < 0 {
		c.usedBytes = 0
	}
	c.evictions++
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
