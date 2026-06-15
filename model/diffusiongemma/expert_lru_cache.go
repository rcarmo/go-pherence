package diffusiongemma

import (
	"errors"
	"fmt"
	"os"
	"sort"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

var ErrExpertCacheFull = errors.New("DiffusionGemma FP8 expert GPU cache full")

// ExpertLRUCache is an LRU GPU cache for FP8 expert weights.
// Experts are uploaded on first use and evicted LRU when the budget is exceeded.
type ExpertLRUCache struct {
	entries   map[expertKey]*expertEntry
	pinned    map[expertKey]bool // prewarmed resident prefix entries; never evicted by lazy overflow uploads
	order     []expertKey        // oldest first
	maxBytes  int64
	usedBytes int64
	hits      int
	misses    int
	evictions int
}

type expertKey struct {
	layer    int
	expertID int
}

type expertEntry struct {
	gate  *gpu.GPUFP8E4M3Linear
	up    *gpu.GPUFP8E4M3Linear
	down  *gpu.GPUFP8E4M3Linear
	bytes int64
}

// NewExpertLRUCache creates an expert cache with the given VRAM budget.
func NewExpertLRUCache(maxBytes int64) *ExpertLRUCache {
	return &ExpertLRUCache{
		entries:  make(map[expertKey]*expertEntry),
		pinned:   make(map[expertKey]bool),
		maxBytes: maxBytes,
	}
}

// Get returns cached GPU expert linears, or nil if not cached.
func (c *ExpertLRUCache) Get(layer, expertID int) (*gpu.GPUFP8E4M3Linear, *gpu.GPUFP8E4M3Linear, *gpu.GPUFP8E4M3Linear) {
	key := expertKey{layer, expertID}
	e, ok := c.entries[key]
	if !ok {
		c.misses++
		return nil, nil, nil
	}
	c.hits++
	// Move to end (most recent)
	c.touch(key)
	return e.gate, e.up, e.down
}

// Put uploads and caches an expert, evicting LRU entries if needed.
func (c *ExpertLRUCache) Put(layer, expertID int, fp8w *FP8TextWeights) (*gpu.GPUFP8E4M3Linear, *gpu.GPUFP8E4M3Linear, *gpu.GPUFP8E4M3Linear, error) {
	if c == nil || fp8w == nil || fp8w.shards == nil {
		return nil, nil, nil, fmt.Errorf("FP8 expert cache put missing cache or weights")
	}
	if layer < 0 || layer >= len(fp8w.Layers) || expertID < 0 {
		return nil, nil, nil, fmt.Errorf("FP8 expert cache put invalid layer=%d/%d expert=%d", layer, len(fp8w.Layers), expertID)
	}
	key := expertKey{layer, expertID}
	if e, ok := c.entries[key]; ok {
		c.touch(key)
		return e.gate, e.up, e.down, nil
	}

	prefix := fmt.Sprintf("model.decoder.layers.%d.experts", layer)

	gW, gS, gSh, err := loadFP8Proj(fp8w.shards, fmt.Sprintf("%s.%d.gate_proj", prefix, expertID))
	if err != nil {
		return nil, nil, nil, err
	}
	uW, uS, uSh, err := loadFP8Proj(fp8w.shards, fmt.Sprintf("%s.%d.up_proj", prefix, expertID))
	if err != nil {
		return nil, nil, nil, err
	}
	dW, dS, dSh, err := loadFP8Proj(fp8w.shards, fmt.Sprintf("%s.%d.down_proj", prefix, expertID))
	if err != nil {
		return nil, nil, nil, err
	}

	entryBytes := int64(len(gW) + len(uW) + len(dW) + (gSh[0]+uSh[0]+dSh[0])*4)
	if c.maxBytes > 0 && entryBytes > c.maxBytes {
		return nil, nil, nil, fmt.Errorf("%w: expert %d layer %d needs %d bytes > budget %d", ErrExpertCacheFull, expertID, layer, entryBytes, c.maxBytes)
	}

	// Evict non-pinned LRU entries, reusing the first evicted entry's GPU buffers.
	// Pinned prewarmed prefix entries are intentionally sticky to avoid the
	// pathological first-overflow-layer thrash that would evict layer 0 before the
	// next denoising step.
	var reuseGate, reuseUp, reuseDown *gpu.GPUFP8E4M3Linear
	for c.maxBytes > 0 && c.usedBytes+entryBytes > c.maxBytes && len(c.order) > 0 {
		evictIdx := -1
		for i, key := range c.order {
			if !c.pinned[key] {
				evictIdx = i
				break
			}
		}
		if evictIdx < 0 {
			break
		}
		oldKey := c.order[evictIdx]
		c.order = append(c.order[:evictIdx], c.order[evictIdx+1:]...)
		if old, ok := c.entries[oldKey]; ok {
			c.usedBytes -= old.bytes
			if reuseGate == nil {
				reuseGate, reuseUp, reuseDown = old.gate, old.up, old.down
			} else {
				if old.gate != nil {
					old.gate.Free()
				}
				if old.up != nil {
					old.up.Free()
				}
				if old.down != nil {
					old.down.Free()
				}
			}
			delete(c.entries, oldKey)
			delete(c.pinned, oldKey)
			c.evictions++
		}
	}
	if c.maxBytes > 0 && c.usedBytes+entryBytes > c.maxBytes {
		return nil, nil, nil, fmt.Errorf("%w: layer %d expert %d needs %d bytes, used=%d budget=%d", ErrExpertCacheFull, layer, expertID, entryBytes, c.usedBytes, c.maxBytes)
	}

	cleanupReuse := func() {
		if reuseGate != nil {
			reuseGate.Free()
			reuseGate = nil
		}
		if reuseUp != nil {
			reuseUp.Free()
			reuseUp = nil
		}
		if reuseDown != nil {
			reuseDown.Free()
			reuseDown = nil
		}
	}
	if err := gpu.UploadFP8E4M3LinearReuse(&reuseGate, gW, gS, nil, gSh[0], gSh[1]); err != nil {
		cleanupReuse()
		return nil, nil, nil, fmt.Errorf("expert %d gate upload: %w", expertID, err)
	}
	if err := gpu.UploadFP8E4M3LinearReuse(&reuseUp, uW, uS, nil, uSh[0], uSh[1]); err != nil {
		cleanupReuse()
		return nil, nil, nil, fmt.Errorf("expert %d up upload: %w", expertID, err)
	}
	if err := gpu.UploadFP8E4M3LinearReuse(&reuseDown, dW, dS, nil, dSh[0], dSh[1]); err != nil {
		cleanupReuse()
		return nil, nil, nil, fmt.Errorf("expert %d down upload: %w", expertID, err)
	}

	e := &expertEntry{gate: reuseGate, up: reuseUp, down: reuseDown, bytes: entryBytes}
	c.entries[key] = e
	c.order = append(c.order, key)
	c.usedBytes += entryBytes
	return reuseGate, reuseUp, reuseDown, nil
}

// PrewarmLayerPrefix uploads complete expert layers and pins them so lazy
// overflow uploads cannot evict the hot prefix. It stops cleanly before the
// first layer that cannot fit fully in the configured budget.
func (c *ExpertLRUCache) PrewarmLayerPrefix(fp8w *FP8TextWeights, layers, numExperts int) (int, int, error) {
	if c == nil || fp8w == nil {
		return 0, 0, fmt.Errorf("FP8 expert prewarm missing cache or weights")
	}
	if layers < 0 {
		layers = 0
	}
	if numExperts <= 0 {
		return 0, 0, fmt.Errorf("FP8 expert prewarm invalid expert count %d", numExperts)
	}
	if layers > len(fp8w.Layers) {
		layers = len(fp8w.Layers)
	}
	completedLayers, completedExperts := 0, 0
	for layer := 0; layer < layers; layer++ {
		var layerKeys []expertKey
		for eid := 0; eid < numExperts; eid++ {
			key := expertKey{layer: layer, expertID: eid}
			_, _, _, err := c.Put(layer, eid, fp8w)
			if err != nil {
				for _, k := range layerKeys {
					c.removeEntry(k)
				}
				if errors.Is(err, ErrExpertCacheFull) {
					return completedLayers, completedExperts, nil
				}
				return completedLayers, completedExperts, err
			}
			c.pinned[key] = true
			layerKeys = append(layerKeys, key)
		}
		completedLayers++
		completedExperts += numExperts
	}
	return completedLayers, completedExperts, nil
}

func (c *ExpertLRUCache) HasPinnedEntries() bool {
	return c != nil && len(c.pinned) > 0
}

func (c *ExpertLRUCache) LayerFullyPinned(layer, numExperts int) bool {
	if c == nil || numExperts <= 0 {
		return false
	}
	for eid := 0; eid < numExperts; eid++ {
		key := expertKey{layer: layer, expertID: eid}
		if !c.pinned[key] || c.entries[key] == nil {
			return false
		}
	}
	return true
}

func (c *ExpertLRUCache) PinnedLayerPrefix(numExperts int) int {
	if c == nil || numExperts <= 0 {
		return 0
	}
	prefix := 0
	for c.LayerFullyPinned(prefix, numExperts) {
		prefix++
	}
	return prefix
}

func (c *ExpertLRUCache) removeEntry(key expertKey) {
	e, ok := c.entries[key]
	if !ok {
		delete(c.pinned, key)
		return
	}
	if e.gate != nil {
		e.gate.Free()
	}
	if e.up != nil {
		e.up.Free()
	}
	if e.down != nil {
		e.down.Free()
	}
	c.usedBytes -= e.bytes
	delete(c.entries, key)
	delete(c.pinned, key)
	for i, k := range c.order {
		if k == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			break
		}
	}
}

func (c *ExpertLRUCache) touch(key expertKey) {
	for i, k := range c.order {
		if k == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			c.order = append(c.order, key)
			return
		}
	}
}

func (c *ExpertLRUCache) evictOldest() {
	if len(c.order) == 0 {
		return
	}
	for i, key := range c.order {
		if c.pinned[key] {
			continue
		}
		c.order = append(c.order[:i], c.order[i+1:]...)
		if e, ok := c.entries[key]; ok {
			c.usedBytes -= e.bytes
			if e.gate != nil {
				e.gate.Free()
			}
			if e.up != nil {
				e.up.Free()
			}
			if e.down != nil {
				e.down.Free()
			}
			delete(c.entries, key)
			delete(c.pinned, key)
			c.evictions++
		}
		return
	}
}

func (c *ExpertLRUCache) Stats() string {
	return fmt.Sprintf("experts_cached=%d pinned=%d used=%.1fMB/%dMB hits=%d misses=%d evictions=%d",
		len(c.entries), len(c.pinned), float64(c.usedBytes)/1e6, c.maxBytes/1e6, c.hits, c.misses, c.evictions)
}

// runLRUCachedExperts runs MoE using LRU-cached GPU FP8 expert weights.
func runLRUCachedExperts(op LayerOp, weights *TextWeights, scratch ForwardScratch, fp8w *FP8TextWeights, cache *ExpertLRUCache) error {
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	nExperts := 0
	if lb.ExpertsGateUpProj != nil && len(lb.ExpertsGateUpProj.Shape) == 3 {
		nExperts = lb.ExpertsGateUpProj.Shape[0]
	} else if lb.ExpertsDownProj != nil && len(lb.ExpertsDownProj.Shape) == 3 {
		nExperts = lb.ExpertsDownProj.Shape[0]
	} else if scratch.FP8ExpertIndex != nil {
		nExperts = scratch.FP8ExpertIndex.NumExperts
	}
	if nExperts <= 0 {
		return fmt.Errorf("FP8 LRU experts: missing expert-count metadata for layer %d", op.Layer)
	}
	preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
	if err != nil {
		return err
	}
	hiddenSize := len(preNorm2)
	if hiddenSize <= 0 || len(scratch.Residual)%hiddenSize != 0 {
		return fmt.Errorf("FP8 LRU experts: invalid hidden/residual size hidden=%d residual=%d", hiddenSize, len(scratch.Residual))
	}
	positions := len(scratch.Residual) / hiddenSize
	if positions <= 0 {
		return fmt.Errorf("FP8 LRU experts: no positions")
	}
	topK := scratch.TopKExperts
	if topK <= 0 {
		topK = len(scratch.TopKIDs) / positions
	}
	if topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK {
		return fmt.Errorf("FP8 LRU experts: invalid top-k scratch positions=%d topK=%d ids=%d vals=%d", positions, topK, len(scratch.TopKIDs), len(scratch.TopKVals))
	}

	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}

	intermediate := 0

	// Pre-norm all positions
	normedRows := make([]float32, positions*hiddenSize)
	for pos := 0; pos < positions; pos++ {
		resRow := scratch.Residual[pos*hiddenSize : (pos+1)*hiddenSize]
		dst := normedRows[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(dst, resRow)
		if !simd.RMSNormTo(dst, preNorm2, 1e-6) {
			return fmt.Errorf("pre_norm_2 rejected")
		}
	}

	// Collect unique experts and ensure they're cached
	neededExperts := map[int]bool{}
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			eid := scratch.TopKIDs[pos*topK+k]
			if eid < 0 {
				continue
			}
			if eid >= nExperts {
				return fmt.Errorf("FP8 LRU experts: routed expert id=%d outside [0,%d) at pos=%d topk=%d", eid, nExperts, pos, k)
			}
			neededExperts[eid] = true
		}
	}
	type cachedExpert struct {
		gate, up, down *gpu.GPUFP8E4M3Linear
	}
	type posWeight struct {
		pos int
		w   float32
	}
	expertMap := make(map[int]cachedExpert, len(neededExperts))
	for eid := range neededExperts {
		gateL, upL, downL := cache.Get(op.Layer, eid)
		if gateL == nil {
			var err error
			gateL, upL, downL, err = cache.Put(op.Layer, eid, fp8w)
			if err != nil {
				return fmt.Errorf("expert %d cache put: %w", eid, err)
			}
		}
		if gateL == nil || upL == nil || downL == nil || gateL.OutDim <= 0 || upL.OutDim != gateL.OutDim || downL.InDim != gateL.OutDim || downL.OutDim != hiddenSize {
			return fmt.Errorf("expert %d cached shape mismatch gate=%v up=%v down=%v hidden=%d", eid, gateL, upL, downL, hiddenSize)
		}
		if intermediate == 0 {
			intermediate = gateL.OutDim
		} else if intermediate != gateL.OutDim {
			return fmt.Errorf("expert %d intermediate=%d want %d", eid, gateL.OutDim, intermediate)
		}
		expertMap[eid] = cachedExpert{gateL, upL, downL}
	}
	if intermediate <= 0 {
		return nil
	}
	expertUsers := make(map[int][]posWeight, len(expertMap))
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			eid := scratch.TopKIDs[pos*topK+k]
			if _, ok := expertMap[eid]; ok {
				expertUsers[eid] = append(expertUsers[eid], posWeight{pos, scratch.TopKVals[pos*topK+k]})
			}
		}
	}
	expertIDs := make([]int, 0, len(expertMap))
	for eid := range expertMap {
		expertIDs = append(expertIDs, eid)
	}
	sort.Slice(expertIDs, func(i, j int) bool {
		ui, uj := len(expertUsers[expertIDs[i]]), len(expertUsers[expertIDs[j]])
		if ui == uj {
			return expertIDs[i] < expertIDs[j]
		}
		return ui > uj
	})

	// Batched expert GEMM: all positions through each expert at once.
	midLen, okMid := checked.MulInt(positions, intermediate)
	hidLen, okHid := checked.MulInt(positions, hiddenSize)
	if !okMid || !okHid {
		return fmt.Errorf("FP8 LRU expert batch size overflow positions=%d intermediate=%d hidden=%d", positions, intermediate, hiddenSize)
	}
	gateBatch := make([]float32, midLen)
	upBatch := make([]float32, midLen)
	actBatch := make([]float32, midLen)
	downBatch := make([]float32, hidLen)

	for _, eid := range expertIDs {
		el := expertMap[eid]
		users := expertUsers[eid]
		if len(users) == 0 {
			continue
		}

		// Build batched input from normed rows of positions using this expert
		batchInputLen, okBatchInput := checked.MulInt(len(users), hiddenSize)
		if !okBatchInput {
			return fmt.Errorf("FP8 LRU expert %d input batch overflow users=%d hidden=%d", eid, len(users), hiddenSize)
		}
		batchInput := make([]float32, batchInputLen)
		for i, u := range users {
			copy(batchInput[i*hiddenSize:(i+1)*hiddenSize], normedRows[u.pos*hiddenSize:(u.pos+1)*hiddenSize])
		}
		batch := len(users)

		// Batched GEMM through gate and up. They share the same input, so use the
		// paired GPU primitive to avoid a second host→device upload and to keep the
		// FP8 scratch cache coherent.
		gB := gateBatch[:batch*intermediate]
		uB := upBatch[:batch*intermediate]
		gateInput := batchInput
		if diffusionGemmaFP8DynamicActivationEnabled() {
			gateInput = quantizeDynamicTokenBatch(nil, batchInput, batch, hiddenSize)
		}
		if err := gpu.Gemm2FP8E4M3SameInput(gB, uB, gateInput, batch, el.gate, el.up); err != nil {
			// Fallback to per-projection/per-position if the paired GPU primitive is
			// unavailable. Preserve the same numeric operation rather than skipping an
			// expert silently.
			for i, u := range users {
				rowIn := gateInput[i*hiddenSize : (i+1)*hiddenSize]
				if err := gpu.GemvFP8E4M3(gB[i*intermediate:(i+1)*intermediate], rowIn, el.gate); err != nil {
					return fmt.Errorf("expert %d gate fallback pos %d: %w", eid, u.pos, err)
				}
				if err := gpu.GemvFP8E4M3(uB[i*intermediate:(i+1)*intermediate], rowIn, el.up); err != nil {
					return fmt.Errorf("expert %d up fallback pos %d: %w", eid, u.pos, err)
				}
			}
		}

		// Activation per position
		aB := actBatch[:batch*intermediate]
		for i := 0; i < batch; i++ {
			if !diffusionGemmaGELUMulTo(aB[i*intermediate:(i+1)*intermediate], gB[i*intermediate:(i+1)*intermediate], uB[i*intermediate:(i+1)*intermediate]) {
				return fmt.Errorf("expert %d activation rejected batch row %d", eid, i)
			}
		}

		// Batched down projection
		dB := downBatch[:batch*hiddenSize]
		downInput := aB
		if diffusionGemmaFP8DynamicActivationEnabled() {
			downInput = quantizeDynamicTokenBatch(nil, aB, batch, intermediate)
		}
		if err := gpu.GemmFP8E4M3(dB, downInput, batch, el.down); err != nil {
			for i, u := range users {
				if ferr := gpu.GemvFP8E4M3(dB[i*hiddenSize:(i+1)*hiddenSize], downInput[i*intermediate:(i+1)*intermediate], el.down); ferr != nil {
					return fmt.Errorf("expert %d down fallback pos %d after batched error %v: %w", eid, u.pos, err, ferr)
				}
			}
		}

		// Accumulate weighted results
		for i, u := range users {
			dst := scratch.MoeOut[u.pos*hiddenSize : (u.pos+1)*hiddenSize]
			expertSlice := dB[i*hiddenSize : (i+1)*hiddenSize]
			for j := range dst {
				dst[j] += u.w * expertSlice[j]
			}
		}
	}

	postNorm2, err := loadFloatVector(weights, lb.PostFFNLayerNorm2)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.MoeOut); off += hiddenSize {
		if !simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6) {
			return fmt.Errorf("FP8 LRU expert post_norm_2 rejected")
		}
	}
	return nil
}

// ReportCacheStats prints cache statistics to stderr.
func (c *ExpertLRUCache) ReportCacheStats() {
	if c != nil {
		fmt.Fprintf(os.Stderr, "DiffusionGemma expert LRU: %s\n", c.Stats())
	}
}

// ResetCounters zeroes hit/miss/eviction counters without touching the cache.
func (c *ExpertLRUCache) ResetCounters() {
	if c != nil {
		c.hits = 0
		c.misses = 0
		c.evictions = 0
	}
}

// ClearAll evicts all cached experts, freeing GPU memory.
func (c *ExpertLRUCache) ClearAll() {
	if c == nil {
		return
	}
	for _, e := range c.entries {
		if e.gate != nil {
			e.gate.Free()
		}
		if e.up != nil {
			e.up.Free()
		}
		if e.down != nil {
			e.down.Free()
		}
	}
	c.entries = make(map[expertKey]*expertEntry)
	c.pinned = make(map[expertKey]bool)
	c.order = c.order[:0]
	c.usedBytes = 0
}
