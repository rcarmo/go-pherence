package diffusiongemma

import (
	"fmt"
	"os"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// ExpertLRUCache is an LRU GPU cache for FP8 expert weights.
// Experts are uploaded on first use and evicted LRU when the budget is exceeded.
type ExpertLRUCache struct {
	entries   map[expertKey]*expertEntry
	order     []expertKey // oldest first
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

	// Evict until we have room
	for c.usedBytes+entryBytes > c.maxBytes && len(c.order) > 0 {
		c.evictOldest()
	}

	gate, err := gpu.UploadFP8E4M3Linear(gW, gS, nil, gSh[0], gSh[1])
	if err != nil {
		return nil, nil, nil, fmt.Errorf("expert %d gate upload: %w", expertID, err)
	}
	up, err := gpu.UploadFP8E4M3Linear(uW, uS, nil, uSh[0], uSh[1])
	if err != nil {
		return nil, nil, nil, fmt.Errorf("expert %d up upload: %w", expertID, err)
	}
	down, err := gpu.UploadFP8E4M3Linear(dW, dS, nil, dSh[0], dSh[1])
	if err != nil {
		return nil, nil, nil, fmt.Errorf("expert %d down upload: %w", expertID, err)
	}

	e := &expertEntry{gate: gate, up: up, down: down, bytes: entryBytes}
	c.entries[key] = e
	c.order = append(c.order, key)
	c.usedBytes += entryBytes
	return gate, up, down, nil
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
	key := c.order[0]
	c.order = c.order[1:]
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
		c.evictions++
	}
}

func (c *ExpertLRUCache) Stats() string {
	return fmt.Sprintf("experts_cached=%d used=%.1fMB/%dMB hits=%d misses=%d evictions=%d",
		len(c.entries), float64(c.usedBytes)/1e6, c.maxBytes/1e6, c.hits, c.misses, c.evictions)
}

// runLRUCachedExperts runs MoE using LRU-cached GPU FP8 expert weights.
func runLRUCachedExperts(op LayerOp, weights *TextWeights, scratch ForwardScratch, fp8w *FP8TextWeights, cache *ExpertLRUCache) error {
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
	if err != nil {
		return err
	}
	hiddenSize := len(preNorm2)
	positions := len(scratch.Residual) / hiddenSize
	topK := len(scratch.TopKIDs) / positions

	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}

	intermediate := 704

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
			if eid >= 0 {
				neededExperts[eid] = true
			}
		}
	}
	type cachedExpert struct {
		gate, up, down *gpu.GPUFP8E4M3Linear
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
		expertMap[eid] = cachedExpert{gateL, upL, downL}
	}

	// Batched expert GEMM: all positions through each expert at once
	gateBatch := make([]float32, positions*intermediate)
	upBatch := make([]float32, positions*intermediate)
	actBatch := make([]float32, positions*intermediate)
	downBatch := make([]float32, positions*hiddenSize)

	for eid, el := range expertMap {
		// Collect positions that use this expert and their weights
		type posWeight struct {
			pos int
			w   float32
		}
		var users []posWeight
		for pos := 0; pos < positions; pos++ {
			for k := 0; k < topK; k++ {
				if scratch.TopKIDs[pos*topK+k] == eid {
					users = append(users, posWeight{pos, scratch.TopKVals[pos*topK+k]})
					break
				}
			}
		}
		if len(users) == 0 {
			continue
		}

		// Build batched input from normed rows of positions using this expert
		batchInput := make([]float32, len(users)*hiddenSize)
		for i, u := range users {
			copy(batchInput[i*hiddenSize:(i+1)*hiddenSize], normedRows[u.pos*hiddenSize:(u.pos+1)*hiddenSize])
		}
		batch := len(users)

		// Batched GEMM through gate, up, down
		gB := gateBatch[:batch*intermediate]
		uB := upBatch[:batch*intermediate]
		if err := gpu.GemmFP8E4M3(gB, batchInput, batch, el.gate); err != nil {
			// Fallback to per-position
			for i, u := range users {
				gpu.GemvFP8E4M3(gB[i*intermediate:(i+1)*intermediate], normedRows[u.pos*hiddenSize:(u.pos+1)*hiddenSize], el.gate)
			}
		}
		if err := gpu.GemmFP8E4M3(uB, batchInput, batch, el.up); err != nil {
			for i, u := range users {
				gpu.GemvFP8E4M3(uB[i*intermediate:(i+1)*intermediate], normedRows[u.pos*hiddenSize:(u.pos+1)*hiddenSize], el.up)
			}
		}

		// Activation per position
		aB := actBatch[:batch*intermediate]
		for i := 0; i < batch; i++ {
			simd.GELUTanhMulTo(aB[i*intermediate:(i+1)*intermediate], gB[i*intermediate:(i+1)*intermediate], uB[i*intermediate:(i+1)*intermediate])
		}

		// Batched down projection
		dB := downBatch[:batch*hiddenSize]
		if err := gpu.GemmFP8E4M3(dB, aB, batch, el.down); err != nil {
			for i := range users {
				gpu.GemvFP8E4M3(dB[i*hiddenSize:(i+1)*hiddenSize], aB[i*intermediate:(i+1)*intermediate], el.down)
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
		simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6)
	}
	return nil
}

// ReportCacheStats prints cache statistics to stderr.
func (c *ExpertLRUCache) ReportCacheStats() {
	if c != nil {
		fmt.Fprintf(os.Stderr, "DiffusionGemma expert LRU: %s\n", c.Stats())
	}
}
