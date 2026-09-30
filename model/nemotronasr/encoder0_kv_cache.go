package nemotronasr

import (
	"fmt"
	"math"
)

const asrKVWindow = 57

// Encoder0KVCache owns the first ASR attention layer's sliding key/value
// history. This state belongs to one stream and is not safe for concurrent
// updates. It does not implement attention scoring, positional encoding or
// cross-layer cache coordination.
type Encoder0KVCache struct {
	keys, values []float32 // Immutable head-major packed history or private visible view
	retained     int
	seen         int
	// Internal immutable visible-buffer view. Public snapshots remain packed.
	stride, skip int
}

// Update appends head-major [8,frames,128] K/V chunks, returning owned
// [8,retained+frames,128] visible states. It retains at most 56 frames after
// returning, matching the released DynamicSlidingWindowLayer update rule.
// Validation failures preserve the previous state.
func (c *Encoder0KVCache) Update(keys, values []float32, frames int) (visibleK, visibleV []float32, err error) {
	return c.update(keys, values, frames, false)
}

// updateVisibleView is private to attention scoring. Its visible buffers do not
// escape that operator; retaining an immutable view avoids a second tail copy.
func (c *Encoder0KVCache) updateVisibleView(keys, values []float32, frames int) ([]float32, []float32, error) {
	return c.update(keys, values, frames, true)
}

func (c *Encoder0KVCache) update(keys, values []float32, frames int, view bool) (visibleK, visibleV []float32, err error) {
	if c == nil {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 KV cache input")
	}
	stride := c.stride
	if stride == 0 {
		stride = c.retained
	}
	if frames < 1 || frames > asrKVWindow || len(keys) != encoderWidth*frames || len(values) != encoderWidth*frames || c.retained < 0 || c.retained >= asrKVWindow || c.seen < c.retained || c.seen > int(^uint(0)>>1)-frames || stride < c.retained || stride > 2*asrKVWindow || c.skip < 0 || c.skip+c.retained > stride || len(c.keys) != encoderWidth*stride || len(c.values) != encoderWidth*stride {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 KV cache input")
	}
	for _, tensor := range [][]float32{keys, values} {
		for _, value := range tensor {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, nil, fmt.Errorf("non-finite Nemotron ASR encoder-0 KV cache input")
			}
		}
	}
	visibleRows := c.retained + frames
	visibleK = make([]float32, encoderWidth*visibleRows)
	visibleV = make([]float32, encoderWidth*visibleRows)
	for head := 0; head < asrAttentionHeads; head++ {
		oldStart := (head*stride + c.skip) * asrAttentionHeadWidth
		newStart := head * frames * asrAttentionHeadWidth
		outStart := head * visibleRows * asrAttentionHeadWidth
		copy(visibleK[outStart:outStart+c.retained*asrAttentionHeadWidth], c.keys[oldStart:oldStart+c.retained*asrAttentionHeadWidth])
		copy(visibleV[outStart:outStart+c.retained*asrAttentionHeadWidth], c.values[oldStart:oldStart+c.retained*asrAttentionHeadWidth])
		copy(visibleK[outStart+c.retained*asrAttentionHeadWidth:outStart+visibleRows*asrAttentionHeadWidth], keys[newStart:newStart+frames*asrAttentionHeadWidth])
		copy(visibleV[outStart+c.retained*asrAttentionHeadWidth:outStart+visibleRows*asrAttentionHeadWidth], values[newStart:newStart+frames*asrAttentionHeadWidth])
	}
	keep := min(visibleRows, asrKVWindow-1)
	if view {
		c.keys, c.values, c.retained, c.seen = visibleK, visibleV, keep, c.seen+frames
		c.stride, c.skip = visibleRows, visibleRows-keep
		return visibleK, visibleV, nil
	}
	nextK, nextV := make([]float32, encoderWidth*keep), make([]float32, encoderWidth*keep)
	for head := 0; head < asrAttentionHeads; head++ {
		start := head*visibleRows*asrAttentionHeadWidth + (visibleRows-keep)*asrAttentionHeadWidth
		copy(nextK[head*keep*asrAttentionHeadWidth:(head+1)*keep*asrAttentionHeadWidth], visibleK[start:start+keep*asrAttentionHeadWidth])
		copy(nextV[head*keep*asrAttentionHeadWidth:(head+1)*keep*asrAttentionHeadWidth], visibleV[start:start+keep*asrAttentionHeadWidth])
	}
	c.keys, c.values, c.retained, c.seen = nextK, nextV, keep, c.seen+frames
	c.stride, c.skip = 0, 0
	return visibleK, visibleV, nil
}

// Snapshot returns owned head-major state and cumulative input-frame count.
func (c *Encoder0KVCache) Snapshot() (keys, values []float32, retained, seen int) {
	if c == nil {
		return nil, nil, 0, 0
	}
	keys, values = make([]float32, c.retained*encoderWidth), make([]float32, c.retained*encoderWidth)
	stride := c.stride
	if stride == 0 {
		stride = c.retained
	}
	for head := 0; head < asrAttentionHeads; head++ {
		src := (head*stride + c.skip) * asrAttentionHeadWidth
		dst := head * c.retained * asrAttentionHeadWidth
		copy(keys[dst:dst+c.retained*asrAttentionHeadWidth], c.keys[src:src+c.retained*asrAttentionHeadWidth])
		copy(values[dst:dst+c.retained*asrAttentionHeadWidth], c.values[src:src+c.retained*asrAttentionHeadWidth])
	}
	return keys, values, c.retained, c.seen
}
