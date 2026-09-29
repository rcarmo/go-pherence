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
	keys, values []float32 // [8,retained,128] head-major
	retained     int
	seen         int
}

// Update appends head-major [8,frames,128] K/V chunks, returning owned
// [8,retained+frames,128] visible states. It retains at most 56 frames after
// returning, matching the released DynamicSlidingWindowLayer update rule.
// Validation failures preserve the previous state.
func (c *Encoder0KVCache) Update(keys, values []float32, frames int) (visibleK, visibleV []float32, err error) {
	if c == nil || frames < 1 || frames > asrKVWindow || len(keys) != encoderWidth*frames || len(values) != encoderWidth*frames || c.retained < 0 || c.retained >= asrKVWindow || c.seen < c.retained || c.seen > int(^uint(0)>>1)-frames || len(c.keys) != encoderWidth*c.retained || len(c.values) != encoderWidth*c.retained {
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
		oldStart := head * c.retained * asrAttentionHeadWidth
		newStart := head * frames * asrAttentionHeadWidth
		outStart := head * visibleRows * asrAttentionHeadWidth
		copy(visibleK[outStart:outStart+c.retained*asrAttentionHeadWidth], c.keys[oldStart:oldStart+c.retained*asrAttentionHeadWidth])
		copy(visibleV[outStart:outStart+c.retained*asrAttentionHeadWidth], c.values[oldStart:oldStart+c.retained*asrAttentionHeadWidth])
		copy(visibleK[outStart+c.retained*asrAttentionHeadWidth:outStart+visibleRows*asrAttentionHeadWidth], keys[newStart:newStart+frames*asrAttentionHeadWidth])
		copy(visibleV[outStart+c.retained*asrAttentionHeadWidth:outStart+visibleRows*asrAttentionHeadWidth], values[newStart:newStart+frames*asrAttentionHeadWidth])
	}
	keep := min(visibleRows, asrKVWindow-1)
	nextK, nextV := make([]float32, encoderWidth*keep), make([]float32, encoderWidth*keep)
	for head := 0; head < asrAttentionHeads; head++ {
		start := head*visibleRows*asrAttentionHeadWidth + (visibleRows-keep)*asrAttentionHeadWidth
		copy(nextK[head*keep*asrAttentionHeadWidth:(head+1)*keep*asrAttentionHeadWidth], visibleK[start:start+keep*asrAttentionHeadWidth])
		copy(nextV[head*keep*asrAttentionHeadWidth:(head+1)*keep*asrAttentionHeadWidth], visibleV[start:start+keep*asrAttentionHeadWidth])
	}
	c.keys, c.values, c.retained, c.seen = nextK, nextV, keep, c.seen+frames
	return visibleK, visibleV, nil
}

// Snapshot returns owned head-major state and cumulative input-frame count.
func (c *Encoder0KVCache) Snapshot() (keys, values []float32, retained, seen int) {
	if c == nil {
		return nil, nil, 0, 0
	}
	return append([]float32(nil), c.keys...), append([]float32(nil), c.values...), c.retained, c.seen
}
