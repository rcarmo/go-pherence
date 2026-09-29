package nemotrondiarization

import (
	"context"
	"fmt"
)

// StreamingWindow composes the released 31-layer audio tower, head and
// speaker cache for one prepared low-latency chunk. It does not implement
// the streaming audio frontend, chunk scheduling, or speaker segmentation.
type StreamingWindow struct {
	Tower       *OfflineAudioTower
	VulkanTower *VulkanStreamingTower // optional; caller owns Close
	Head        *OfflineHead
	Cache       *SpeakerCache
}

// ForwardPrepared accepts [frames+lookahead,512] projected stack embeddings.
// Its owned logits include cached context and lookahead; current-frame logits
// are at [cached*8:(cached+frames)*8] rows, matching the reference slice.
// Only fully valid prepared frames are supported here; the caller must supply
// a separate path for masked padding rows.
func (m *StreamingWindow) ForwardPrepared(chunk []float32, frames, lookahead int) (input, logits []float32, err error) {
	return m.ForwardPreparedContext(context.Background(), chunk, frames, lookahead)
}

// ForwardPreparedContext checks cancellation before cache mutation. A failed
// tower or head run leaves the cache unchanged; the request owns terminal
// error handling and must release its optional Vulkan tower.
func (m *StreamingWindow) ForwardPreparedContext(ctx context.Context, chunk []float32, frames, lookahead int) (input, logits []float32, err error) {
	if ctx == nil || m == nil || m.Tower == nil || m.Head == nil || m.Cache == nil {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization streaming window")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	input, err = m.Cache.Prepare(chunk, frames, lookahead)
	if err != nil {
		return nil, nil, err
	}
	rows := len(input) / projectedWidth
	if rows > maxPreparedDiarizationRows {
		return nil, nil, fmt.Errorf("Nemotron diarization streaming window exceeds qualified prepared row bound")
	}
	var hidden []float32
	if m.VulkanTower != nil {
		hidden, err = m.VulkanTower.Forward(ctx, input, rows)
	} else {
		hidden, err = m.Tower.ForwardOffline(input, rows)
	}
	if err != nil {
		return nil, nil, err
	}
	logits, err = m.Head.ForwardOffline(hidden, rows)
	if err != nil {
		return nil, nil, err
	}
	mask := make([]bool, rows)
	for i := range mask {
		mask[i] = true
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if err := m.Cache.Update(chunk, frames, lookahead, logits, mask); err != nil {
		return nil, nil, err
	}
	return input, logits, nil
}
