package nemotrondiarization

import "fmt"

// StreamingWindow composes the released 31-layer audio tower, head and
// speaker cache for one prepared low-latency chunk. It does not implement
// the streaming audio frontend, chunk scheduling, or speaker segmentation.
type StreamingWindow struct {
	Tower *OfflineAudioTower
	Head  *OfflineHead
	Cache *SpeakerCache
}

// ForwardPrepared accepts [frames+lookahead,512] projected stack embeddings.
// Its owned logits include cached context and lookahead; current-frame logits
// are at [cached*8:(cached+frames)*8] rows, matching the reference slice.
// Only fully valid prepared frames are supported here; the caller must supply
// a separate path for masked padding rows.
func (m *StreamingWindow) ForwardPrepared(chunk []float32, frames, lookahead int) (input, logits []float32, err error) {
	if m == nil || m.Tower == nil || m.Head == nil || m.Cache == nil {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization streaming window")
	}
	input, err = m.Cache.Prepare(chunk, frames, lookahead)
	if err != nil {
		return nil, nil, err
	}
	rows := len(input) / projectedWidth
	if rows > maxPreparedDiarizationRows {
		return nil, nil, fmt.Errorf("Nemotron diarization streaming window exceeds qualified prepared row bound")
	}
	hidden, err := m.Tower.ForwardOffline(input, rows)
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
	if err := m.Cache.Update(chunk, frames, lookahead, logits, mask); err != nil {
		return nil, nil, err
	}
	return input, logits, nil
}
