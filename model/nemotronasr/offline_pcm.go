package nemotronasr

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// ForwardPCM32 computes a bounded 32-frame ASR window from a mono F32 PCM
// prefix at 16 kHz. The extra 200 samples beyond frame 31's centre supply
// its full 400-sample STFT window; the frontend's final masked row is then
// discarded. The output is projected encoder state, not tokens or text;
// later windows need streaming frontend and encoder caches.
func (m *OfflineProjection) ForwardPCM32(pcm []float32, lookahead, prompt int) (tower, encoder []float32, err error) {
	if len(pcm) != 31*160+200 {
		return nil, nil, fmt.Errorf("Nemotron ASR 32-frame prefix requires 5160 PCM samples")
	}
	features, frames, err := audio.NemotronLogMel(pcm)
	if err != nil {
		return nil, nil, err
	}
	if frames != 33 {
		return nil, nil, fmt.Errorf("Nemotron ASR PCM frontend returned %d frames", frames)
	}
	return m.ForwardFeatures(features[:32*128], lookahead, prompt)
}
