package nemotronasr

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// OfflineProjection composes 32-frame subsampling, all 24 encoder blocks,
// prompt fusion and the 640-wide encoder projection. It does not select
// tokens or decode audio; this is a bounded prepared-feature operator.
type OfflineProjection struct {
	sub   *Subsampling
	tower *OfflineEncoderTower
	rnnt  *RNNTProjection
}

func LoadOfflineProjection(file *safetensors.File) (*OfflineProjection, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	sub, err := LoadSubsampling(file)
	if err != nil {
		return nil, err
	}
	tower, err := LoadOfflineEncoderTower(file)
	if err != nil {
		return nil, err
	}
	rnnt, err := LoadRNNTProjection(file)
	if err != nil {
		return nil, err
	}
	return &OfflineProjection{sub: sub, tower: tower, rnnt: rnnt}, nil
}

// ForwardFeatures accepts exactly 32 owned-or-caller 128-mel frames and
// returns independent owned [5,1024] tower and [5,640] prompt-projected
// encoder states. Only lookahead 0 and 3 and a prompt ID in [0,128) pass.
func (m *OfflineProjection) ForwardFeatures(features []float32, lookahead, prompt int) (tower, encoder []float32, err error) {
	if m == nil || m.sub == nil || m.tower == nil || m.rnnt == nil || len(features) != 32*128 {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR offline projection")
	}
	for _, value := range features {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, fmt.Errorf("non-finite Nemotron ASR features")
		}
	}
	projected, err := m.sub.ForwardOffline(features, 32, 32)
	if err != nil {
		return nil, nil, err
	}
	tower, err = m.tower.ForwardOfflineLookahead(projected, lookahead)
	if err != nil {
		return nil, nil, err
	}
	_, _, encoder, err = m.rnnt.Project(tower, 5, prompt)
	if err != nil {
		return nil, nil, err
	}
	return tower, encoder, nil
}
