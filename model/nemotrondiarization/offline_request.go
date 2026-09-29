package nemotrondiarization

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// OfflineRequest joins the CPU frontend, stacking, full-window audio tower,
// and speaker head. It returns raw frame logits, without thresholding or a
// speaker cache. This bounded path does not accept streaming chunks.
type OfflineRequest struct {
	stacking *StackingProjection
	tower    *OfflineAudioTower
	head     *OfflineHead
}

func LoadOfflineRequest(file *safetensors.File) (*OfflineRequest, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	stacking, err := LoadStackingProjection(file)
	if err != nil {
		return nil, err
	}
	tower, err := LoadOfflineAudioTower(file)
	if err != nil {
		return nil, err
	}
	head, err := LoadOfflineHead(file)
	if err != nil {
		return nil, err
	}
	return &OfflineRequest{stacking: stacking, tower: tower, head: head}, nil
}

// ForwardPCM accepts owned-or-caller mono F32 PCM at 16 kHz and returns
// owned [frames,8] raw logits. The frontend's last masked row is retained;
// the caller can apply the processor's attention mask when extracting spans.
func (m *OfflineRequest) ForwardPCM(pcm []float32) ([]float32, int, error) {
	if m == nil || m.stacking == nil || m.tower == nil || m.head == nil {
		return nil, 0, fmt.Errorf("invalid Nemotron diarization offline request")
	}
	features, frames, err := audio.NemotronLogMel(pcm)
	if err != nil {
		return nil, 0, err
	}
	stacked, err := m.stacking.Project(features, frames)
	if err != nil {
		return nil, 0, err
	}
	rows := (frames + diarizationUpsample - 1) / diarizationUpsample
	if rows > 376 {
		return nil, 0, fmt.Errorf("Nemotron diarization offline window exceeds 376 rows")
	}
	hidden, err := m.tower.ForwardOffline(stacked, rows)
	if err != nil {
		return nil, 0, err
	}
	logits, err := m.head.ForwardOffline(hidden, rows)
	if err != nil {
		return nil, 0, err
	}
	return logits[:frames*diarizationSpeakers], frames, nil
}
