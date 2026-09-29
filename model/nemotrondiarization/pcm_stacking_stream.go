package nemotrondiarization

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// PCMStackingStream composes bounded Nemotron PCM features and diarization
// embedding groups. It returns projected embeddings, not speaker logits.
// One stream owns its own mel and pending stacking state.
type PCMStackingStream struct {
	mel   audio.NemotronMelStream
	stack StackingStream
}

func LoadPCMStackingStream(file *safetensors.File) (*PCMStackingStream, error) {
	projection, err := LoadStackingProjection(file)
	if err != nil {
		return nil, err
	}
	return &PCMStackingStream{stack: StackingStream{Projection: projection}}, nil
}

// AppendPCM requires at most five seconds per call. Invalid input leaves
// both stages unchanged, and returned embeddings are owned by the caller.
func (s *PCMStackingStream) AppendPCM(pcm []float32) ([]float32, error) {
	if s == nil || s.stack.Projection == nil || len(s.stack.Projection.weight) != projectedWidth*stackWidth || s.stack.closed {
		return nil, fmt.Errorf("invalid Nemotron PCM stacking stream")
	}
	if len(pcm) == 0 || len(pcm) > 16000*5 {
		return nil, fmt.Errorf("invalid Nemotron PCM stacking chunk length")
	}
	for _, value := range pcm {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron PCM stacking chunk")
		}
	}
	// A valid 5s PCM call can emit 501 rows; split before the stacker's
	// 500-row per-call admission limit, preserving group order.
	features, err := s.mel.AppendPCM(pcm)
	if err != nil {
		return nil, err
	}
	var out []float32
	for len(features) > 0 {
		n := min(len(features), 500*melBins)
		part, err := s.stack.AppendFeatures(features[:n])
		if err != nil {
			return nil, err
		}
		out = append(out, part...)
		features = features[n:]
	}
	return out, nil
}

// Finish flushes the masked terminal mel frame through stacking and emits
// the zero-padded final embedding, when one is required.
func (s *PCMStackingStream) Finish() ([]float32, error) {
	if s == nil || s.stack.Projection == nil || len(s.stack.Projection.weight) != projectedWidth*stackWidth || s.stack.closed {
		return nil, fmt.Errorf("invalid Nemotron PCM stacking finish")
	}
	features, err := s.mel.Finish()
	if err != nil {
		return nil, err
	}
	var out []float32
	if len(features) > 0 {
		out, err = s.stack.AppendFeatures(features)
		if err != nil {
			return nil, err
		}
	}
	last, err := s.stack.Finish()
	if err != nil {
		return nil, err
	}
	return append(out, last...), nil
}
