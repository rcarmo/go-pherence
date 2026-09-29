package nemotronasr

import (
	"fmt"
	"math"
)

// GreedyRNNTStream carries the predictor state across prompt-projected encoder
// chunks. It returns raw token IDs (including blanks) and absolute zero-based
// encoder frame indices. No tokenizer or PCM frontend is owned here.
type GreedyRNNTStream struct {
	Model                           *GreedyRNNT
	state                           DecoderState
	token                           int
	frames                          int64
	closed                          bool
	logitsScratch, activatedScratch []float32
}

// Append consumes one to 139 complete, fully valid [rows,640] states. Invalid
// input is rejected without advancing predictor state. A model error closes
// the stream since a partially decoded chunk cannot be safely retried.
func (s *GreedyRNNTStream) Append(encoder []float32, rows int) (tokens []int, frames []int64, err error) {
	if s == nil || s.closed || s.Model == nil || s.Model.Decoder == nil || s.Model.Projection == nil || rows < 1 || rows > maxQualifiedRNNTFrames || len(encoder) != rows*rnntHidden || s.frames > math.MaxInt64-int64(rows) {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR streaming greedy input")
	}
	for _, v := range encoder {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, nil, fmt.Errorf("non-finite Nemotron ASR streaming greedy input")
		}
	}
	if len(s.logitsScratch) == 0 {
		s.logitsScratch = make([]float32, rnntVocabulary)
	}
	if len(s.activatedScratch) == 0 {
		s.activatedScratch = make([]float32, rnntHidden)
	}
	if len(s.logitsScratch) != rnntVocabulary || len(s.activatedScratch) != rnntHidden {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR streaming greedy scratch")
	}
	token := s.token
	if !s.state.initialized {
		token = rnntBlank
	}
	for frame := 0; frame < rows; frame++ {
		symbols := 0
		for {
			decoded, e := s.Model.Decoder.stepBorrowed(token, &s.state)
			if e != nil {
				s.closed = true
				return nil, nil, e
			}
			if e = s.Model.Projection.jointTo(s.logitsScratch, s.activatedScratch, encoder[frame*rnntHidden:(frame+1)*rnntHidden], decoded, 1); e != nil {
				s.closed = true
				return nil, nil, e
			}
			selected := 0
			for i := 1; i < len(s.logitsScratch); i++ {
				if s.logitsScratch[i] > s.logitsScratch[selected] {
					selected = i
				}
			}
			tokens = append(tokens, selected)
			frames = append(frames, s.frames+int64(frame))
			token = selected
			if advanceGreedyFrame(token, &symbols) {
				break
			}
		}
	}
	s.token = token
	s.frames += int64(rows)
	return tokens, frames, nil
}

// Finish prevents further chunks; a completed stream has no buffered logits.
func (s *GreedyRNNTStream) Finish() error {
	if s == nil || s.closed || s.frames == 0 {
		return fmt.Errorf("invalid Nemotron ASR streaming greedy finish")
	}
	s.closed = true
	return nil
}
