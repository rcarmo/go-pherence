package nemotronasr

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// ASRMelChunk is an owned processor-style [Frames,128] feature chunk.
// Valid counts complete mel rows; the masked operator uses this count.
type ASRMelChunk struct {
	Features []float32
	Frames   int
	Valid    int
}

// ASRMelChunkStream turns arbitrary PCM calls into a default lookahead-3
// processor-style schedule: 25 valid rows + one masked row first, followed
// by 32-row chunks. It omits the temporary first-chunk mel row from its
// state; the next real frame is computed when its PCM context is complete.
// Its terminal chunk right-pads mel rows to 26 or 32. Streaming generate
// uses the first 25 rows then all 32 rows per later chunk, without passing
// a 2D attention mask, even on the padded terminal chunk. The separate
// masked-operator tests pass Valid explicitly. It retains at most 31 valid
// mel rows between calls, independent of recording length. One stream must
// not be used concurrently. Consumers should release returned chunks.
type ASRMelChunkStream struct {
	mel     audio.NemotronMelStream
	pending []float32
	first   bool
	closed  bool
}

// AppendPCM accepts the frontend's usual one-to-80,000-sample chunks. Invalid
// input leaves state unchanged. Returned chunks and input are not modified by
// later calls. The returned slice is bounded by the size of this call.
func (s *ASRMelChunkStream) AppendPCM(pcm []float32) ([]ASRMelChunk, error) {
	if s == nil || s.closed {
		return nil, fmt.Errorf("invalid Nemotron ASR mel chunk stream")
	}
	features, err := s.mel.AppendPCM(pcm)
	if err != nil {
		return nil, err
	}
	return s.appendValid(features), nil
}

// Finish flushes the frontend, discards its single masked row and emits one
// right-masked terminal chunk if valid mel rows remain. For recordings under
// 160 samples the frontend has no valid rows and Finish emits no chunks.
// It can be called once.
func (s *ASRMelChunkStream) Finish() ([]ASRMelChunk, error) {
	if s == nil || s.closed {
		return nil, fmt.Errorf("invalid Nemotron ASR mel chunk finish")
	}
	features, err := s.mel.Finish()
	if err != nil {
		return nil, err
	}
	chunks := s.appendValid(features[:len(features)-128])
	if len(s.pending) != 0 {
		frames := 32
		if !s.first {
			frames = 26
		}
		valid := len(s.pending) / 128
		chunk := ASRMelChunk{Features: make([]float32, frames*128), Frames: frames, Valid: valid}
		copy(chunk.Features, s.pending)
		chunks = append(chunks, chunk)
	}
	s.pending = nil
	s.closed = true
	return chunks, nil
}

func (s *ASRMelChunkStream) appendValid(features []float32) []ASRMelChunk {
	var chunks []ASRMelChunk
	for len(features) != 0 {
		need := 32
		if !s.first {
			need = 25
		}
		take := need*128 - len(s.pending)
		if take > len(features) {
			take = len(features)
		}
		s.pending = append(s.pending, features[:take]...)
		features = features[take:]
		if len(s.pending) != need*128 {
			continue
		}
		frames := need
		if !s.first {
			frames = 26
		}
		chunk := ASRMelChunk{Features: make([]float32, frames*128), Frames: frames, Valid: need}
		copy(chunk.Features, s.pending)
		chunks = append(chunks, chunk)
		s.pending = s.pending[:0]
		s.first = true
	}
	return chunks
}
