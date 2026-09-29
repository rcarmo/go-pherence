package nemotronasr

import (
	"context"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// PCMGenerationModel shares released weights across single-stream requests.
// The lookahead-3 streaming generation path returns raw token decisions,
// including blank IDs, without tokeniser or word timestamps.
type PCMGenerationModel struct {
	Subsampling *Subsampling
	Tower       *OfflineEncoderTower
	Projection  *RNNTProjection
	Decoder     *RNNTDecoder
}

func LoadPCMGenerationModel(file *safetensors.File) (*PCMGenerationModel, error) {
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
	projection, err := LoadRNNTProjection(file)
	if err != nil {
		return nil, err
	}
	decoder, err := LoadRNNTDecoder(file)
	if err != nil {
		return nil, err
	}
	return &PCMGenerationModel{Subsampling: sub, Tower: tower, Projection: projection, Decoder: decoder}, nil
}

// PCMGenerationStream owns duration-independent frontend, convolution,
// 24-layer encoder and greedy predictor state. A stream is not concurrent.
// Invalid PCM calls leave it unchanged. Cancellation or model errors close
// it, since completed chunks cannot be replayed after a partial failure.
type PCMGenerationStream struct {
	Model    *PCMGenerationModel
	frontend ASRMelChunkStream
	sub      SubsamplingStream
	tower    CachedEncoderTower
	greedy   GreedyRNNTStream
	closed   bool
	onStage  func(string, []float32) // test-only observer; do not retain borrowed values
}

// AppendPCM consumes 1–80,000 mono 16-kHz finite samples and returns owned
// raw decisions and absolute zero-based encoder frame positions. It may
// return empty slices while accumulating a chunk; it never retains PCM.
func (s *PCMGenerationStream) AppendPCM(ctx context.Context, pcm []float32) ([]int, []int64, error) {
	if err := s.validate(ctx); err != nil {
		return nil, nil, err
	}
	// Check before frontend.AppendPCM to keep validation failures atomic.
	if len(pcm) < 1 || len(pcm) > 80000 {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR PCM generation chunk")
	}
	for _, v := range pcm {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, nil, fmt.Errorf("non-finite Nemotron ASR PCM generation chunk")
		}
	}
	chunks, err := s.frontend.AppendPCM(pcm)
	if err != nil {
		return nil, nil, err
	}
	return s.process(ctx, chunks)
}

// Finish flushes the terminal zero-padded mel chunk once. A recording with
// fewer than 160 samples has no valid mel row and produces no decisions.
func (s *PCMGenerationStream) Finish(ctx context.Context) ([]int, []int64, error) {
	if err := s.validate(ctx); err != nil {
		return nil, nil, err
	}
	chunks, err := s.frontend.Finish()
	if err != nil {
		return nil, nil, err
	}
	// Frontend Finish is irreversible even if the model or context fails.
	s.closed = true
	tokens, frames, err := s.process(ctx, chunks)
	if err != nil {
		return nil, nil, err
	}
	if s.greedy.frames != 0 {
		if err := s.greedy.Finish(); err != nil {
			return nil, nil, err
		}
	}
	return tokens, frames, nil
}

func (s *PCMGenerationStream) validate(ctx context.Context) error {
	if s == nil || s.closed || s.Model == nil || s.Model.Subsampling == nil || s.Model.Tower == nil || s.Model.Projection == nil || s.Model.Decoder == nil || ctx == nil {
		return fmt.Errorf("invalid Nemotron ASR PCM generation stream")
	}
	if err := ctx.Err(); err != nil {
		s.closed = true
		return err
	}
	return nil
}

func (s *PCMGenerationStream) process(ctx context.Context, chunks []ASRMelChunk) ([]int, []int64, error) {
	var tokens []int
	var frames []int64
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			s.closed = true
			return nil, nil, err
		}
		features, rows := chunk.Features, chunk.Frames
		if !s.sub.started {
			if chunk.Frames != 26 || chunk.Valid != 25 || len(features) != 26*128 {
				s.closed = true
				return nil, nil, fmt.Errorf("invalid Nemotron ASR first generation mel chunk")
			}
			features, rows = features[:25*128], 25
		} else if chunk.Frames != 32 || chunk.Valid < 1 || chunk.Valid > 32 || len(features) != 32*128 {
			s.closed = true
			return nil, nil, fmt.Errorf("invalid Nemotron ASR subsequent generation mel chunk")
		}
		if s.onStage != nil {
			s.onStage("mel", features)
		}
		s.sub.Model = s.Model.Subsampling
		projected, err := s.sub.ForwardUnmaskedChunk(features, rows)
		if err != nil {
			s.closed = true
			return nil, nil, err
		}
		if s.onStage != nil {
			s.onStage("subsampling", projected)
		}
		s.tower.Tower = s.Model.Tower
		hidden, err := s.tower.ForwardChunk(projected, 4, 3)
		if err != nil {
			s.closed = true
			return nil, nil, err
		}
		if s.onStage != nil {
			s.onStage("tower", hidden)
		}
		_, _, encoded, err := s.Model.Projection.Project(hidden, 4, 101)
		if err != nil {
			s.closed = true
			return nil, nil, err
		}
		s.greedy.Model = &GreedyRNNT{Decoder: s.Model.Decoder, Projection: s.Model.Projection}
		partTokens, partFrames, err := s.greedy.Append(encoded, 4)
		if err != nil {
			s.closed = true
			return nil, nil, err
		}
		tokens = append(tokens, partTokens...)
		frames = append(frames, partFrames...)
	}
	return tokens, frames, nil
}
