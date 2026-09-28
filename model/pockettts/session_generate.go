package pockettts

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// GenerateInto resets the session, writes up to maxFrames*SamplesPerFrame into
// caller-owned PCM, and returns the written sample count. The warm path performs
// no internal heap allocation when the supplied noise source does not allocate.
func (s *Session) GenerateInto(pcm []float32, tokens []uint32, maxFrames, framesAfterEOS, decodeSteps int, eosThreshold float32, noise NoiseSource) (int, error) {
	return s.generateInto(nil, pcm, tokens, maxFrames, framesAfterEOS, decodeSteps, eosThreshold, noise)
}

// GenerateIntoContext checks cancellation at generation boundaries. On
// cancellation it returns the number of complete PCM frames already written;
// the session is reset before its next generation call. It cannot interrupt a
// single transformer, flow-head, or Mimi kernel already in progress.
func (s *Session) GenerateIntoContext(ctx context.Context, pcm []float32, tokens []uint32, maxFrames, framesAfterEOS, decodeSteps int, eosThreshold float32, noise NoiseSource) (int, error) {
	if ctx == nil {
		return 0, fmt.Errorf("nil Pocket TTS generation context")
	}
	return s.generateInto(ctx, pcm, tokens, maxFrames, framesAfterEOS, decodeSteps, eosThreshold, noise)
}

func (s *Session) generateInto(ctx context.Context, pcm []float32, tokens []uint32, maxFrames, framesAfterEOS, decodeSteps int, eosThreshold float32, noise NoiseSource) (int, error) {
	if s == nil || s.Generator == nil || maxFrames <= 0 || maxFrames > s.MaxFrames || len(pcm) < maxFrames*SamplesPerFrame || len(tokens) == 0 || framesAfterEOS < 0 || decodeSteps <= 0 || noise == nil || math.IsNaN(float64(eosThreshold)) || len(s.FlowState.Layers) == 0 || s.VoiceTemplate.Position+len(tokens)+maxFrames > s.FlowState.Layers[0].Capacity {
		return 0, fmt.Errorf("invalid Pocket TTS session generation")
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
	}
	if err := s.Reset(); err != nil {
		return 0, err
	}
	g := s.Generator
	if err := g.FlowLM.PromptText(s.FlowState, tokens); err != nil {
		return 0, err
	}
	copy(s.Normalized, g.FlowLM.BOS)
	written, eosAt := 0, -1
	for frame := 0; frame < maxFrames; frame++ {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return written, err
			}
		}
		if err := g.FlowLM.Input.Forward(s.FlowRow, s.Normalized); err != nil {
			return 0, err
		}
		if err := g.FlowLM.Transformer.StepInto(s.Hidden, s.FlowRow, s.FlowState); err != nil {
			return 0, err
		}
		if err := g.FlowLM.EOS.Forward(s.EOS, s.Hidden); err != nil {
			return 0, err
		}
		if eosAt < 0 && s.EOS[0] > eosThreshold {
			eosAt = frame
		}
		if eosAt >= 0 && frame >= eosAt+framesAfterEOS {
			break
		}
		if err := noise(frame, s.Noise); err != nil {
			return 0, fmt.Errorf("Pocket TTS noise frame %d: %w", frame, err)
		}
		copy(s.Normalized, s.Noise)
		for step := 0; step < decodeSteps; step++ {
			if ctx != nil {
				if err := ctx.Err(); err != nil {
					return written, err
				}
			}
			start, target := float32(step)/float32(decodeSteps), float32(step+1)/float32(decodeSteps)
			if err := g.Flow.ForwardInto(s.Velocity, s.Hidden, []float32{start, target}, s.Normalized, s.FlowScratch); err != nil {
				return 0, err
			}
			if !simd.VecScaleAddTo(s.Normalized, s.Normalized, s.Velocity, 1/float32(decodeSteps)) {
				return 0, fmt.Errorf("Pocket TTS LSD update failed")
			}
		}
		if !simd.VecMulTo(s.Raw, s.Normalized, g.Std) || !simd.VecAddTo(s.Raw, s.Raw, g.Mean) {
			return 0, fmt.Errorf("Pocket TTS latent denormalization failed")
		}
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return written, err
			}
		}
		chunk := pcm[written : written+SamplesPerFrame]
		if err := g.Mimi.DecodeFrameInto(chunk, s.Raw, s.MimiState); err != nil {
			return 0, err
		}
		written += SamplesPerFrame
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return written, err
		}
	}
	if written == 0 {
		return 0, fmt.Errorf("Pocket TTS generated no audio")
	}
	return written, nil
}
