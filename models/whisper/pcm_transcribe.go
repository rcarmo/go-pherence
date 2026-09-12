package whisper

import (
	"context"
	"fmt"
	"math"
)

// PCMTranscribeOptions configures the opt-in checked path. Language is an
// explicit tokenizer language code (e.g. "pt"); this path transcribes rather
// than translates. Automatic detection, temperature fallback, word alignment
// and cross-window text reconciliation are not implemented here.
type PCMTranscribeOptions struct {
	Language                 string
	OverlapSamples           int64
	MaxNewTokens             int                      // zero uses the model position limit minus the 3-token prompt
	MaxInitialTimestampIndex int                      // 20ms units; zero forces 0.00 unless Generation supplies it
	Generation               *CheckedGenerationConfig // optional immutable HF generation policy; no decoder mutation
	// SkipDigitalSilence emits an empty callback for windows containing only
	// exact PCM zeros (including signed zero and padding), before frontend/model
	// execution. Opt-in; no energy threshold, VAD or quiet-speech classification.
	SkipDigitalSilence bool
	// Optional caller-owned fixed-MaxLength resident encoder. Host w.Encoder
	// may be released/nil after resident construction. Pair with the
	// decoder from the same checkpoint; geometry admission cannot prove identity.
	// Caller excludes Close/other use throughout transcription and handles any
	// retained submission with VulkanDrain before reuse/Close. Nil keeps CPU path.
	VulkanEncoder *VulkanEncoder
}

// WindowTranscript contains raw per-window output in canonical PCM seconds.
// Segments are clipped to real audio, excluding the padded tail. Overlapping
// windows may repeat text. Window.EmitStart/EmitEnd describe ownership eligibility;
// no segment/word deduplication is claimed. Callers must reconcile overlaps before
// publishing final VTT. Tokens and segments belong to the callback recipient.
type WindowTranscript struct {
	Window   Window
	Segments []Segment
}

// Serialize the checked entry point: existing kernels have package-level timers,
// packing caches and feature switches. This does not synchronize legacy APIs;
// callers must own the model and exclude other legacy Whisper execution too.
var pcmInferenceGate = make(chan struct{}, 1)

// TranscribePCMWindows connects bounded PCM reads to the exact frontend and Go
// encoder/decoder. It does not load models, invoke FFmpeg or change legacy APIs.
// A media.PCMReader implements SampleReader; pass its actual Timeline().Samples.
// The source, model and tokenizer must remain immutable for the call. emit is
// synchronous, must not re-enter this API, and may durably checkpoint each window.
// On failure, earlier callbacks remain valid; the failed window is not emitted.
//
// Input scratch, features, encoder output and decoder KV are window-bounded.
// This API never accumulates the whole recording/transcript or caps it at 100
// windows. It rejects jobs over four hours, overlap above half a window, or
// more than 10000 windows (never silently truncating). Retention is caller-owned.
// Context is checked between frontend frames, encoder operators, cross-KV
// projections and decoder tokens. A running kernel/allocation/reorder is not
// interruptible; no wall-clock cancellation deadline is claimed. CPU work does
// not survive return. With opts.VulkanEncoder, timeout/cancellation may retain a
// native submission; the error preserves ErrVulkanInFlight and the caller must
// drain before reuse/Close. No hidden CPU fallback. This call does not close the
// resident encoder. Backend flags and legacy APIs retain their existing meanings.
func (w *Whisper) TranscribePCMWindows(ctx context.Context, source SampleReader, totalSamples int64, tokenizer *Tokenizer, opts PCMTranscribeOptions, emit func(WindowTranscript) error) error {
	if ctx == nil {
		return fmt.Errorf("checked PCM transcription requires context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if source == nil || emit == nil || totalSamples < 0 || totalSamples > 4*3600*SpeechSampleRate {
		return fmt.Errorf("checked PCM transcription requires source, callback and a 0..4h sample count")
	}
	select {
	case pcmInferenceGate <- struct{}{}:
		defer func() { <-pcmInferenceGate }()
	case <-ctx.Done():
		return ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := w.validatePCMModelForEncoder(opts.VulkanEncoder != nil); err != nil {
		return err
	}
	if opts.VulkanEncoder != nil {
		if err := opts.VulkanEncoder.checkPCMConfig(ctx, w.Config); err != nil {
			return err
		}
	}
	v, err := checkedTimestampVocabulary(w.Config, tokenizer, opts.Language)
	if err != nil {
		return err
	}
	opts, suppress, beginSuppress, err := resolvePCMGeneration(w.Config, v, opts, w.Decoder.SuppressTokens, w.Decoder.BeginSuppressTokens)
	if err != nil {
		return err
	}
	if opts.MaxNewTokens < 0 || opts.MaxNewTokens > w.Config.MaxDecoderLength-3 || opts.MaxInitialTimestampIndex < 0 || opts.MaxInitialTimestampIndex > 1500 {
		return fmt.Errorf("invalid checked generation bounds")
	}
	windowSamples := int64(w.Config.MaxLength) * 160
	if opts.OverlapSamples > windowSamples/2 {
		return fmt.Errorf("checked PCM overlap must not exceed half a window")
	}
	plan, err := NewWindowPlan(totalSamples, windowSamples, opts.OverlapSamples)
	if err != nil {
		return err
	}
	if plan.Count() > 10000 {
		return fmt.Errorf("checked PCM plan exceeds 10000 windows")
	}
	return transcribePCMPlan(ctx, source, plan, emit, func(samples []float32) ([]Segment, error) {
		if opts.SkipDigitalSilence {
			zero, err := pcmDigitalSilence(ctx, samples)
			if err != nil {
				return nil, err
			}
			if zero {
				return nil, nil
			}
		}
		mel, frames, err := MelFlatFromSamplesCheckedContext(ctx, samples, w.Config)
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var output []float32
		if opts.VulkanEncoder != nil {
			output, err = opts.VulkanEncoder.Forward(ctx, mel)
		} else {
			output, err = w.Encoder.ForwardContext(ctx, mel, frames)
		}
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(output) != ((frames+1)/2)*w.Config.EncoderDModel {
			return nil, fmt.Errorf("invalid encoder output shape")
		}
		for index, value := range output {
			if index%16384 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite encoder output")
			}
		}
		state, err := NewDecoderStateContext(ctx, w.Config, output, (frames+1)/2, w.Decoder)
		if err != nil {
			return nil, err
		}
		return decodeCheckedTimestamps(ctx, w.Config, tokenizer, v, opts, suppress, beginSuppress, func(token int) ([]float32, error) {
			if state.Pos >= w.Config.MaxDecoderLength {
				return nil, ErrGenerationLimit
			}
			return w.Decoder.ForwardToken(token, state), nil
		})
	})
}

// pcmDigitalSilence is deliberately narrower than a no-speech classifier. A
// nonzero sample, NaN or infinity cannot be silently discarded by this shortcut.
func pcmDigitalSilence(ctx context.Context, samples []float32) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	for i, sample := range samples {
		if i%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return false, err
			}
		}
		if sample != 0 {
			return false, nil
		}
	}
	return true, ctx.Err()
}

// Separate orchestration allows testing every sample/window and failure boundary
// with a fake infer function without representing those tests as neural quality.
func transcribePCMPlan(ctx context.Context, source SampleReader, plan WindowPlan, emit func(WindowTranscript) error, infer func([]float32) ([]Segment, error)) error {
	if plan.Count() == 0 {
		return ctx.Err()
	}
	first, err := plan.At(0)
	if err != nil {
		return err
	}
	scratch := make([]float32, int(first.InputSamples))
	for i := int64(0); i < plan.Count(); i++ {
		window, err := plan.ReadWindow(ctx, source, i, scratch)
		if err != nil {
			return err
		}
		segments, err := infer(scratch)
		if err != nil {
			return fmt.Errorf("infer PCM window %d: %w", i, err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		mapped, err := canonicalWindowSegments(window, segments)
		if err != nil {
			return err
		}
		if err := emit(WindowTranscript{Window: window, Segments: mapped}); err != nil {
			return fmt.Errorf("emit PCM window %d: %w", i, err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

func canonicalWindowSegments(window Window, segments []Segment) ([]Segment, error) {
	valid := float64(window.End-window.Start) / float64(SpeechSampleRate)
	duration := float64(window.InputSamples) / float64(SpeechSampleRate)
	offset := float64(window.Start) / float64(SpeechSampleRate)
	out := make([]Segment, 0, len(segments))
	lastEnd := 0.0
	for _, segment := range segments {
		if math.IsNaN(segment.Start) || math.IsNaN(segment.End) || math.IsInf(segment.Start, 0) || math.IsInf(segment.End, 0) || segment.Start < lastEnd || segment.End <= segment.Start || segment.End > duration {
			return nil, fmt.Errorf("invalid timestamps in PCM window %d", window.Index)
		}
		lastEnd = segment.End
		if segment.Start >= valid {
			continue
		}
		segment.End = math.Min(segment.End, valid) + offset
		segment.Start += offset
		out = append(out, segment)
	}
	return out, nil
}
