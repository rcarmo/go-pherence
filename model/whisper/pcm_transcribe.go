package whisper

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
)

// PCMTranscribeOptions configures the opt-in checked path. Language is an
// explicit tokenizer language code (e.g. "pt") or "auto" for checked
// per-window language detection. This path transcribes rather than translates.
// Temperature fallback and cross-window text reconciliation are not implemented
// here. Word alignment is explicit.
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
	// WordTimestamps runs the checked CPU cross-attention alignment pass for each
	// decoded segment. It is explicit because it consumes an additional decoder
	// state and cannot observe the resident Vulkan encoder itself.
	WordTimestamps bool
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
	// Language is populated only for automatic language detection. Fixed-language
	// callers retain their existing serialized output bytes.
	Language string `json:",omitempty"`
	// Words is an independent checked alignment over all generated text tokens.
	// Segment timestamp boundaries are not used to clip or invent word timing.
	Words []WordTiming `json:",omitempty"`
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
	return w.TranscribePCMWindowsFrom(ctx, source, totalSamples, tokenizer, opts, 0, emit)
}

// TranscribePCMWindowsFrom resumes at a caller-verified window prefix. It keeps
// original absolute timestamps and ownership intervals and never reads/infers
// earlier windows. The caller must verify persisted outputs and the complete
// model/tokenizer/generation/options identity before selecting firstWindow.
// This is safe only for this independent-window path (no prior-text state).
// All ownership, gate, cancellation and Vulkan drain rules above still apply.
func (w *Whisper) TranscribePCMWindowsFrom(ctx context.Context, source SampleReader, totalSamples int64, tokenizer *Tokenizer, opts PCMTranscribeOptions, firstWindow int64, emit func(WindowTranscript) error) error {
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
	auto := opts.Language == "auto"
	validationLanguage := opts.Language
	if auto {
		validationLanguage = "en"
	}
	v, err := checkedTimestampVocabulary(w.Config, tokenizer, validationLanguage)
	if err != nil {
		return err
	}
	baseOpts := opts
	var suppress, beginSuppress []int
	if !auto {
		opts, suppress, beginSuppress, err = resolvePCMGeneration(w.Config, v, opts, w.Decoder.SuppressTokens, w.Decoder.BeginSuppressTokens)
		if err != nil {
			return err
		}
	} else if opts.Generation == nil {
		return fmt.Errorf("checked automatic language detection requires generation metadata")
	}
	if opts.MaxNewTokens < 0 || opts.MaxNewTokens > w.Config.MaxDecoderLength-3 || opts.MaxInitialTimestampIndex < 0 || opts.MaxInitialTimestampIndex > 1500 {
		return fmt.Errorf("invalid checked generation bounds")
	}
	if opts.WordTimestamps && (opts.Generation == nil || len(opts.Generation.alignmentHeads) == 0) {
		return fmt.Errorf("checked word timestamps require generation alignment heads")
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
	detectedLanguage := "auto"
	emitWindow := emit
	if auto {
		emitWindow = func(window WindowTranscript) error {
			window.Language = detectedLanguage
			return emit(window)
		}
	}
	return transcribePCMPlanFrom(ctx, source, plan, firstWindow, emitWindow, func(samples []float32, validSamples int) ([]Segment, []WordTiming, error) {
		if opts.SkipDigitalSilence {
			zero, err := pcmDigitalSilence(ctx, samples)
			if err != nil {
				return nil, nil, err
			}
			if zero {
				return nil, nil, nil
			}
		}
		mel, frames, err := MelFlatFromSamplesCheckedContext(ctx, samples, w.Config)
		if err != nil {
			return nil, nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		var output []float32
		if opts.VulkanEncoder != nil {
			output, err = opts.VulkanEncoder.Forward(ctx, mel)
		} else {
			output, err = w.Encoder.ForwardContext(ctx, mel, frames)
		}
		if err != nil {
			return nil, nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		if len(output) != ((frames+1)/2)*w.Config.EncoderDModel {
			return nil, nil, fmt.Errorf("invalid encoder output shape")
		}
		for index, value := range output {
			if index%16384 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, nil, err
				}
			}
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, nil, fmt.Errorf("non-finite encoder output")
			}
		}
		windowOpts := opts
		windowV := v
		windowSuppress, windowBeginSuppress := suppress, beginSuppress
		if auto {
			detected, err := detectLanguageChecked(ctx, w.Config, w.Decoder, output, (frames+1)/2, baseOpts.Generation)
			if err != nil {
				return nil, nil, err
			}
			detectedLanguage = detected
			windowOpts = baseOpts
			windowOpts.Language = detected
			windowV, err = checkedTimestampVocabulary(w.Config, tokenizer, detected)
			if err != nil {
				return nil, nil, err
			}
			windowOpts, windowSuppress, windowBeginSuppress, err = resolvePCMGeneration(w.Config, windowV, windowOpts, w.Decoder.SuppressTokens, w.Decoder.BeginSuppressTokens)
			if err != nil {
				return nil, nil, err
			}
		}
		state, err := NewDecoderStateContext(ctx, w.Config, output, (frames+1)/2, w.Decoder)
		if err != nil {
			return nil, nil, err
		}
		segments, err := decodeCheckedTimestamps(ctx, w.Config, tokenizer, windowV, windowOpts, windowSuppress, windowBeginSuppress, func(token int) ([]float32, error) {
			if state.Pos >= w.Config.MaxDecoderLength {
				return nil, ErrGenerationLimit
			}
			return w.Decoder.ForwardToken(token, state), nil
		})
		if err != nil || !windowOpts.WordTimestamps {
			return segments, nil, err
		}
		allTokens := make([]int, 0)
		for _, segment := range segments {
			allTokens = append(allTokens, segment.Tokens...)
		}
		if len(allTokens) > 0 {
			alignmentState, err := NewDecoderStateContext(ctx, w.Config, output, (frames+1)/2, w.Decoder)
			if err != nil {
				return nil, nil, err
			}
			audioFrames := (validSamples + 159) / 160
			words, err := AlignWordsChecked(ctx, w.Decoder, alignmentState, tokenizer, windowOpts.Generation, windowOpts.Language, allTokens, audioFrames)
			if err != nil {
				return nil, nil, fmt.Errorf("align window: %w", err)
			}
			return segments, words, nil
		}
		return segments, nil, nil
	})
}

func detectLanguageChecked(ctx context.Context, cfg Config, dec *Decoder, output []float32, frames int, generation *CheckedGenerationConfig) (string, error) {
	if ctx == nil || dec == nil || generation == nil || generation.cfg != cfg || frames < 1 || len(output) != frames*cfg.EncoderDModel {
		return "", fmt.Errorf("invalid checked language detection input")
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	state, err := NewDecoderStateContext(ctx, cfg, output, frames, dec)
	if err != nil {
		return "", err
	}
	logits := dec.ForwardToken(generation.vocabulary.sot, state)
	if err := ctx.Err(); err != nil {
		return "", err
	}
	languages := make([]string, 0, len(generation.languages))
	for language := range generation.languages {
		languages = append(languages, language)
	}
	sort.Strings(languages)
	best, bestScore := "", float32(math.Inf(-1))
	for _, language := range languages {
		id := generation.languages[language]
		if id < 0 || id >= len(logits) {
			return "", fmt.Errorf("invalid checked language token")
		}
		score := logits[id]
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 1) {
			return "", fmt.Errorf("non-finite language score")
		}
		if best == "" || score > bestScore {
			best, bestScore = language, score
		}
	}
	if best == "" {
		return "", fmt.Errorf("no checked language candidates")
	}
	return best, nil
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
func transcribePCMPlan(ctx context.Context, source SampleReader, plan WindowPlan, emit func(WindowTranscript) error, infer func([]float32, int) ([]Segment, []WordTiming, error)) error {
	return transcribePCMPlanFrom(ctx, source, plan, 0, emit, infer)
}

func transcribePCMPlanFrom(ctx context.Context, source SampleReader, plan WindowPlan, firstWindow int64, emit func(WindowTranscript) error, infer func([]float32, int) ([]Segment, []WordTiming, error)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if firstWindow < 0 || firstWindow > plan.Count() {
		return fmt.Errorf("invalid PCM resume window")
	}
	if firstWindow == plan.Count() {
		return ctx.Err()
	}
	first, err := plan.At(0)
	if err != nil {
		return err
	}
	scratch := make([]float32, int(first.InputSamples))
	for i := firstWindow; i < plan.Count(); i++ {
		window, err := plan.ReadWindow(ctx, source, i, scratch)
		if err != nil {
			return err
		}
		segments, words, err := infer(scratch, int(window.End-window.Start))
		if err != nil {
			return fmt.Errorf("infer PCM window %d: %w", i, err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		mapped, mappedWords, err := canonicalWindowOutput(window, segments, words)
		if err != nil {
			return err
		}
		if err := emit(WindowTranscript{Window: window, Segments: mapped, Words: mappedWords}); err != nil {
			return fmt.Errorf("emit PCM window %d: %w", i, err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

func canonicalWindowSegments(window Window, segments []Segment) ([]Segment, error) {
	mapped, _, err := canonicalWindowOutput(window, segments, nil)
	return mapped, err
}

func canonicalWindowOutput(window Window, segments []Segment, words []WordTiming) ([]Segment, []WordTiming, error) {
	valid := float64(window.End-window.Start) / float64(SpeechSampleRate)
	duration := float64(window.InputSamples) / float64(SpeechSampleRate)
	offset := float64(window.Start) / float64(SpeechSampleRate)
	out := make([]Segment, 0, len(segments))
	lastEnd := 0.0
	for _, segment := range segments {
		if math.IsNaN(segment.Start) || math.IsNaN(segment.End) || math.IsInf(segment.Start, 0) || math.IsInf(segment.End, 0) || segment.Start < lastEnd || segment.End <= segment.Start || segment.End > duration {
			return nil, nil, fmt.Errorf("invalid timestamps in PCM window %d", window.Index)
		}
		lastEnd = segment.End
		if segment.Start >= valid {
			continue
		}
		segment.End = math.Min(segment.End, valid) + offset
		segment.Start += offset
		out = append(out, segment)
	}
	var mappedWords []WordTiming
	if words != nil {
		mappedWords = make([]WordTiming, 0, len(words))
	}
	previousEnd, previousTokenEnd := 0.0, 0
	for _, word := range words {
		if math.IsNaN(word.Start) || math.IsNaN(word.End) || math.IsInf(word.Start, 0) || math.IsInf(word.End, 0) || word.Start < previousEnd || word.End < word.Start || word.End > valid || word.TokenStart != previousTokenEnd || word.TokenEnd <= word.TokenStart || strings.TrimSpace(word.Word) == "" {
			return nil, nil, fmt.Errorf("invalid word timestamps in PCM window %d", window.Index)
		}
		previousEnd, previousTokenEnd = word.End, word.TokenEnd
		word.Start += offset
		word.End += offset
		mappedWords = append(mappedWords, word)
	}
	return out, mappedWords, nil
}
