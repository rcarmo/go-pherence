package whisper

import (
	"bytes"
	"compress/zlib"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// PCMTranscribeOptions configures the opt-in checked path. Language is an
// explicit tokenizer language code (e.g. "pt") or "auto" for checked
// per-window language detection. This path transcribes rather than translates.
// A generation-limit failure retries only the failing portion using bounded
// padded subdivisions. Temperature fallback and cross-window text reconciliation are not
// implemented here. Word alignment is explicit.
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
	// OriginalDecoderCompatibility is experimental: decoder cross-attention adds
	// the original's zero-padded key extent (multiple of 256) and the decoder
	// MLP uses the original tanh-form GELU. Default false keeps existing output.
	OriginalDecoderCompatibility bool
	// OriginalWindowCompatibility is experimental and applies across the 30 s
	// windows of one TranscribePCMWindows call, as whisper.cpp's whisper_full
	// does: (1) every window's log-mel uses the whole-clip clamp floor (an extra
	// mel pass over all windows computes the clip maximum first); (2) each
	// window after the first is prompted with <|startofprev|> plus the last
	// <=223 previously decoded tokens (timestamps included), cleared when at
	// most 5 s of audio remain. Requires firstWindow 0 (no resume). Default
	// false keeps independent windows and existing output.
	OriginalWindowCompatibility bool
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
	// Prompt is the whisper.cpp rolling previous-text context after this window
	// (at most the 223 tokens a later window can read). It is set only for
	// OriginalWindowCompatibility and is not serialized with the transcript; a
	// caller may persist it to resume with TranscribePCMWindowsFromPrompt.
	Prompt *[]int `json:"-"`
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
	return w.transcribePCMWindowsFrom(ctx, source, totalSamples, tokenizer, opts, firstWindow, nil, emit)
}

// WindowResume continues an OriginalWindowCompatibility run after a persisted
// window: Start is that window's Window.EmitEnd (whisper.cpp's next seek) and
// Prompt is the rolling previous-text context emitted with it.
type WindowResume struct {
	Start  int64
	Prompt []int
}

// TranscribePCMWindowsFromPrompt resumes an OriginalWindowCompatibility run at
// window firstWindow from the state persisted with window firstWindow-1,
// instead of replaying earlier windows. The caller must have verified that
// state as part of the same persisted, identity-bound window record.
func (w *Whisper) TranscribePCMWindowsFromPrompt(ctx context.Context, source SampleReader, totalSamples int64, tokenizer *Tokenizer, opts PCMTranscribeOptions, firstWindow int64, resume WindowResume, emit func(WindowTranscript) error) error {
	if !opts.OriginalWindowCompatibility || firstWindow <= 0 || resume.Start <= 0 || resume.Start > totalSamples || len(resume.Prompt) > maxPreviousTextTokens {
		return fmt.Errorf("invalid PCM resume state")
	}
	for _, t := range resume.Prompt {
		if t < 0 || t >= w.Config.VocabSize {
			return fmt.Errorf("invalid PCM resume state")
		}
	}
	return w.transcribePCMWindowsFrom(ctx, source, totalSamples, tokenizer, opts, firstWindow, &resume, emit)
}

func (w *Whisper) transcribePCMWindowsFrom(ctx context.Context, source SampleReader, totalSamples int64, tokenizer *Tokenizer, opts PCMTranscribeOptions, firstWindow int64, resume *WindowResume, emit func(WindowTranscript) error) error {
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
	var clipMaxLog *float32
	var history *[]int
	startWindow, startSample := firstWindow, int64(0)
	if opts.OriginalWindowCompatibility {
		// whisper.cpp slides its window to the last timestamp of each decode
		// (seek += seek_delta) and conditions on every earlier window's tokens.
		// A resume either starts from the persisted seek and prompt of the
		// previous window or replays windows 0..firstWindow-1 deterministically
		// without emitting them; either way emitted windows equal an
		// uninterrupted run.
		startWindow = 0
		if v.previous == 0 {
			return fmt.Errorf("original window compatibility requires <|startofprev|>")
		}
		maxLog, err := pcmClipMelMax(ctx, source, plan, w.Config)
		if err != nil {
			return err
		}
		clipMaxLog, history = &maxLog, new([]int)
		if resume != nil {
			startWindow, startSample = firstWindow, resume.Start
			*history = append([]int(nil), resume.Prompt...)
		}
	}
	detectedLanguage := "auto"
	emitWindow := emit
	if auto {
		emitWindow = func(window WindowTranscript) error {
			window.Language = detectedLanguage
			return emit(window)
		}
	}
	if startWindow < firstWindow {
		emitAll := emitWindow
		emitWindow = func(window WindowTranscript) error {
			if window.Window.Index < firstWindow {
				return nil
			}
			return emitAll(window)
		}
	}
	if history != nil {
		emitPrompt := emitWindow
		emitWindow = func(window WindowTranscript) error {
			prompt := *history
			prompt = append([]int{}, prompt[max(0, len(prompt)-maxPreviousTextTokens):]...)
			window.Prompt = &prompt
			return emitPrompt(window)
		}
	}
	seekEnd := WhisperSeekEnd(totalSamples)
	var windowStart, seekDelta int64 // seek-mode state shared with infer
	infer := func(samples []float32, validSamples int) ([]Segment, []WordTiming, error) {
		seekDelta = whisperChunkFrames
		if history != nil {
			// whisper.cpp clears the rolling context for a short tail:
			// seek > 0 && seek+500 >= n_len_org, n_len_org = 1+(n+200-400)/160.
			if seek := windowStart / 160; seek > 0 && seek+500 >= seekEnd {
				*history = nil
			}
		}
		if opts.SkipDigitalSilence {
			zero, err := pcmDigitalSilence(ctx, samples)
			if err != nil {
				return nil, nil, err
			}
			if zero {
				return nil, nil, nil
			}
		}
		output, frames, cross, err := w.encodePCMWindow(ctx, samples, opts.VulkanEncoder, clipMaxLog)
		if err != nil {
			return nil, nil, err
		}
		windowOpts := opts
		windowV := v
		windowSuppress, windowBeginSuppress := suppress, beginSuppress
		if auto {
			detected, err := detectLanguageChecked(ctx, w.Config, w.Decoder, output, frames, baseOpts.Generation)
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
		segments, words, generated, err := decodeDroppingHistoryOnLimit(history, func() ([]Segment, []WordTiming, error) {
			return w.decodePCMWindow(ctx, tokenizer, output, frames, validSamples, windowV, windowOpts, windowSuppress, windowBeginSuppress, history, cross)
		})
		if err == nil {
			seekDelta = whisperSeekDelta(generated, windowV.timestampBegin)
		}
		if !errors.Is(err, ErrGenerationLimit) || validSamples < 2*int(MinWindowSamples) {
			return segments, words, err
		}
		// Split recovery decodes the whole window, so the next seek is a full chunk.
		return w.decodePCMWindowHalves(ctx, samples, validSamples, tokenizer, windowV, windowOpts, windowSuppress, windowBeginSuppress, clipMaxLog, history)
	}
	if history == nil {
		return transcribePCMPlanFrom(ctx, source, plan, startWindow, emitWindow, infer)
	}
	return transcribePCMSeekFrom(ctx, source, totalSamples, windowSamples, startWindow, startSample, &windowStart, &seekDelta, emitWindow, infer)
}

// whisperChunkFrames is one 30 s window in 10 ms mel frames (100*WHISPER_CHUNK_SIZE).
const whisperChunkFrames = 3000

// maxSeekWindows bounds a seek-driven run; each window advances at least 20 ms.
const maxSeekWindows = 10000

// WhisperSeekEnd is whisper.cpp's seek_end for a clip: n_len_org = 1+(n+200-400)/160
// mel frames.
func WhisperSeekEnd(totalSamples int64) int64 { return 1 + (totalSamples-200)/160 }

// WhisperSeekDone reports whisper.cpp's loop exit (seek + delta_min >= seek_end,
// delta_min 10 frames) for the next window starting at sample next.
func WhisperSeekDone(next, totalSamples int64) bool {
	return next >= totalSamples || next/160+10 >= WhisperSeekEnd(totalSamples)
}

// whisperSeekDelta is whisper.cpp's seek_delta in frames for one decode: twice
// the last timestamp index above <|0.00|>, or a full chunk when there is none or
// the output ends with a single timestamp after text (the last segment closed).
func whisperSeekDelta(generated []int, timestampBegin int) int64 {
	delta := int64(whisperChunkFrames)
	for _, t := range generated {
		if t > timestampBegin {
			delta = 2 * int64(t-timestampBegin)
		}
	}
	if n := len(generated); n > 1 && generated[n-2] < timestampBegin && generated[n-1] > timestampBegin {
		delta = whisperChunkFrames
	}
	if delta <= 0 {
		delta = whisperChunkFrames
	}
	return delta
}

// transcribePCMSeekFrom is whisper.cpp's sliding window for
// OriginalWindowCompatibility. Window index starts at sample start; after each
// decode the next window starts at the last timestamp (seek += seek_delta),
// so speech after a window's last complete segment is decoded again instead of
// being dropped. A window owns [Start, EmitEnd); EmitEnd is the next Start.
func transcribePCMSeekFrom(ctx context.Context, source SampleReader, totalSamples, length, index, start int64, current, seekDelta *int64, emit func(WindowTranscript) error, infer func([]float32, int) ([]Segment, []WordTiming, error)) error {
	if ctx == nil || source == nil || index < 0 || start < 0 || start > totalSamples || length < MinWindowSamples || length > MaxWindowSamples {
		return fmt.Errorf("invalid PCM seek window")
	}
	seekEnd := WhisperSeekEnd(totalSamples)
	scratch := make([]float32, length)
	for !WhisperSeekDone(start, totalSamples) {
		if err := ctx.Err(); err != nil {
			return err
		}
		if index >= maxSeekWindows {
			return fmt.Errorf("checked PCM seek exceeds %d windows", maxSeekWindows)
		}
		valid := minInt64(length, totalSamples-start)
		n, err := source.ReadSamplesAt(ctx, scratch[:valid], start)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if int64(n) != valid || (err != nil && err != io.EOF) {
			if err == nil || err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return fmt.Errorf("read PCM window %d: %w", index, err)
		}
		clear(scratch[valid:])
		*current = start
		segments, words, err := infer(scratch, int(valid))
		if err != nil {
			return fmt.Errorf("infer PCM window %d: %w", index, err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		delta := *seekDelta
		if delta == whisperChunkFrames {
			delta = minInt64(seekEnd-start/160, whisperChunkFrames) // single timestamp ending
		}
		if delta < 1 {
			delta = 1
		}
		end := start + valid
		emitEnd := minInt64(start+delta*160, end)
		window := Window{Index: index, Start: start, End: end, EmitStart: start, EmitEnd: emitEnd, InputSamples: length, PadSamples: length - valid}
		mapped, mappedWords, err := canonicalWindowOutput(window, segments, words)
		if err != nil {
			return err
		}
		if err := emit(WindowTranscript{Window: window, Segments: mapped, Words: mappedWords}); err != nil {
			return fmt.Errorf("emit PCM window %d: %w", index, err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		index, start = index+1, emitEnd
	}
	return nil
}

// minInt64 avoids this package's int-only min helpers.
func minInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// ValidSeekWindow checks one persisted seek-mode window against its
// predecessor's EmitEnd (0 for the first) and the clip geometry.
func ValidSeekWindow(w Window, index, start, totalSamples, length int64) bool {
	if index < 0 || index >= maxSeekWindows || w.Index != index || w.Start != start || start < 0 || start >= totalSamples || WhisperSeekDone(start, totalSamples) {
		return false
	}
	valid := minInt64(length, totalSamples-start)
	if w.End != start+valid || w.InputSamples != length || w.PadSamples != length-valid || w.EmitStart != start {
		return false
	}
	return w.EmitEnd > start && w.EmitEnd <= w.End && ((w.EmitEnd-start)%160 == 0 || w.EmitEnd == w.End)
}

// Process-local stage timers (checked requests serialise); benchmark only.
var pcmMelNs, pcmEncodeNs, pcmDecoderStateNs int64

// windowCrossKV carries original-compatible cross K/V computed by the encoder.
type windowCrossKV struct{ k, v [][]float32 }

func (w *Whisper) encodePCMWindow(ctx context.Context, samples []float32, encoder *VulkanEncoder, clipMaxLog *float32) ([]float32, int, *windowCrossKV, error) {
	melStart := nowNs()
	mel, frames, err := melFlatChecked(ctx, samples, w.Config, clipMaxLog)
	pcmMelNs += nowNs() - melStart
	defer func(start int64) { pcmEncodeNs += nowNs() - start }(nowNs())
	if err != nil {
		return nil, 0, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, nil, err
	}
	var output []float32
	var cross *windowCrossKV
	if encoder != nil && encoder.hasCrossKV() {
		var k, v [][]float32
		output, k, v, err = encoder.ForwardCross(ctx, mel)
		cross = &windowCrossKV{k: k, v: v}
	} else if encoder != nil {
		output, err = encoder.Forward(ctx, mel)
	} else {
		output, err = w.Encoder.ForwardContext(ctx, mel, frames)
	}
	if err != nil {
		return nil, 0, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, nil, err
	}
	if len(output) != ((frames+1)/2)*w.Config.EncoderDModel {
		return nil, 0, nil, fmt.Errorf("invalid encoder output shape")
	}
	for index, value := range output {
		if index%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, nil, err
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, 0, nil, fmt.Errorf("non-finite encoder output")
		}
	}
	return output, (frames + 1) / 2, cross, nil
}

// history, when non-nil, is the whisper.cpp rolling previous-text context: the
// prompt uses its last 223 tokens and, on success, it becomes those prompt
// tokens plus this window's generated tokens (prompt_past1 semantics).
func (w *Whisper) decodePCMWindow(ctx context.Context, tokenizer *Tokenizer, output []float32, frames, validSamples int, v timestampVocabulary, opts PCMTranscribeOptions, suppress, beginSuppress []int, history *[]int, cross *windowCrossKV) ([]Segment, []WordTiming, error) {
	stateStart := nowNs()
	var state *DecoderState
	var err error
	if cross != nil {
		state, err = newDecoderStateFromCrossContext(ctx, w.Config, cross.k, cross.v, frames)
	} else {
		state, err = NewDecoderStateContext(ctx, w.Config, output, frames, w.Decoder)
	}
	pcmDecoderStateNs += nowNs() - stateStart
	if err != nil {
		return nil, nil, err
	}
	if opts.OriginalDecoderCompatibility {
		state.applyOriginalDecoderCompatibility(frames)
	}
	var previous []int
	if history != nil {
		previous = *history
		if len(previous) > maxPreviousTextTokens {
			previous = previous[len(previous)-maxPreviousTextTokens:]
		}
	}
	segments, generated, err := decodeCheckedTimestampsPrompted(ctx, w.Config, tokenizer, v, opts, suppress, beginSuppress, previous, func(token int) ([]float32, error) {
		if state.Pos >= w.Config.MaxDecoderLength {
			return nil, ErrGenerationLimit
		}
		return w.Decoder.ForwardToken(token, state), nil
	}, func(tokens []int) error {
		if state.Pos+len(tokens) > w.Config.MaxDecoderLength {
			return ErrGenerationLimit
		}
		w.Decoder.AdvanceTokens(tokens, state)
		return nil
	})
	if err == nil && history != nil {
		*history = append(append([]int(nil), previous...), generated...)
	}
	if err != nil || !opts.WordTimestamps {
		return segments, nil, err
	}
	allTokens := make([]int, 0)
	for _, segment := range segments {
		allTokens = append(allTokens, segment.Tokens...)
	}
	if len(allTokens) == 0 {
		return segments, nil, nil
	}
	alignmentState, err := newAlignmentDecoderStateContext(ctx, w.Config, frames, state)
	if err != nil {
		return nil, nil, err
	}
	alignmentState.crossPadKeys, alignmentState.tanhGELU = state.crossPadKeys, state.tanhGELU
	audioFrames := (validSamples + 159) / 160
	if history != nil {
		// Seek windows own audio only up to the next seek; whisper.cpp bounds
		// DTW the same way (n_frames = min(3000, seek_delta, ...)), so words of
		// the last complete segment cannot spill into the next window.
		if seek := whisperSeekDelta(generated, v.timestampBegin); seek < whisperChunkFrames && int(seek) < audioFrames {
			audioFrames = int(seek)
		}
	}
	words, err := AlignWordsChecked(ctx, w.Decoder, alignmentState, tokenizer, opts.Generation, opts.Language, allTokens, audioFrames)
	if err != nil {
		return nil, nil, fmt.Errorf("align window: %w", err)
	}
	return segments, words, nil
}

// decodePCMWindowHalves recovers only a full-window generation limit. Failing
// halves may be subdivided a bounded number of times; successful halves are
// never decoded twice. The detected language is shared across the whole window.
func (w *Whisper) decodePCMWindowHalves(ctx context.Context, samples []float32, validSamples int, tokenizer *Tokenizer, v timestampVocabulary, opts PCMTranscribeOptions, suppress, beginSuppress []int, clipMaxLog *float32, history *[]int) ([]Segment, []WordTiming, error) {
	return splitPCMGenerationLimit(ctx, samples, validSamples, func(padded []float32, valid int) ([]Segment, []WordTiming, error) {
		output, frames, cross, err := w.encodePCMWindow(ctx, padded, opts.VulkanEncoder, clipMaxLog)
		if err != nil {
			return nil, nil, err
		}
		segments, words, _, err := decodeDroppingHistoryOnLimit(history, func() ([]Segment, []WordTiming, error) {
			return w.decodePCMWindow(ctx, tokenizer, output, frames, valid, v, opts, suppress, beginSuppress, history, cross)
		})
		return segments, words, err
	})
}

// decodeDroppingHistoryOnLimit mirrors whisper.cpp's fallback: past the
// temperature cutoff (0.5) it decodes without the previous-text prompt, which
// is what breaks repetition loops seeded by repetitive context. whisper.cpp
// fails a decode on a generation limit/repetition loop or when the last 32
// generated tokens have entropy < 2.4 (entropy_thold); OpenAI Whisper also
// fails text whose zlib compression ratio exceeds 2.4. Three consecutive
// identical segments are also treated as a loop. Here the greedy retry
// without the prompt runs once, only when a rolling prompt was present; its
// result is accepted as whisper.cpp accepts its best decoder.
func decodeDroppingHistoryOnLimit(history *[]int, decode func() ([]Segment, []WordTiming, error)) ([]Segment, []WordTiming, []int, error) {
	attempt := func() ([]Segment, []WordTiming, []int, error) {
		previous := 0
		if history != nil {
			previous = min(len(*history), maxPreviousTextTokens)
		}
		segments, words, err := decode()
		var generated []int
		if err == nil && history != nil && len(*history) >= previous {
			generated = (*history)[previous:]
		}
		return segments, words, generated, err
	}
	if history == nil || len(*history) == 0 {
		return attempt()
	}
	segments, words, generated, err := attempt()
	if err == nil && !repetitiveGeneration(generated) && !highCompressionRatio(segments) && !repeatedSegments(segments) {
		return segments, words, generated, nil
	}
	if err != nil && !errors.Is(err, ErrGenerationLimit) {
		return segments, words, generated, err
	}
	*history = nil
	return attempt()
}

// repeatedSegments reports three consecutive identical non-empty segment
// texts, a prompt-seeded loop too short to move the compression ratio.
func repeatedSegments(segments []Segment) bool {
	run := 1
	for i := 1; i < len(segments); i++ {
		text := strings.TrimSpace(segments[i].Text)
		if text != "" && text == strings.TrimSpace(segments[i-1].Text) {
			if run++; run >= 3 {
				return true
			}
		} else {
			run = 1
		}
	}
	return false
}

// highCompressionRatio is OpenAI Whisper's compression_ratio_threshold test
// (2.4) on the decoded text: repeated sentences compress far better than speech.
func highCompressionRatio(segments []Segment) bool {
	var text strings.Builder
	for _, s := range segments {
		text.WriteString(s.Text)
	}
	if text.Len() == 0 {
		return false
	}
	var packed bytes.Buffer
	z := zlib.NewWriter(&packed)
	z.Write([]byte(text.String()))
	z.Close()
	return float64(text.Len())/float64(packed.Len()) > 2.4
}

// repetitiveGeneration is whisper.cpp's sequence entropy test: more than 32
// tokens and entropy of the last 32 token ids below 2.4 nats.
func repetitiveGeneration(tokens []int) bool {
	const n, threshold = 32, 2.4
	if len(tokens) <= n {
		return false
	}
	counts := map[int]int{}
	for _, t := range tokens[len(tokens)-n:] {
		counts[t]++
	}
	entropy := 0.0
	for _, c := range counts {
		p := float64(c) / n
		entropy -= p * math.Log(p)
	}
	return entropy < threshold
}

const maxPCMSplitDepth = 4 // at most 16 leaves, each independently bounded by the decoder

func splitPCMGenerationLimit(ctx context.Context, samples []float32, validSamples int, infer func([]float32, int) ([]Segment, []WordTiming, error)) ([]Segment, []WordTiming, error) {
	if ctx == nil || infer == nil || validSamples < 2*int(MinWindowSamples) || validSamples > len(samples) {
		return nil, nil, ErrGenerationLimit
	}
	var decode func(start, end, depth int) ([]Segment, []WordTiming, error)
	decode = func(start, end, depth int) ([]Segment, []WordTiming, error) {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		if depth > 0 {
			padded := make([]float32, len(samples))
			copy(padded, samples[start:end])
			segments, words, err := infer(padded, end-start)
			if err == nil {
				window := Window{Start: int64(start), End: int64(end), InputSamples: int64(len(samples)), PadSamples: int64(len(samples) - (end - start))}
				return canonicalWindowOutput(window, segments, words)
			}
			if !errors.Is(err, ErrGenerationLimit) || depth == maxPCMSplitDepth || end-start < 2*int(MinWindowSamples) {
				return nil, nil, fmt.Errorf("split PCM offset %d depth %d: %w", start, depth, err)
			}
		}
		mid := start + (end-start)/2
		left, leftWords, err := decode(start, mid, depth+1)
		if err != nil {
			return nil, nil, err
		}
		right, rightWords, err := decode(mid, end, depth+1)
		if err != nil {
			return nil, nil, err
		}
		tokenOffset := 0
		for _, segment := range left {
			tokenOffset += len(segment.Tokens)
		}
		for i := range rightWords {
			rightWords[i].TokenStart += tokenOffset
			rightWords[i].TokenEnd += tokenOffset
		}
		return append(left, right...), append(leftWords, rightWords...), nil
	}
	return decode(0, validSamples, 0)
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
	// Map through integer samples with one division, so equal absolute sample
	// positions from adjacent windows give identical seconds (window starts are
	// arbitrary 10 ms seek positions, not only 30 s multiples).
	absolute := func(local float64) float64 {
		return float64(window.Start+int64(math.Round(local*float64(SpeechSampleRate)))) / float64(SpeechSampleRate)
	}
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
		segment.End = absolute(math.Min(segment.End, valid))
		segment.Start = absolute(segment.Start)
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
		word.Start = absolute(word.Start)
		word.End = absolute(word.End)
		mappedWords = append(mappedWords, word)
	}
	return out, mappedWords, nil
}

// pcmClipMelMax is the explicit OriginalWindowCompatibility pre-pass: the raw
// log10 mel maximum over every planned (zero-padded) window, matching the
// maximum whisper.cpp takes over its whole-clip spectrogram except for frames
// at internal window edges (reflect padding). Reads only; no inference.
func pcmClipMelMax(ctx context.Context, source SampleReader, plan WindowPlan, cfg Config) (float32, error) {
	first, err := plan.At(0)
	if err != nil {
		return 0, err
	}
	scratch := make([]float32, int(first.InputSamples))
	maxLog := float32(math.Inf(-1))
	for i := int64(0); i < plan.Count(); i++ {
		window, err := plan.ReadWindow(ctx, source, i, scratch)
		if err != nil {
			return 0, err
		}
		m, err := audio.WhisperLogMelMaxContext(ctx, scratch[:window.InputSamples], cfg.NumMelBins)
		if err != nil {
			return 0, err
		}
		if m > maxLog {
			maxLog = m
		}
	}
	if math.IsInf(float64(maxLog), -1) {
		return 0, fmt.Errorf("empty PCM clip for mel maximum")
	}
	return maxLog, ctx.Err()
}
