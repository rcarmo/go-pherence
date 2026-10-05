package whisper

import (
	"context"
	"fmt"
	"math"
	"strings"

	vad "github.com/rcarmo/go-pherence/model/silero"
)

// PCMVADOptions explicitly opts into the experimental native Silero path.
// It is not a serving default and does not replace checkpoint identities.
// The caller owns pinned immutable weights and must qualify model probabilities
// and timestamp behaviour before production use. No energy/CPU fallback occurs.
type PCMVADOptions struct {
	Model             *vad.Model
	Segmentation      vad.SegmentOptions
	AllowExperimental bool
	// PreserveWindowGaps groups nearby eligibility spans into original-audio
	// windows, retaining actual internal silence for recognition/alignment.
	// Groups are decoded separately so no word can straddle deleted gaps.
	// This explicit mode may cost more inference than speech compaction.
	PreserveWindowGaps bool
}

// VADWindowTranscript keeps compact-window coordinates separate from original
// timestamps. Segments and Words use the original sample timeline in seconds;
// OriginalSpeech lists retained original intervals, which can be discontinuous.
// A segment interval may enclose silence; this does not label that gap as
// speech. A word crossing removed audio is rejected, never interpolated.
// PreserveWindowGaps instead aligns over actual internal silence. CompactedWindow
// then counts concatenated retained audio including that silence across groups.
// No resume API is provided yet: deterministic verified VAD spans must join a
// caller's checkpoint identity before a durable resumed pipeline can use them.
type VADWindowTranscript struct {
	CompactedWindow Window
	OriginalSpeech  []vad.Span
	// OriginalAudio includes any real silence retained by PreserveWindowGaps.
	// Nil in the legacy compacted mode; OriginalSpeech then lists all input.
	OriginalAudio []vad.Span
	Segments      []Segment
	Words         []WordTiming
	Language      string
}

// TranscribePCMWindowsWithVAD performs genuine native VAD then runs the existing
// checked Go encoder/decoder against bounded compacted speech reads. Vulkan and
// word alignment remain explicit in opts. The VAD result is local and fresh per
// recording; any VAD failure occurs before transcription callbacks. An all-silence
// recording runs VAD but skips the Whisper encoder/decoder. Existing no-VAD/resume APIs retain
// their semantics. The caller must exclude concurrent use/reentry of Whisper.
func (w *Whisper) TranscribePCMWindowsWithVAD(ctx context.Context, source SampleReader, totalSamples int64, tokenizer *Tokenizer, opts PCMTranscribeOptions, vadOptions PCMVADOptions, emit func(VADWindowTranscript) error) error {
	if ctx == nil || source == nil || emit == nil || !vadOptions.AllowExperimental || vadOptions.Model == nil || totalSamples < MinWindowSamples || totalSamples > 4*3600*SpeechSampleRate {
		return fmt.Errorf("invalid experimental Whisper VAD request")
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
	language := opts.Language
	if language == "auto" {
		language = "en"
		if opts.Generation == nil {
			return fmt.Errorf("automatic VAD transcription requires generation metadata")
		}
	}
	vocabulary, err := checkedTimestampVocabulary(w.Config, tokenizer, language)
	if err != nil {
		return err
	}
	// Resolve immutable generation metadata before VAD so malformed policy
	// cannot consume source or turn an invalid request into silent success.
	validationOptions := opts
	validationOptions.Language = language
	if _, _, _, err := resolvePCMGeneration(w.Config, vocabulary, validationOptions, w.Decoder.SuppressTokens, w.Decoder.BeginSuppressTokens); err != nil {
		return err
	}
	if opts.MaxNewTokens < 0 || opts.MaxNewTokens > w.Config.MaxDecoderLength-3 || opts.OverlapSamples < 0 || opts.OverlapSamples > int64(w.Config.MaxLength)*80 || opts.MaxInitialTimestampIndex < 0 || opts.MaxInitialTimestampIndex > 1500 {
		return fmt.Errorf("invalid Whisper VAD decoding/window bounds")
	}
	if opts.WordTimestamps && (opts.Generation == nil || len(opts.Generation.alignmentHeads) == 0) {
		return fmt.Errorf("Whisper VAD word timing requires alignment metadata")
	}
	spans, err := vadOptions.Model.Detect(ctx, source, totalSamples, vadOptions.Segmentation)
	if err != nil {
		return err
	}
	if vadOptions.PreserveWindowGaps {
		windowSamples := int64(w.Config.MaxLength) * 160
		groups := groupVADAudioSpans(spans, windowSamples)
		// Apply the existing per-recording window budget across groups before
		// any encoder invocation or callback, not separately to each group.
		var windows int64
		for _, group := range groups {
			if err := ctx.Err(); err != nil {
				return err
			}
			p, err := NewWindowPlan(group.End-group.Start, windowSamples, opts.OverlapSamples)
			if err != nil {
				return err
			}
			windows += p.Count()
			if windows > 10000 {
				return fmt.Errorf("Whisper VAD recording exceeds window budget")
			}
		}
		index, audioOffset := int64(0), int64(0)
		for _, group := range groups {
			if err := ctx.Err(); err != nil {
				return err
			}
			reader, err := vad.NewCompactedReader(source, []vad.Span{group}, totalSamples)
			if err != nil {
				return err
			}
			if err := w.TranscribePCMWindows(ctx, reader, reader.Samples(), tokenizer, opts, func(window WindowTranscript) error {
				mapped, err := mapVADWindow(window, reader, []vad.Span{group})
				if err != nil {
					return err
				}
				mapped.CompactedWindow.Index = index
				// Keep window coordinates in one concatenated retained-audio
				// timeline, including real internal silence, across all groups.
				mapped.CompactedWindow.Start += audioOffset
				mapped.CompactedWindow.End += audioOffset
				mapped.CompactedWindow.EmitStart += audioOffset
				mapped.CompactedWindow.EmitEnd += audioOffset
				mapped.OriginalAudio = append([]vad.Span(nil), mapped.OriginalSpeech...)
				mapped.OriginalSpeech = intersectVADSpans(spans, mapped.OriginalAudio)
				if err := emit(mapped); err != nil {
					return err
				}
				index++
				return nil
			}); err != nil {
				return err
			}
			audioOffset += reader.Samples()
		}
		return ctx.Err()
	}
	reader, err := vad.NewCompactedReader(source, spans, totalSamples)
	if err != nil {
		return err
	}
	if reader.Samples() == 0 {
		return ctx.Err()
	}
	return w.TranscribePCMWindows(ctx, reader, reader.Samples(), tokenizer, opts, func(window WindowTranscript) error {
		mapped, err := mapVADWindow(window, reader, spans)
		if err != nil {
			return err
		}
		return emit(mapped)
	})
}

// Greedy chronological groups only bridge gaps when the full real-audio extent
// fits a decoder window. A longer speech span stays intact and uses normal
// checked window planning; a disjoint next group is never decoded with it.
func groupVADAudioSpans(spans []vad.Span, windowSamples int64) []vad.Span {
	groups := make([]vad.Span, 0, len(spans))
	for _, span := range spans {
		if len(groups) > 0 && span.End-groups[len(groups)-1].Start <= windowSamples {
			groups[len(groups)-1].End = span.End
		} else {
			groups = append(groups, span)
		}
	}
	return groups
}
func intersectVADSpans(speech, audio []vad.Span) []vad.Span {
	var out []vad.Span
	for _, a := range audio {
		// Speech is sorted and disjoint. Start at the first possible overlap
		// so long recordings do not repeatedly scan all preceding spans.
		loIndex, hiIndex := 0, len(speech)
		for loIndex < hiIndex {
			mid := (loIndex + hiIndex) / 2
			if speech[mid].End <= a.Start {
				loIndex = mid + 1
			} else {
				hiIndex = mid
			}
		}
		for _, s := range speech[loIndex:] {
			if s.Start >= a.End {
				break
			}
			lo, hi := a.Start, a.End
			if s.Start > lo {
				lo = s.Start
			}
			if s.End < hi {
				hi = s.End
			}
			if hi > lo {
				out = append(out, vad.Span{Start: lo, End: hi})
			}
		}
	}
	return out
}

func mapVADWindow(window WindowTranscript, reader *vad.CompactedReader, spans []vad.Span) (VADWindowTranscript, error) {
	fail := func() (VADWindowTranscript, error) {
		return VADWindowTranscript{}, fmt.Errorf("invalid Whisper compacted VAD output")
	}
	if reader == nil || window.Window.Start < 0 || window.Window.End <= window.Window.Start || window.Window.End > reader.Samples() {
		return fail()
	}
	result := VADWindowTranscript{CompactedWindow: window.Window, Language: window.Language, Segments: make([]Segment, 0, len(window.Segments))}
	// Preserve precisely which original ranges contributed to this window.
	var offset int64
	for _, span := range spans {
		next := offset + span.End - span.Start
		left, right := window.Window.Start, window.Window.End
		if left < offset {
			left = offset
		}
		if right > next {
			right = next
		}
		if right > left {
			result.OriginalSpeech = append(result.OriginalSpeech, vad.Span{Start: span.Start + left - offset, End: span.Start + right - offset})
		}
		offset = next
		if offset >= window.Window.End {
			break
		}
	}
	mapTimes := func(start, end float64, word bool) (float64, float64, error) {
		if math.IsNaN(start) || math.IsNaN(end) || math.IsInf(start, 0) || math.IsInf(end, 0) || start < 0 || end < start || end > float64(reader.Samples())/16000 {
			return 0, 0, fmt.Errorf("invalid compacted VAD timestamps")
		}
		a, b := int64(math.Round(start*16000)), int64(math.Round(end*16000))
		originalStart, err := reader.MapStart(a)
		if err != nil {
			return 0, 0, err
		}
		originalEnd := originalStart
		if b > a {
			originalEnd, err = reader.MapEnd(b)
			if err != nil {
				return 0, 0, err
			}
		}
		if originalEnd < originalStart {
			return 0, 0, fmt.Errorf("inverted VAD timestamp mapping")
		}
		if word && originalEnd-originalStart != b-a {
			return 0, 0, fmt.Errorf("word crosses a removed VAD gap; original timing unavailable")
		}
		return float64(originalStart) / 16000, float64(originalEnd) / 16000, nil
	}
	var previousEnd float64
	for _, segment := range window.Segments {
		if segment.Start < previousEnd || segment.End <= segment.Start || segment.Start < float64(window.Window.Start)/16000 || segment.End > float64(window.Window.End)/16000 {
			return fail()
		}
		previousEnd = segment.End
		a, b, err := mapTimes(segment.Start, segment.End, false)
		if err != nil {
			return VADWindowTranscript{}, err
		}
		segment.Start, segment.End = a, b
		segment.Tokens = append([]int(nil), segment.Tokens...)
		result.Segments = append(result.Segments, segment)
	}
	if window.Words != nil {
		result.Words = make([]WordTiming, 0, len(window.Words))
	}
	previousEnd = 0
	previousTokenEnd := 0
	for _, word := range window.Words {
		if strings.TrimSpace(word.Word) == "" || word.TokenStart != previousTokenEnd || word.TokenEnd <= word.TokenStart || word.Start < previousEnd || word.Start < float64(window.Window.Start)/16000 || word.End > float64(window.Window.End)/16000 {
			return fail()
		}
		previousEnd = word.End
		previousTokenEnd = word.TokenEnd
		a, b, err := mapTimes(word.Start, word.End, true)
		if err != nil {
			return VADWindowTranscript{}, err
		}
		word.Start, word.End = a, b
		result.Words = append(result.Words, word)
	}
	return result, nil
}
