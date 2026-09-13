//go:build linux && amd64

package speechjob

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"sort"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/models/whisper"
)

var (
	ErrTranscriptOverlap    = errors.New("conflicting ASR overlap requires reconciliation")
	ErrTranscriptSampleGrid = errors.New("ASR timestamp is not on the canonical sample grid")
)

// TranscriptStageConfig binds the declared language/geometry to one exact ASR
// stage version. Callers must use the same language and geometry as that stage;
// v1 raw window records do not themselves carry language/generation metadata.
// This does not validate language recognition or invent word alignment.
type TranscriptStageConfig struct {
	ASRVersion                    string
	Language                      string
	WindowSamples, OverlapSamples int64
}

func validateTranscriptStageConfig(cfg TranscriptStageConfig) error {
	if !validHash(cfg.ASRVersion) {
		return ErrConfiguration
	}
	if _, e := whisper.NewWindowPlan(0, cfg.WindowSamples, cfg.OverlapSamples); e != nil || cfg.OverlapSamples > cfg.WindowSamples/2 {
		return ErrConfiguration
	}
	return validateTranscript(context.Background(), Transcript{Schema: 2, SampleRate: 16000, Language: cfg.Language})
}

// NewTranscriptStage consumes the complete verified "asr-windows" checkpoint and
// creates an unlabelled "transcript". All raw segments are retained except exact
// duplicates (identical seconds, text and token IDs). Any other temporal overlap
// fails explicitly; ownership intervals never discard conflicting text. Missing,
// duplicate or reordered windows fail even if their segment lists are empty.
// Timestamps must be within 1e-6 sample of an integer sample (float64 arithmetic
// allowance only); otherwise conversion fails. No word-time interpolation occurs.
// This conservative policy is not a general cross-window text alignment model.
func NewTranscriptStage(cfg TranscriptStageConfig) (Stage, error) {
	if e := validateTranscriptStageConfig(cfg); e != nil {
		return Stage{}, e
	}
	identity, _ := json.Marshal(struct {
		Schema string
		Config TranscriptStageConfig
	}{"speechjob-transcript-exact-overlap-source-timing-v2", cfg})
	return Stage{Name: "transcript", Version: hash(identity), Run: func(ctx context.Context, in *Input, out io.Writer) (err error) {
		var asr Checkpoint
		var decoded Blob
		for _, cp := range in.job.Checkpoints {
			switch cp.Stage {
			case "asr-windows":
				asr = cp
			case "decode":
				decoded = cp.Blob
			}
		}
		if asr.Stage == "" || asr.Version != cfg.ASRVersion || decoded.File == "" {
			return ErrConfiguration
		}
		// All dependencies have been verified by Store.Run. As with model adapters,
		// path re-opening depends on the private immutable store/payload contract.
		pcm, e := media.OpenCanonicalPCM(ctx, filepath.Join(in.store.root.Name(), in.job.ID, decoded.File))
		if e != nil {
			return e
		}
		total := int64(pcm.Timeline().Samples)
		sourceTiming := pcm.SourceTiming()
		if e = pcm.Close(); e != nil {
			return e
		}
		r, e := in.OpenCheckpoint(ctx, "asr-windows")
		if e != nil {
			return e
		}
		defer func() { err = errors.Join(err, r.Close()) }()
		transcript, e := reconcileASR(ctx, r, total, asr.Key, cfg)
		if e != nil {
			return e
		}
		transcript.SourceTiming = sourceTiming
		if e = in.store.hit("transcript-output-ready"); e != nil {
			return e
		}
		return WriteTranscriptJSON(ctx, out, transcript)
	}}, nil
}

type rawCue struct {
	segment whisper.Segment
	words   []whisper.WordTiming
	window  int64
}

// readASRLine has explicit per-line and whole-stream limits. A record must end
// in LF; no permissive Scanner final-token or CRLF normalisation is used.
func readASRLine(ctx context.Context, r *bufio.Reader, total *int64) ([]byte, error) {
	var line []byte
	for {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		fragment, e := r.ReadSlice('\n')
		*total += int64(len(fragment))
		if *total > 64<<20 || len(line)+len(fragment) > 1<<20 {
			return nil, ErrLimit
		}
		line = append(line, fragment...)
		if e == nil {
			return line, nil
		}
		if errors.Is(e, bufio.ErrBufferFull) {
			continue
		}
		if errors.Is(e, io.EOF) && len(line) != 0 {
			return nil, fmt.Errorf("%w: unterminated ASR record", ErrCorrupt)
		}
		return nil, e
	}
}
func reconcileASR(ctx context.Context, reader io.Reader, total int64, key string, cfg TranscriptStageConfig) (Transcript, error) {
	var zero Transcript
	if e := ctx.Err(); e != nil {
		return zero, e
	}
	if reader == nil || !validHash(key) || total < 0 || total > 4*3600*16000 {
		return zero, ErrCorrupt
	}
	if e := validateTranscriptStageConfig(cfg); e != nil {
		return zero, e
	}
	plan, e := whisper.NewWindowPlan(total, cfg.WindowSamples, cfg.OverlapSamples)
	if e != nil {
		return zero, e
	}
	if plan.Count() > 10000 {
		return zero, ErrLimit
	}
	r := bufio.NewReaderSize(reader, 32<<10)
	var consumed int64
	var textBytes int
	raw := make([]rawCue, 0)
	for i := int64(0); i < plan.Count(); i++ {
		line, e := readASRLine(ctx, r, &consumed)
		if e != nil {
			return zero, fmt.Errorf("read ASR window %d: %w", i, e)
		}
		var record windowRecord
		if e = json.Unmarshal(line, &record); e != nil {
			return zero, ErrCorrupt
		}
		canonical, e := json.Marshal(record)
		if e != nil {
			return zero, ErrCorrupt
		}
		canonical = append(canonical, '\n')
		if !bytes.Equal(canonical, line) || record.Schema != 1 || record.Key != key || record.Result.Window.Index != i {
			return zero, ErrCorrupt
		}
		if e = validateWindow(record.Result, plan, 448, 51866); e != nil {
			return zero, e
		}
		words := record.Result.Words
		windowTokenOffset := 0
		for _, s := range record.Result.Segments {
			if len(raw) >= MaxTranscriptCues {
				return zero, ErrLimit
			}
			textBytes += len(s.Text)
			if textBytes > MaxTranscriptBytes {
				return zero, ErrLimit
			}
			segmentWords := make([]whisper.WordTiming, 0)
			segmentTokens := len(s.Tokens)
			for len(words) > 0 && words[0].TokenStart < windowTokenOffset+segmentTokens {
				if words[0].TokenStart < windowTokenOffset {
					return zero, ErrCorrupt
				}
				segmentWords = append(segmentWords, words[0])
				words = words[1:]
			}
			windowTokenOffset += segmentTokens
			raw = append(raw, rawCue{segment: s, words: segmentWords, window: i})
		}
		if len(words) != 0 {
			return zero, ErrCorrupt
		}
	}
	if _, e = readASRLine(ctx, r, &consumed); e != io.EOF {
		if e != nil {
			return zero, e
		}
		return zero, fmt.Errorf("%w: extra ASR record", ErrCorrupt)
	}
	if e = ctx.Err(); e != nil {
		return zero, e
	}
	sort.Slice(raw, func(i, j int) bool {
		a, b := raw[i], raw[j]
		if a.segment.Start != b.segment.Start {
			return a.segment.Start < b.segment.Start
		}
		if a.segment.End != b.segment.End {
			return a.segment.End < b.segment.End
		}
		return a.window < b.window
	})
	if e = ctx.Err(); e != nil {
		return zero, e
	}
	result := Transcript{Schema: 2, SampleRate: 16000, TotalSamples: total, Language: cfg.Language, Cues: []Cue{}}
	accepted := make([]rawCue, 0, len(raw))
	var previous whisper.Segment
	for i, c := range raw {
		if i%256 == 0 {
			if e = ctx.Err(); e != nil {
				return zero, e
			}
		}
		s := c.segment
		if len(result.Cues) > 0 && s.Start < previous.End {
			if s.Start == previous.Start && s.End == previous.End && s.Text == previous.Text && slices.Equal(s.Tokens, previous.Tokens) {
				continue
			}
			return zero, fmt.Errorf("%w at window %d", ErrTranscriptOverlap, c.window)
		}
		start, e := sampleTimestamp(s.Start, total)
		if e != nil {
			return zero, e
		}
		end, e := sampleTimestamp(s.End, total)
		if e != nil {
			return zero, e
		}
		if end <= start {
			return zero, ErrTranscriptSampleGrid
		}
		result.Cues = append(result.Cues, Cue{StartSample: start, EndSample: end, Speaker: -1, Text: s.Text})
		accepted = append(accepted, c)
		previous = s
	}
	if slices.ContainsFunc(accepted, func(c rawCue) bool { return c.words != nil }) {
		result.Words = []WordCue{}
	}
	for _, c := range accepted {
		for _, word := range c.words {
			start, e := sampleTimestamp(word.Start, total)
			if e != nil {
				return zero, e
			}
			end, e := sampleTimestamp(word.End, total)
			if e != nil || end <= start {
				return zero, ErrTranscriptSampleGrid
			}
			result.Words = append(result.Words, WordCue{StartSample: start, EndSample: end, Speaker: -1, Text: word.Word})
		}
	}
	if e = validateTranscript(ctx, result); e != nil {
		return zero, e
	}
	return result, nil
}
func sampleTimestamp(seconds float64, total int64) (int64, error) {
	if !finite(seconds) || seconds < 0 || seconds > float64(total)/16000 {
		return 0, ErrTranscriptSampleGrid
	}
	samples := seconds * 16000
	rounded := math.Round(samples)
	if math.Abs(samples-rounded) > 1e-6 || rounded < 0 || rounded > float64(total) {
		return 0, ErrTranscriptSampleGrid
	}
	return int64(rounded), nil
}
