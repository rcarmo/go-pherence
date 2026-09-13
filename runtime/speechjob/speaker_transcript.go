//go:build linux && amd64

package speechjob

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
)

// SpeakerTranscriptConfig pins both input stage versions. Experimental source
// labels require a separate explicit opt-in. Diagnostic tie-resolved results and
// gap-filled turns are rejected; neither establishes unique speaker coverage.
type SpeakerTranscriptConfig struct {
	TranscriptVersion, DiarizationVersion string
	AllowExperimental                     bool
}

// SpeakerTranscript retains provenance/experimental status outside the plain
// transcript schema. Raw diarization is never relabelled as qualified output.
type SpeakerTranscript struct {
	Schema          int        `json:"schema"`
	Experimental    bool       `json:"experimental"`
	TranscriptKey   string     `json:"transcript_key"`
	DiarizationKey  string     `json:"diarization_key"`
	Policy          string     `json:"policy"`
	LabelledCues    int        `json:"labelled_cues"`
	UnlabelledCues  int        `json:"unlabelled_cues"`
	LabelledWords   int        `json:"labelled_words"`
	UnlabelledWords int        `json:"unlabelled_words"`
	Transcript      Transcript `json:"transcript"`
}

const speakerCoveragePolicy = "exclusive-turns-maximum-positive-overlap-per-word-v3"

// NewSpeakerTranscriptStage creates "speaker-transcript" separately from the
// unlabelled "transcript" so failed diarization never erases downloadable text.
// Checked words receive the exclusive-turn speaker with maximum positive overlap.
// An exact overlap tie or no overlap leaves Speaker=-1. Legacy cues without word
// timings retain the conservative full-turn complete-coverage policy. No timing
// interpolation, nearest-speaker fallback, or identity recognition is performed.
func NewSpeakerTranscriptStage(cfg SpeakerTranscriptConfig) (Stage, error) {
	if !cfg.AllowExperimental || !validHash(cfg.TranscriptVersion) || !validHash(cfg.DiarizationVersion) {
		return Stage{}, ErrConfiguration
	}
	identity, _ := json.Marshal(struct {
		Schema string
		Config SpeakerTranscriptConfig
	}{speakerCoveragePolicy, cfg})
	return Stage{Name: "speaker-transcript", Version: hash(identity), Run: func(ctx context.Context, in *Input, w io.Writer) error {
		var text, diar Checkpoint
		for _, cp := range in.job.Checkpoints {
			switch cp.Stage {
			case "transcript":
				text = cp
			case "diarization":
				diar = cp
			}
		}
		if text.Version != cfg.TranscriptVersion || diar.Version != cfg.DiarizationVersion {
			return ErrConfiguration
		}
		tr, e := in.OpenCheckpoint(ctx, "transcript")
		if e != nil {
			return e
		}
		t, e := ReadTranscriptJSON(ctx, tr)
		e = errors.Join(e, tr.Close())
		if e != nil {
			return e
		}
		dr, e := in.OpenCheckpoint(ctx, "diarization")
		if e != nil {
			return e
		}
		d, e := ReadDiarizationJSON(ctx, dr)
		e = errors.Join(e, dr.Close())
		if e != nil {
			return e
		}
		if d.StageKey != diar.Key || d.TotalSamples != t.TotalSamples || d.SourceTiming != t.SourceTiming {
			return ErrCorrupt
		}
		result, e := labelSpeakerTranscript(ctx, t, d, text.Key)
		if e != nil {
			return e
		}
		b, e := json.Marshal(result)
		if e != nil {
			return e
		}
		b = append(b, '\n')
		if len(b) > MaxTranscriptBytes {
			return ErrLimit
		}
		if e = in.store.hit("speaker-transcript-output-ready"); e != nil {
			return e
		}
		return writeContext(ctx, w, b)
	}}, nil
}

type speakerSpan struct{ start, end float64 }

func labelSpeakerTranscript(ctx context.Context, t Transcript, d DiarizationDocument, textKey string) (SpeakerTranscript, error) {
	var zero SpeakerTranscript
	if e := validateTranscript(ctx, t); e != nil {
		return zero, e
	}
	if e := validateDiarizationDocument(ctx, d); e != nil {
		return zero, e
	}
	if !validHash(textKey) || d.TotalSamples != t.TotalSamples || d.SourceTiming != t.SourceTiming {
		return zero, ErrCorrupt
	}
	if d.Policy.MinDurationOff != 0 || len(d.AmbiguousFrames) > 0 || d.Path != "silence" && !d.ConstraintSatisfied {
		return zero, ErrConfiguration
	}
	var spans [64][]speakerSpan
	for i, turn := range d.FullTurns {
		if i%256 == 0 {
			if e := ctx.Err(); e != nil {
				return zero, e
			}
		}
		row := spans[turn.Speaker]
		// Only exactly contiguous same-speaker intervals may be joined. No positive
		// gap is filled and no competing speaker interval is removed.
		if len(row) > 0 && row[len(row)-1].end == turn.Start {
			row[len(row)-1].end = turn.End
		} else {
			row = append(row, speakerSpan{turn.Start, turn.End})
		}
		spans[turn.Speaker] = row
	}
	result := SpeakerTranscript{Schema: 2, Experimental: true, TranscriptKey: textKey, DiarizationKey: d.StageKey, Policy: speakerCoveragePolicy, Transcript: t}
	result.Transcript.Cues = append([]Cue(nil), t.Cues...)
	if t.Words != nil {
		result.Transcript.Words = append([]WordCue(nil), t.Words...)
	}
	label := func(start, end float64, source [64][]speakerSpan, requireFull bool) int {
		bestSpeaker := -1
		bestOverlap := 0.0
		tied := false
		intersecting := 0
		for speaker, row := range source {
			var overlap float64
			for _, span := range row {
				left, right := math.Max(start, span.start), math.Min(end, span.end)
				if right > left {
					overlap += right - left
				}
			}
			if overlap > 0 {
				intersecting++
			}
			if requireFull && overlap != end-start {
				continue
			}
			if overlap > bestOverlap {
				bestSpeaker, bestOverlap, tied = speaker, overlap, false
			} else if overlap > 0 && overlap == bestOverlap {
				tied = true
			}
		}
		if requireFull && intersecting != 1 || tied || bestOverlap <= 0 || bestSpeaker >= d.Clusters {
			return -1
		}
		return bestSpeaker
	}
	var exclusive [64][]speakerSpan
	for _, turn := range d.ExclusiveTurns {
		exclusive[turn.Speaker] = append(exclusive[turn.Speaker], speakerSpan{turn.Start, turn.End})
	}
	if len(result.Transcript.Words) > 0 && d.Path == "silence" {
		return zero, ErrConfiguration
	}
	for i, cue := range result.Transcript.Cues {
		if e := ctx.Err(); e != nil {
			return zero, e
		}
		if cue.Speaker != -1 {
			return zero, ErrConfiguration
		}
		start, end := float64(cue.StartSample)/16000, float64(cue.EndSample)/16000
		result.Transcript.Cues[i].Speaker = label(start, end, spans, true)
		if result.Transcript.Cues[i].Speaker >= 0 {
			result.LabelledCues++
		} else {
			result.UnlabelledCues++
		}
	}
	for i := range result.Transcript.Words {
		word := &result.Transcript.Words[i]
		if word.Speaker != -1 {
			return zero, ErrConfiguration
		}
		word.Speaker = label(float64(word.StartSample)/16000, float64(word.EndSample)/16000, exclusive, false)
		if word.Speaker >= 0 {
			result.LabelledWords++
		} else {
			result.UnlabelledWords++
		}
	}
	return result, ctx.Err()
}

// ReadSpeakerTranscriptJSON is a bounded canonical checkpoint reader. It checks
// document provenance shape and accounting, not speech/speaker accuracy.
func ReadSpeakerTranscriptJSON(ctx context.Context, r io.Reader) (SpeakerTranscript, error) {
	var d SpeakerTranscript
	if r == nil {
		return d, ErrCorrupt
	}
	var b bytes.Buffer
	buf := make([]byte, 32<<10)
	for {
		if e := ctx.Err(); e != nil {
			return d, e
		}
		n, e := r.Read(buf)
		if n > 0 {
			if b.Len()+n > MaxTranscriptBytes {
				return d, ErrLimit
			}
			b.Write(buf[:n])
		}
		if e == io.EOF {
			break
		}
		if e != nil {
			return d, e
		}
		if n == 0 {
			return d, io.ErrNoProgress
		}
	}
	if e := json.Unmarshal(b.Bytes(), &d); e != nil {
		return SpeakerTranscript{}, ErrCorrupt
	}
	canonical, e := json.Marshal(d)
	if e != nil {
		return SpeakerTranscript{}, ErrCorrupt
	}
	canonical = append(canonical, '\n')
	if !bytes.Equal(canonical, b.Bytes()) {
		return SpeakerTranscript{}, ErrCorrupt
	}
	if e = validateSpeakerTranscript(ctx, d); e != nil {
		return SpeakerTranscript{}, e
	}
	return d, ctx.Err()
}

func validateSpeakerTranscript(ctx context.Context, d SpeakerTranscript) error {
	if d.Schema != 2 || !d.Experimental || !validHash(d.TranscriptKey) || !validHash(d.DiarizationKey) || d.Policy != speakerCoveragePolicy || d.Transcript.Cues == nil {
		return ErrCorrupt
	}
	if e := validateTranscript(ctx, d.Transcript); e != nil {
		return e
	}
	labelled, labelledWords, words := 0, 0, 0
	for _, c := range d.Transcript.Cues {
		if c.Speaker >= 0 {
			labelled++
		}
	}
	for _, word := range d.Transcript.Words {
		words++
		if word.Speaker >= 0 {
			labelledWords++
		}
	}
	if d.LabelledCues != labelled || d.UnlabelledCues != len(d.Transcript.Cues)-labelled || d.LabelledWords != labelledWords || d.UnlabelledWords != words-labelledWords {
		return ErrCorrupt
	}
	return ctx.Err()
}

// WriteSpeakerTranscriptVTT emits one VTT cue per checked word when present and
// preserves legacy cue-level output otherwise.
func WriteSpeakerTranscriptVTT(ctx context.Context, w io.Writer, d SpeakerTranscript) error {
	if w == nil {
		return ErrConfiguration
	}
	if err := validateSpeakerTranscript(ctx, d); err != nil {
		return err
	}
	transcript := d.Transcript
	if len(transcript.Words) > 0 {
		cues := make([]Cue, len(transcript.Words))
		for i, word := range transcript.Words {
			cues[i] = Cue{StartSample: word.StartSample, EndSample: word.EndSample, Speaker: word.Speaker, Text: word.Text}
		}
		transcript.Cues = cues
		transcript.Words = nil
	}
	var body bytes.Buffer
	if err := WriteWebVTT(ctx, &body, transcript); err != nil {
		return err
	}
	const header = "WEBVTT\n\n"
	note := "NOTE Experimental Community-1 speaker labels; exclusive-turn word attribution when available.\n\n"
	if body.Len()+len(note) > MaxTranscriptBytes {
		return ErrLimit
	}
	if err := writeContext(ctx, w, []byte(header+note)); err != nil {
		return err
	}
	return writeContext(ctx, w, body.Bytes()[len(header):])
}

// NewSpeakerVTTStage emits a distinct VTT with an explicit experimental NOTE.
// Source provenance is retained in speaker-transcript and the original plain
// transcript/VTT are untouched. Checked words become individual cues so each
// may carry its exclusive-turn speaker; legacy cue-level output is preserved.
func NewSpeakerVTTStage() Stage {
	return Stage{Name: "speaker-vtt", Version: hash([]byte("speaker-vtt-v3:" + speakerCoveragePolicy + ":transcript-schema2-source-timing:namedrefs-experimental-note")), Run: func(ctx context.Context, in *Input, w io.Writer) error {
		r, e := in.OpenCheckpoint(ctx, "speaker-transcript")
		if e != nil {
			return e
		}
		d, e := ReadSpeakerTranscriptJSON(ctx, r)
		e = errors.Join(e, r.Close())
		if e != nil {
			return e
		}
		// Validate provenance against the current verified checkpoint prefix.
		var textKey, diarKey string
		for _, cp := range in.job.Checkpoints {
			if cp.Stage == "transcript" {
				textKey = cp.Key
			}
			if cp.Stage == "diarization" {
				diarKey = cp.Key
			}
		}
		if d.TranscriptKey != textKey || d.DiarizationKey != diarKey {
			return ErrCorrupt
		}
		if e = in.store.hit("speaker-vtt-output-ready"); e != nil {
			return e
		}
		return WriteSpeakerTranscriptVTT(ctx, w, d)
	}}
}
