//go:build linux && amd64

package speechjob

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"sort"
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
	Schema         int        `json:"schema"`
	Experimental   bool       `json:"experimental"`
	TranscriptKey  string     `json:"transcript_key"`
	DiarizationKey string     `json:"diarization_key"`
	Policy         string     `json:"policy"`
	LabelledCues   int        `json:"labelled_cues"`
	UnlabelledCues int        `json:"unlabelled_cues"`
	Transcript     Transcript `json:"transcript"`
}

const speakerCoveragePolicy = "full-turns-unique-complete-cue-source-timing-v2"

// NewSpeakerTranscriptStage creates "speaker-transcript" separately from the
// unlabelled "transcript" so failed diarization never erases downloadable text.
// A cue receives a cluster ID only if full turns cover its entire duration with
// one speaker and no other speaker intersects it. Partial coverage, gaps, speaker
// changes and overlaps leave Speaker=-1. No cue splitting or word interpolation.
// Comparisons use raw float64 seconds; no widening tolerance or tail clipping.
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
	result := SpeakerTranscript{Schema: 1, Experimental: true, TranscriptKey: textKey, DiarizationKey: d.StageKey, Policy: speakerCoveragePolicy, Transcript: t}
	result.Transcript.Cues = append([]Cue{}, t.Cues...)
	for i, cue := range result.Transcript.Cues {
		if e := ctx.Err(); e != nil {
			return zero, e
		}
		if cue.Speaker != -1 {
			return zero, ErrConfiguration
		}
		start, end := float64(cue.StartSample)/16000, float64(cue.EndSample)/16000
		candidate := -1
		intersecting := 0
		for speaker, row := range spans {
			j := sort.Search(len(row), func(j int) bool { return row[j].end > start })
			if j == len(row) || row[j].start >= end {
				continue
			}
			intersecting++
			if row[j].start <= start && row[j].end >= end {
				candidate = speaker
			}
		}
		if intersecting == 1 && candidate >= 0 && candidate < d.Clusters {
			result.Transcript.Cues[i].Speaker = candidate
			result.LabelledCues++
		} else {
			result.UnlabelledCues++
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
	if !bytes.Equal(canonical, b.Bytes()) || d.Schema != 1 || !d.Experimental || !validHash(d.TranscriptKey) || !validHash(d.DiarizationKey) || d.Policy != speakerCoveragePolicy || d.Transcript.Cues == nil {
		return SpeakerTranscript{}, ErrCorrupt
	}
	if e = validateTranscript(ctx, d.Transcript); e != nil {
		return SpeakerTranscript{}, e
	}
	labelled := 0
	for _, c := range d.Transcript.Cues {
		if c.Speaker >= 0 {
			labelled++
		}
	}
	if d.LabelledCues != labelled || d.UnlabelledCues != len(d.Transcript.Cues)-labelled {
		return SpeakerTranscript{}, ErrCorrupt
	}
	return d, ctx.Err()
}

// NewSpeakerVTTStage emits a distinct VTT with an explicit experimental NOTE.
// Source provenance is also retained in speaker-transcript; the original plain
// transcript/VTT are untouched. No new inference or speaker mapping is performed.
func NewSpeakerVTTStage() Stage {
	return Stage{Name: "speaker-vtt", Version: hash([]byte("speaker-vtt-v2:" + speakerCoveragePolicy + ":transcript-schema2-source-timing:namedrefs-experimental-note")), Run: func(ctx context.Context, in *Input, w io.Writer) error {
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
		var body bytes.Buffer
		if e = WriteWebVTT(ctx, &body, d.Transcript); e != nil {
			return e
		}
		const header = "WEBVTT\n\n"
		note := "NOTE Experimental Community-1 speaker labels; complete-cue coverage only.\n\n"
		if body.Len()+len(note) > MaxTranscriptBytes {
			return ErrLimit
		}
		if e = in.store.hit("speaker-vtt-output-ready"); e != nil {
			return e
		}
		if e = writeContext(ctx, w, []byte(header+note)); e != nil {
			return e
		}
		return writeContext(ctx, w, body.Bytes()[len(header):])
	}}
}
