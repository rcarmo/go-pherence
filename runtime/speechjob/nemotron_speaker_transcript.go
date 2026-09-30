//go:build linux && amd64

package speechjob

import (
	"context"
	"encoding/json"
	"errors"
	"io"

	c1 "github.com/rcarmo/go-pherence/model/speaker/community1"
)

// NewNemotronSpeakerTranscriptStage keeps the existing transcript/export schema,
// but consumes only Nemotron's own checkpoint. Overlapping model spans remain
// overlapping: an exact word-overlap tie stays unlabelled rather than choosing
// a fabricated exclusive speaker.
func NewNemotronSpeakerTranscriptStage(cfg SpeakerTranscriptConfig) (Stage, error) {
	if !cfg.AllowExperimental || !validHash(cfg.TranscriptVersion) || !validHash(cfg.DiarizationVersion) {
		return Stage{}, ErrConfiguration
	}
	identity, _ := json.Marshal(struct {
		Schema string
		Config SpeakerTranscriptConfig
	}{nemotronSpeakerPolicy, cfg})
	return Stage{Name: "speaker-transcript", Version: hash(identity), Run: func(ctx context.Context, in *Input, out io.Writer) error {
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
		r, err := in.OpenCheckpoint(ctx, "transcript")
		if err != nil {
			return err
		}
		t, err := ReadTranscriptJSON(ctx, r)
		err = errors.Join(err, r.Close())
		if err != nil {
			return err
		}
		r, err = in.OpenCheckpoint(ctx, "diarization")
		if err != nil {
			return err
		}
		d, err := ReadNemotronDiarizationJSON(ctx, r)
		err = errors.Join(err, r.Close())
		if err != nil {
			return err
		}
		if d.StageKey != diar.Key || d.TotalSamples != t.TotalSamples || d.SourceTiming != t.SourceTiming {
			return ErrCorrupt
		}
		if err = validateTranscript(ctx, t); err != nil {
			return err
		}
		turns := make([]c1.SpeakerTurn, len(d.Turns))
		for i, turn := range d.Turns {
			turns[i] = c1.SpeakerTurn{Start: turn.Start, End: turn.End, Speaker: turn.Speaker}
		}
		result, err := labelTranscriptTurns(ctx, t, turns, turns, 8, text.Key, d.StageKey, nemotronSpeakerPolicy, false)
		if err != nil {
			return err
		}
		data, err := json.Marshal(result)
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if len(data) > MaxTranscriptBytes {
			return ErrLimit
		}
		return writeContext(ctx, out, data)
	}}, nil
}
