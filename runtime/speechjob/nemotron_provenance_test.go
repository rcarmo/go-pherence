//go:build linux && amd64

package speechjob

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	nem "github.com/rcarmo/go-pherence/model/nemotrondiarization"
)

func TestNemotronSpeakerRejectsMismatchedProvenance(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*NemotronDiarizationDocument)
	}{
		{"stage-key", func(d *NemotronDiarizationDocument) { d.StageKey = hash([]byte("other")) }},
		{"duration", func(d *NemotronDiarizationDocument) { d.TotalSamples = 32000 }},
		{"provider", func(d *NemotronDiarizationDocument) { d.Provider = "community1" }},
		{"experimental", func(d *NemotronDiarizationDocument) { d.Experimental = false }},
		{"source-timing", func(d *NemotronDiarizationDocument) { d.SourceTiming.Start = 1000000000 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, _ := openTest(t)
			job := createTest(t, s)
			text := stage("transcript", func(ctx context.Context, _ *Input, out io.Writer) error {
				return WriteTranscriptJSON(ctx, out, Transcript{Schema: 2, SampleRate: 16000, TotalSamples: 16000, Language: "en", Cues: []Cue{{StartSample: 0, EndSample: 16000, Speaker: -1, Text: "retained"}}})
			})
			diar := stage("diarization", func(ctx context.Context, in *Input, out io.Writer) error {
				d := NemotronDiarizationDocument{Schema: 1, Provider: "nemotron", Experimental: true, SampleRate: 16000, TotalSamples: 16000, StageKey: checkpointKey(in.job, Stage{Name: "diarization", Version: hash([]byte("diarizationv1"))}), Turns: []nem.Segment{}}
				tc.change(&d)
				data, _ := json.Marshal(d)
				return writeContext(ctx, out, append(data, '\n'))
			})
			speaker, err := NewNemotronSpeakerTranscriptStage(SpeakerTranscriptConfig{TranscriptVersion: text.Version, DiarizationVersion: diar.Version, AllowExperimental: true})
			if err != nil {
				t.Fatal(err)
			}
			job, err = s.Run(context.Background(), job.ID, config, []Stage{text, NewVTTStage(), diar, speaker}, nil)
			if !errors.Is(err, ErrCorrupt) || job.Status != Failed || len(job.Checkpoints) != 3 {
				t.Fatal(job, err)
			}
			r, err := s.OpenCheckpoint(context.Background(), job.ID, "vtt")
			if output := readAll(t, r, err); !strings.Contains(output, "retained") {
				t.Fatal(output)
			}
			assertNoScratch(t, s, job.ID)
		})
	}
}

type invalidNemotronStream struct{ rows int }

func (s invalidNemotronStream) AppendPCMContext(context.Context, []float32) ([]float32, error) {
	return make([]float32, s.rows*8), nil
}
func (s invalidNemotronStream) FinishContext(context.Context) ([]float32, error) { return nil, nil }

func TestNemotronStageRejectsIncompleteOrExtraFrames(t *testing.T) {
	for _, rows := range []int{0, 99, 101} {
		s, _ := openTest(t)
		job := createTest(t, s)
		diar, err := newNemotronDiarizationStage(nemotronTestConfig(), func() (nemotronPCMStream, error) { return invalidNemotronStream{rows}, nil })
		if err != nil {
			t.Fatal(err)
		}
		job, err = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(16000), textStage("transcript", "retained"), diar}, nil)
		if !errors.Is(err, ErrCorrupt) || len(job.Checkpoints) != 2 {
			t.Fatal(rows, job, err)
		}
		assertNoScratch(t, s, job.ID)
	}
}

func TestNemotronReaderRequiresCanonicalBoundedJSON(t *testing.T) {
	d := NemotronDiarizationDocument{Schema: 1, Provider: "nemotron", Experimental: true, SampleRate: 16000, TotalSamples: 16000, StageKey: hash([]byte("stage")), Turns: []nem.Segment{}}
	data, _ := json.Marshal(d)
	canonical := string(data) + "\n"
	for _, invalid := range []string{string(data), canonical + "\n", canonical + "{}", strings.Replace(canonical, `"schema":1`, `"schema":1,"schema":1`, 1), strings.Replace(canonical, `"provider":"nemotron"`, `"provider":"nemotron","unexpected":true`, 1)} {
		if _, err := ReadNemotronDiarizationJSON(context.Background(), strings.NewReader(invalid)); !errors.Is(err, ErrCorrupt) {
			t.Fatal(err)
		}
	}
	if _, err := ReadNemotronDiarizationJSON(context.Background(), strings.NewReader(strings.Repeat(" ", 16<<20+1))); !errors.Is(err, ErrLimit) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ReadNemotronDiarizationJSON(ctx, strings.NewReader(canonical)); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
