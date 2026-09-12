//go:build linux && amd64

package speechjob

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
	"github.com/rcarmo/go-pherence/models/whisper"
)

func speakerDocument(t *testing.T, turns []c1.SpeakerTurn) DiarizationDocument {
	t.Helper()
	cfg := communityConfig()
	r := communityFixtureResult(3361, cfg)
	r.Postprocess.Path = "clustered"
	r.Postprocess.TrainingRows = 2
	r.Postprocess.Clusters = 2
	r.Postprocess.Timeline.Classes = 3
	r.Postprocess.FullTurns = turns
	r.Postprocess.ExclusiveTurns = nil
	d, e := diarizationDocument(context.Background(), r, cfg.PCM, 3361, hash([]byte("diar key")))
	if e != nil {
		t.Fatal(e)
	}
	return d
}
func speakerCue() Transcript {
	return Transcript{Schema: 1, SampleRate: 16000, TotalSamples: 3361, Language: "pt", Cues: []Cue{{StartSample: 320, EndSample: 1280, Speaker: -1, Text: "Olá"}}}
}
func TestSpeakerCompleteCoverage(t *testing.T) {
	for _, tc := range []struct {
		name  string
		turns []c1.SpeakerTurn
		want  int
	}{
		{"full", []c1.SpeakerTurn{{Start: .01, End: .1, Speaker: 0}}, 0},
		{"exact", []c1.SpeakerTurn{{Start: .02, End: .08, Speaker: 1}}, 1},
		{"partial", []c1.SpeakerTurn{{Start: .03, End: .08, Speaker: 0}}, -1},
		{"tiny-gap", []c1.SpeakerTurn{{Start: .02, End: .04, Speaker: 0}, {Start: .040000000000001, End: .08, Speaker: 0}}, -1},
		{"contiguous", []c1.SpeakerTurn{{Start: .02, End: .04, Speaker: 0}, {Start: .04, End: .08, Speaker: 0}}, 0},
		{"switch", []c1.SpeakerTurn{{Start: .02, End: .04, Speaker: 0}, {Start: .04, End: .08, Speaker: 1}}, -1},
		{"competing", []c1.SpeakerTurn{{Start: .01, End: .1, Speaker: 0}, {Start: .03, End: .04, Speaker: 1}}, -1},
		{"touch-only", []c1.SpeakerTurn{{Start: .01, End: .02, Speaker: 1}, {Start: .02, End: .08, Speaker: 0}, {Start: .08, End: .1, Speaker: 1}}, 0},
		{"padded-class", []c1.SpeakerTurn{{Start: .01, End: .1, Speaker: 2}}, -1},
		{"none", nil, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := speakerDocument(t, tc.turns)
			tr := speakerCue()
			before := tr.Cues[0]
			out, e := labelSpeakerTranscript(context.Background(), tr, d, hash([]byte("text")))
			if e != nil {
				t.Fatal(e)
			}
			if out.Transcript.Cues[0].Speaker != tc.want || tr.Cues[0] != before || !out.Experimental || out.LabelledCues+out.UnlabelledCues != 1 {
				t.Fatal(out)
			}
		})
	}
}
func TestSpeakerPolicyAndDocumentValidation(t *testing.T) {
	ctx := context.Background()
	key := hash([]byte("text"))
	d := speakerDocument(t, []c1.SpeakerTurn{{Start: .01, End: .1, Speaker: 0}})
	tr := speakerCue()
	good, e := labelSpeakerTranscript(ctx, tr, d, key)
	if e != nil {
		t.Fatal(e)
	}
	raw, _ := json.Marshal(good)
	raw = append(raw, '\n')
	read, e := ReadSpeakerTranscriptJSON(ctx, bytes.NewReader(raw))
	if e != nil || !reflect.DeepEqual(read, good) {
		t.Fatal(read, e)
	}
	for _, kind := range []string{"gap-fill", "tie", "extent", "already-labelled", "constraint"} {
		c := d
		tt := speakerCue()
		switch kind {
		case "gap-fill":
			c.Policy.MinDurationOff = .1
		case "tie":
			c.Policy.TiePolicy = c1.LowestIndexTies
			c.AmbiguousFrames = []int{1}
		case "extent":
			tt.TotalSamples++
		case "already-labelled":
			tt.Cues[0].Speaker = 0
		case "constraint":
			c.Path = "single-training-row"
			c.TrainingRows = 1
			c.Clusters = 1
			c.ConstraintSatisfied = false
			c.Policy.NumSpeakers = 4
		}
		if _, e = labelSpeakerTranscript(ctx, tt, c, key); e == nil {
			t.Fatal("bad policy", kind)
		}
	}
	for _, b := range [][]byte{
		[]byte("null"), bytes.Replace(raw, []byte(`"schema":1`), []byte(`"schema":1,"schema":1`), 1),
		bytes.Replace(raw, []byte(`"experimental":true`), []byte(`"experimental":false`), 1),
		bytes.Replace(raw, []byte(`"labelled_cues":1`), []byte(`"labelled_cues":0`), 1),
		bytes.Replace(raw, []byte(`"unlabelled_cues":0,`), nil, 1),
		append([]byte(" "), raw...), bytes.Repeat([]byte{'x'}, MaxTranscriptBytes+1),
	} {
		if _, e = ReadSpeakerTranscriptJSON(ctx, bytes.NewReader(b)); e == nil {
			t.Fatal("invalid canonical document accepted")
		}
	}
	cc, cancel := context.WithCancel(ctx)
	cancel()
	if _, e = labelSpeakerTranscript(cc, tr, d, key); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if _, e = ReadSpeakerTranscriptJSON(cc, bytes.NewReader(raw)); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	for _, cfg := range []SpeakerTranscriptConfig{{}, {TranscriptVersion: key, DiarizationVersion: key}, {AllowExperimental: true, TranscriptVersion: "bad", DiarizationVersion: key}} {
		if _, e := NewSpeakerTranscriptStage(cfg); e == nil {
			t.Fatal("invalid config")
		}
	}
}
func speakerPipeline(t *testing.T, failDiar bool) []Stage {
	t.Helper()
	cfg := reconcileConfig()
	plain, e := NewTranscriptStage(cfg)
	if e != nil {
		t.Fatal(e)
	}
	diarCfg := communityConfig()
	diar := community1Stage(diarCfg, func(_ context.Context, _ c1.DiarizationPCMReader, total int64) (*c1.DiarizationPCMResult, error) {
		if failDiar {
			return nil, io.ErrClosedPipe
		}
		return communityFixtureResult(total, diarCfg), nil
	})
	speakers, e := NewSpeakerTranscriptStage(SpeakerTranscriptConfig{TranscriptVersion: plain.Version, DiarizationVersion: diar.Version, AllowExperimental: true})
	if e != nil {
		t.Fatal(e)
	}
	// 21 windows for this tiny test geometry; only one segment in a non-overlap
	// region. Every other window is explicitly empty; no missing-record shortcut.
	asr := Stage{Name: "asr-windows", Version: cfg.ASRVersion, Run: func(ctx context.Context, in *Input, w io.Writer) error {
		key := checkpointKey(in.job, Stage{Name: "asr-windows", Version: cfg.ASRVersion})
		return writeContext(ctx, w, asrRecords(t, 3361, key, cfg, [][]whisper.Segment{{{Start: .0125, End: .0175, Text: "Olá <&>", Tokens: []int{42}}}}))
	}}
	return []Stage{fixturePCMStage(3361), asr, plain, NewVTTStage(), diar, speakers, NewSpeakerVTTStage()}
}
func TestSpeakerPipelineFailureRetentionAndResume(t *testing.T) {
	s, dir := openTest(t)
	job := createTest(t, s)
	job, e := s.Run(context.Background(), job.ID, config, speakerPipeline(t, true), nil)
	if !errors.Is(e, io.ErrClosedPipe) || len(job.Checkpoints) != 4 {
		t.Fatal(job, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), job.ID, "vtt")
	plain := readAll(t, r, e)
	if !strings.Contains(plain, "Olá") || strings.Contains(plain, "SPEAKER") {
		t.Fatal(plain)
	}
	s.Close()
	s, e = Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	job, e = s.Run(context.Background(), job.ID, config, speakerPipeline(t, false), nil)
	if e != nil || job.Status != Complete || len(job.Checkpoints) != 7 {
		t.Fatal(job, e)
	}
	r, e = s.OpenCheckpoint(context.Background(), job.ID, "vtt")
	if readAll(t, r, e) != plain {
		t.Fatal("plain text replaced")
	}
	r, e = s.OpenCheckpoint(context.Background(), job.ID, "speaker-transcript")
	speaker, e := ReadSpeakerTranscriptJSON(context.Background(), strings.NewReader(readAll(t, r, e)))
	if e != nil || speaker.LabelledCues != 1 {
		t.Fatal(speaker, e)
	}
	r, e = s.OpenCheckpoint(context.Background(), job.ID, "speaker-vtt")
	vtt := readAll(t, r, e)
	if !strings.Contains(vtt, "NOTE Experimental Community-1") || !strings.Contains(vtt, "<v SPEAKER_00>Olá &lt;&amp;&gt;</v>") {
		t.Fatal(vtt)
	}
	// Exact same pipeline and input on a fresh job yields identical output bytes.
	fresh := createTest(t, s)
	fresh, e = s.Run(context.Background(), fresh.ID, config, speakerPipeline(t, false), nil)
	if e != nil {
		t.Fatal(e)
	}
	r, e = s.OpenCheckpoint(context.Background(), fresh.ID, "speaker-vtt")
	if readAll(t, r, e) != vtt {
		t.Fatal("nondeterministic speaker VTT")
	}
}

func TestSpeakerPipelineRejectsWrongProvenance(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	stages := speakerPipeline(t, false)
	original := stages[4].Run
	stages[4].Run = func(ctx context.Context, in *Input, w io.Writer) error {
		var b bytes.Buffer
		if e := original(ctx, in, &b); e != nil {
			return e
		}
		d, e := ReadDiarizationJSON(ctx, &b)
		if e != nil {
			return e
		}
		d.StageKey = hash([]byte("foreign job"))
		raw, e := json.Marshal(d)
		if e != nil {
			return e
		}
		return writeContext(ctx, w, append(raw, '\n'))
	}
	job, e := s.Run(context.Background(), job.ID, config, stages, nil)
	if !errors.Is(e, ErrCorrupt) || len(job.Checkpoints) != 5 {
		t.Fatal(job, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), job.ID, "vtt")
	if !strings.Contains(readAll(t, r, e), "Olá") {
		t.Fatal("plain text lost")
	}
}
