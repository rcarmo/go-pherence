//go:build linux && amd64

package speechjob

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"strings"
	"testing"

	nem "github.com/rcarmo/go-pherence/model/nemotrondiarization"
)

type fakeNemotronPCMStream struct {
	fail    error
	calls   int
	samples int
}

func (s *fakeNemotronPCMStream) AppendPCMContext(ctx context.Context, pcm []float32) ([]float32, error) {
	s.calls++
	if s.fail != nil {
		return nil, s.fail
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.samples += len(pcm)
	rows := len(pcm) / 160
	if s.samples == len(pcm) && s.samples >= 16640 {
		rows--
	}
	logits := make([]float32, rows*8)
	for i := range logits {
		logits[i] = -1
	}
	for i := 0; i < rows; i++ {
		logits[i*8] = 1
	}
	return logits, nil
}
func (s *fakeNemotronPCMStream) FinishContext(ctx context.Context) ([]float32, error) {
	return nil, ctx.Err()
}
func nemotronTestConfig() NemotronDiarizationConfig {
	return NemotronDiarizationConfig{AllowExperimental: true, ModelSHA256: hash([]byte("nemotron")), RuntimeSHA256: hash([]byte("runtime")), ModelRevision: NemotronDiarizationRevision, MaxResultBytes: 65536}
}

func TestNemotronStagePreservesPrefixRetriesAndProgress(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	attempts := 0
	fail := errors.New("model failure")
	diar, err := newNemotronDiarizationStage(nemotronTestConfig(), func() (nemotronPCMStream, error) {
		attempts++
		stream := &fakeNemotronPCMStream{}
		if attempts == 1 {
			stream.fail = fail
		}
		return stream, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	stages := []Stage{fixturePCMStage(96000), textStage("transcript", "plain"), diar}
	job, err = s.Run(context.Background(), job.ID, config, stages, nil)
	if !errors.Is(err, fail) || job.Status != Failed || len(job.Checkpoints) != 2 {
		t.Fatal(job, err)
	}
	var progress []WorkProgress
	ctx := WithWorkProgress(context.Background(), func(p WorkProgress) { progress = append(progress, p) })
	job, err = s.Run(ctx, job.ID, config, stages, nil)
	if err != nil || job.Status != Complete || attempts != 2 || len(job.Checkpoints) != 3 {
		t.Fatal(job, err, attempts)
	}
	if len(progress) != 3 || progress[0].Completed != 0 || progress[1].Completed != 80000 || progress[2].Completed != 96000 || progress[2].Total != 96000 {
		t.Fatal(progress)
	}
	r, err := s.OpenCheckpoint(context.Background(), job.ID, "diarization")
	if err != nil {
		t.Fatal(err)
	}
	d, err := ReadNemotronDiarizationJSON(context.Background(), r)
	r.Close()
	if err != nil || d.Provider != "nemotron" || len(d.Turns) != 1 || d.Turns[0].End != 5.99 || d.StageKey != job.Checkpoints[2].Key {
		t.Fatal(d, err)
	}
	r, err = s.OpenCheckpoint(context.Background(), job.ID, "transcript")
	if readAll(t, r, err) != "plain" {
		t.Fatal("plain transcript lost")
	}
	assertNoScratch(t, s, job.ID)
}

func TestNemotronDocumentBoundsAndDistinctIdentity(t *testing.T) {
	cfg := nemotronTestConfig()
	stage, err := newNemotronDiarizationStage(cfg, func() (nemotronPCMStream, error) { return &fakeNemotronPCMStream{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	cfg.ModelSHA256 = hash([]byte("other"))
	changed, _ := newNemotronDiarizationStage(cfg, func() (nemotronPCMStream, error) { return &fakeNemotronPCMStream{}, nil })
	if changed.Version == stage.Version {
		t.Fatal("model identity ignored")
	}
	d := NemotronDiarizationDocument{Schema: 1, Provider: "nemotron", Experimental: true, SampleRate: 16000, TotalSamples: 16000, StageKey: hash([]byte("stage")), Turns: []nem.Segment{{Start: 0, End: .99, Speaker: 0}}}
	if err := validateNemotronDiarizationDocument(context.Background(), d); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(d)
	b = append(b, '\n')
	if _, err := ReadNemotronDiarizationJSON(context.Background(), strings.NewReader(string(b))); err != nil {
		t.Fatal(err)
	}
	for _, turn := range []nem.Segment{{Start: 0, End: math.NaN(), Speaker: 0}, {Start: -.01, End: .3, Speaker: 0}, {Start: 0, End: 1.01, Speaker: 0}, {Start: 0, End: .2, Speaker: 8}, {Start: .001, End: .3, Speaker: 0}} {
		bad := d
		bad.Turns = []nem.Segment{turn}
		if err := validateNemotronDiarizationDocument(context.Background(), bad); err == nil {
			t.Fatal("invalid turn", turn)
		}
	}
	if _, err := ReadDiarizationJSON(context.Background(), strings.NewReader(string(b))); err == nil {
		t.Fatal("Nemotron accepted as Community-1")
	}
	cfg.ModelRevision = "wrong"
	if _, err := newNemotronDiarizationStage(cfg, func() (nemotronPCMStream, error) { return &fakeNemotronPCMStream{}, nil }); err == nil {
		t.Fatal("revision ignored")
	}
}

func TestNemotronCancellationDoesNotPublishPartial(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	diar, err := newNemotronDiarizationStage(nemotronTestConfig(), func() (nemotronPCMStream, error) { cancel(); return &fakeNemotronPCMStream{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	job, err = s.Run(ctx, job.ID, config, []Stage{fixturePCMStage(16000), textStage("transcript", "retained"), diar}, nil)
	if !errors.Is(err, context.Canceled) || job.Status != Cancelled || len(job.Checkpoints) != 2 {
		t.Fatal(job, err)
	}
	assertNoScratch(t, s, job.ID)
}

func TestNemotronSpeakerAttributionUsesFullOverlappingSpans(t *testing.T) {
	t0 := Transcript{Schema: 2, SampleRate: 16000, TotalSamples: 16000, Language: "en", Cues: []Cue{{StartSample: 0, EndSample: 16000, Speaker: -1, Text: "one two"}}, Words: []WordCue{{StartSample: 0, EndSample: 4000, Speaker: -1, Text: "one"}, {StartSample: 8000, EndSample: 12000, Speaker: -1, Text: "two"}}}
	// Same speaker-turn type is used only for the shared attribution math, not a
	// Community-1 checkpoint. The second word overlaps two speakers equally.
	turns := []nem.Segment{{Start: 0, End: .99, Speaker: 0}, {Start: .5, End: .99, Speaker: 1}}
	s, _ := openTest(t)
	job := createTest(t, s)
	text := stage("transcript", func(ctx context.Context, _ *Input, out io.Writer) error { return WriteTranscriptJSON(ctx, out, t0) })
	diar := stage("diarization", func(ctx context.Context, in *Input, out io.Writer) error {
		d := NemotronDiarizationDocument{Schema: 1, Provider: "nemotron", Experimental: true, SampleRate: 16000, TotalSamples: 16000, StageKey: checkpointKey(in.job, Stage{Name: "diarization", Version: hash([]byte("diarizationv1"))}), Turns: turns}
		b, _ := json.Marshal(d)
		return writeContext(ctx, out, append(b, '\n'))
	})
	speaker, err := NewNemotronSpeakerTranscriptStage(SpeakerTranscriptConfig{TranscriptVersion: text.Version, DiarizationVersion: diar.Version, AllowExperimental: true})
	if err != nil {
		t.Fatal(err)
	}
	job, err = s.Run(context.Background(), job.ID, config, []Stage{text, diar, speaker, NewSpeakerVTTStage()}, nil)
	if err != nil {
		t.Fatal(job, err)
	}
	r, err := s.OpenCheckpoint(context.Background(), job.ID, "speaker-transcript")
	if err != nil {
		t.Fatal(err)
	}
	d, err := ReadSpeakerTranscriptJSON(context.Background(), r)
	r.Close()
	if err != nil || d.Policy != nemotronSpeakerPolicy || d.Transcript.Words[0].Speaker != 0 || d.Transcript.Words[1].Speaker != -1 {
		t.Fatal(d, err)
	}
	r, err = s.OpenCheckpoint(context.Background(), job.ID, "speaker-vtt")
	output := readAll(t, r, err)
	if !strings.Contains(output, "Nemotron") || strings.Contains(output, "Community-1") {
		t.Fatal(output)
	}
}
