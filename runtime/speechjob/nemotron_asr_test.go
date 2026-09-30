//go:build linux && amd64

package speechjob

import (
	"context"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

type fakeASR struct {
	calls int
	fail  error
	bad   bool
}

func (s *fakeASR) AppendPCM(ctx context.Context, pcm []float32) ([]int, []int64, error) {
	s.calls++
	if s.fail != nil {
		return nil, nil, s.fail
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if s.bad {
		return []int{3}, nil, nil
	}
	if s.calls == 1 {
		return []int{3, 4}, []int64{0, 1}, nil
	}
	return []int{5, 6}, []int64{70, 71}, nil
}
func (s *fakeASR) Finish(ctx context.Context) ([]int, []int64, error) { return nil, nil, ctx.Err() }
func asrConfig() NemotronASRConfig {
	return NemotronASRConfig{ModelSHA256: hash([]byte("asr-model")), TokenizerSHA256: hash([]byte("vocab")), RuntimeSHA256: hash([]byte("runtime")), ModelRevision: NemotronASRRevision, Language: "auto", PromptID: 101, MaxResultBytes: 65536}
}
func asrVocab() *tokenizer.Tokenizer {
	v := &tokenizer.Tokenizer{InvVocab: map[int]string{}, AddedSpecial: map[string]int{"<unk>": 0}}
	for i := 0; i < 13088; i++ {
		v.InvVocab[i] = "<unk>"
	}
	v.InvVocab[3] = "▁Hello"
	v.InvVocab[4] = "▁wor"
	v.InvVocab[5] = "ld"
	v.InvVocab[6] = "▁again"
	return v
}

func TestNemotronASRRetryPreservesDecodeAndStartsFresh(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	attempts := 0
	failure := errors.New("inference failed")
	asr, err := newNemotronASRStage(asrVocab(), asrConfig(), func() nemotronASRStream {
		attempts++
		stream := &fakeASR{}
		if attempts == 1 {
			stream.fail = failure
		}
		return stream
	})
	if err != nil {
		t.Fatal(err)
	}
	stages := []Stage{fixturePCMStage(96000), asr, NewVTTStage()}
	job, err = s.Run(context.Background(), job.ID, config, stages, nil)
	if !errors.Is(err, failure) || job.Status != Failed || len(job.Checkpoints) != 1 {
		t.Fatal(job, err)
	}
	prefix := job.Checkpoints[0]
	var progress []WorkProgress
	ctx := WithWorkProgress(context.Background(), func(p WorkProgress) { progress = append(progress, p) })
	job, err = s.Run(ctx, job.ID, config, stages, nil)
	if err != nil || attempts != 2 || job.Status != Complete || job.Checkpoints[0] != prefix {
		t.Fatal(job, err, attempts)
	}
	r, err := s.OpenCheckpoint(context.Background(), job.ID, "transcript")
	if err != nil {
		t.Fatal(err)
	}
	text, err := ReadTranscriptJSON(context.Background(), r)
	r.Close()
	if err != nil || text.Language != "und" || text.Timing != NemotronEmissionTiming || len(text.Words) != 0 || len(text.Cues) != 2 {
		t.Fatal(text, err)
	}
	if text.Cues[0].Text != "Hello" || text.Cues[1].Text != "world again" || text.Cues[0].StartSample != 0 || text.Cues[0].EndSample != 80000 || text.Cues[1].StartSample != 80000 || text.Cues[1].EndSample != 96000 {
		t.Fatal(text)
	}
	r, err = s.OpenCheckpoint(context.Background(), job.ID, "vtt")
	vtt := readAll(t, r, err)
	if !strings.Contains(vtt, "no word alignment") || !strings.Contains(vtt, "world again") {
		t.Fatal(vtt)
	}
	if len(progress) != 3 || progress[0].Completed != 0 || progress[1].Completed != 80000 || progress[2].Completed != 96000 || progress[1].Phase != "samples" {
		t.Fatal(progress)
	}
	assertNoScratch(t, s, job.ID)
}

func TestNemotronASRCancellationIsAtomicAndDecodeRetained(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stage, err := newNemotronASRStage(asrVocab(), asrConfig(), func() nemotronASRStream { cancel(); return &fakeASR{} })
	if err != nil {
		t.Fatal(err)
	}
	job, err = s.Run(ctx, job.ID, config, []Stage{fixturePCMStage(96000), stage}, nil)
	if !errors.Is(err, context.Canceled) || job.Status != Cancelled || len(job.Checkpoints) != 1 || job.MediaReleased {
		t.Fatal(job, err)
	}
	assertNoScratch(t, s, job.ID)
}

func TestNemotronASRRejectsConfigurationAndDecisionGeometry(t *testing.T) {
	for _, change := range []func(*NemotronASRConfig){func(c *NemotronASRConfig) { c.ModelRevision = "wrong" }, func(c *NemotronASRConfig) { c.TokenizerSHA256 = "wrong" }, func(c *NemotronASRConfig) { c.Language = "en" }, func(c *NemotronASRConfig) { c.PromptID = 128 }, func(c *NemotronASRConfig) { c.MaxResultBytes = MaxTranscriptBytes + 1 }} {
		c := asrConfig()
		change(&c)
		if _, err := newNemotronASRStage(asrVocab(), c, func() nemotronASRStream { return &fakeASR{} }); err == nil {
			t.Fatal(c)
		}
	}
	c := asrConfig()
	c.Language = "en"
	c.PromptID = 0
	one, err := newNemotronASRStage(asrVocab(), c, func() nemotronASRStream { return &fakeASR{} })
	if err != nil {
		t.Fatal(err)
	}
	c.Language = "pt"
	c.PromptID = 13
	two, _ := newNemotronASRStage(asrVocab(), c, func() nemotronASRStream { return &fakeASR{} })
	if one.Version == two.Version {
		t.Fatal("locale identity omitted")
	}
	s, _ := openTest(t)
	job := createTest(t, s)
	bad, _ := newNemotronASRStage(asrVocab(), asrConfig(), func() nemotronASRStream { return &fakeASR{bad: true} })
	job, err = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(96000), bad}, nil)
	if !errors.Is(err, ErrCorrupt) || len(job.Checkpoints) != 1 {
		t.Fatal(job, err)
	}
}

func TestNemotronCoarseTimingRoundtripAndLegacyBytes(t *testing.T) {
	text := Transcript{Schema: 2, SampleRate: 16000, TotalSamples: 16000, Language: "en", Cues: []Cue{{StartSample: 0, EndSample: 16000, Speaker: -1, Text: "hello"}}}
	var legacy, coarse strings.Builder
	if err := WriteTranscriptJSON(context.Background(), &legacy, text); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(legacy.String(), `"timing":`) {
		t.Fatal("legacy bytes changed")
	}
	text.Timing = NemotronEmissionTiming
	if err := WriteTranscriptJSON(context.Background(), &coarse, text); err != nil {
		t.Fatal(err)
	}
	got, err := ReadTranscriptJSON(context.Background(), strings.NewReader(coarse.String()))
	if err != nil || !reflect.DeepEqual(text, got) {
		t.Fatal(got, err)
	}
	text.Words = []WordCue{{StartSample: 0, EndSample: 16000, Speaker: -1, Text: "hello"}}
	if err := WriteTranscriptJSON(context.Background(), io.Discard, text); err == nil {
		t.Fatal("coarse declared word alignment")
	}
}
