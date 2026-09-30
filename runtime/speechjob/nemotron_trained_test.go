//go:build linux && amd64

package speechjob

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	nem "github.com/rcarmo/go-pherence/model/nemotrondiarization"
)

// Opt-in CPU-only provider smoke. The ASR transcript is a timing fixture: this
// checks native diarization, integration and retention, not Whisper accuracy.
func TestNemotronTrainedCPUJobRetryChunkParity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_JOB") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_JOB=1")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Fatal("missing pinned Nemotron model")
	}
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, loadErr := nem.LoadPCMStreamingRequest(f)
	if err := errors.Join(loadErr, f.Close()); err != nil {
		t.Fatal(err)
	}
	cfg := nemotronTestConfig()
	cfg.ModelSHA256 = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"
	diar, err := NewNemotronDiarizationStage(model, cfg)
	if err != nil {
		t.Fatal(err)
	}
	decode, err := NewGo264DecodeStage(Go264DecodeConfig{InputExtension: ".wav", MaxInputBytes: 1 << 20, MaxOutputBytes: 1 << 20, MaxDuration: 20 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	transcript := Transcript{Schema: 2, SampleRate: 16000, TotalSamples: 176000, Language: "en", Cues: []Cue{{StartSample: 0, EndSample: 176000, Speaker: -1, Text: "fixture one two"}}, Words: []WordCue{
		{StartSample: 8000, EndSample: 16000, Speaker: -1, Text: "fixture"},
		{StartSample: 56000, EndSample: 64000, Speaker: -1, Text: "one"},
		{StartSample: 96000, EndSample: 104000, Speaker: -1, Text: "two"},
	}}
	asrRuns := 0
	text := stage("transcript", func(ctx context.Context, in *Input, out io.Writer) error {
		asrRuns++
		var decoded Blob
		for _, cp := range in.job.Checkpoints {
			if cp.Stage == "decode" {
				decoded = cp.Blob
			}
		}
		pcm, err := media.OpenCanonicalPCM(ctx, filepath.Join(in.store.root.Name(), in.job.ID, decoded.File))
		if err != nil {
			return err
		}
		transcript.SourceTiming = pcm.SourceTiming()
		if err := pcm.Close(); err != nil {
			return err
		}
		return WriteTranscriptJSON(ctx, out, transcript)
	})
	speaker, err := NewNemotronSpeakerTranscriptStage(SpeakerTranscriptConfig{AllowExperimental: true, TranscriptVersion: text.Version, DiarizationVersion: diar.Version})
	if err != nil {
		t.Fatal(err)
	}
	stages := []Stage{decode, text, NewVTTStage(), diar, speaker, NewSpeakerVTTStage()}
	s, _ := openTest(t)
	wav, err := os.Open(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil {
		t.Fatal(err)
	}
	job, err := s.Create(context.Background(), "jfk.wav", config, wav)
	wav.Close()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	ctx = WithWorkProgress(ctx, func(p WorkProgress) {
		if p.Stage == "diarization" && p.Completed >= 80000 {
			cancel()
		}
	})
	start := time.Now()
	job, err = s.Run(ctx, job.ID, config, stages, nil)
	cancel()
	if !errors.Is(err, context.Canceled) || job.Status != Cancelled || len(job.Checkpoints) != 3 {
		t.Fatal(job, err)
	}
	prefix := append([]Checkpoint(nil), job.Checkpoints...)
	r, err := s.OpenCheckpoint(context.Background(), job.ID, "vtt")
	if output := readAll(t, r, err); !strings.Contains(output, "fixture one two") {
		t.Fatal(output)
	}
	var progress []WorkProgress
	ctx = WithWorkProgress(context.Background(), func(p WorkProgress) { progress = append(progress, p) })
	job, err = s.Run(ctx, job.ID, config, stages, nil)
	if err != nil || job.Status != Complete || asrRuns != 1 || len(job.Checkpoints) != 6 || !reflect.DeepEqual(prefix, job.Checkpoints[:3]) {
		t.Fatal(job, err, asrRuns)
	}
	r, err = s.OpenCheckpoint(context.Background(), job.ID, "diarization")
	if err != nil {
		t.Fatal(err)
	}
	d, err := ReadNemotronDiarizationJSON(context.Background(), r)
	r.Close()
	if err != nil {
		t.Fatal(err)
	}
	// Independently chunked native stream checks deterministic frontend/cache
	// handling. This is not an independent PyTorch/quality oracle.
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		t.Fatal(rate, err)
	}
	stream, err := model.NewCPUStream()
	if err != nil {
		t.Fatal(err)
	}
	var segments nem.SegmentStream
	expected := make([]nem.Segment, 0)
	for offset := 0; offset < len(pcm); offset += 7979 {
		logits, err := stream.AppendPCMContext(context.Background(), pcm[offset:min(offset+7979, len(pcm))])
		if err != nil {
			t.Fatal(err)
		}
		if len(logits) > 0 {
			part, err := segments.Append(logits)
			if err != nil {
				t.Fatal(err)
			}
			expected = append(expected, part...)
		}
	}
	logits, err := stream.FinishContext(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) > 0 {
		part, err := segments.Append(logits)
		if err != nil {
			t.Fatal(err)
		}
		expected = append(expected, part...)
	}
	part, err := segments.Finish()
	if err != nil {
		t.Fatal(err)
	}
	expected = append(expected, part...)
	if !reflect.DeepEqual(d.Turns, expected) {
		t.Fatalf("turns=%v differently-chunked=%v", d.Turns, expected)
	}
	r, err = s.OpenCheckpoint(context.Background(), job.ID, "speaker-transcript")
	if err != nil {
		t.Fatal(err)
	}
	labelled, err := ReadSpeakerTranscriptJSON(context.Background(), r)
	r.Close()
	if err != nil || !labelled.Experimental || labelled.Policy != nemotronSpeakerPolicy || labelled.LabelledWords < 1 {
		t.Fatal(labelled, err)
	}
	for i, w := range labelled.Transcript.Words {
		w.Speaker = -1
		if !reflect.DeepEqual(w, transcript.Words[i]) {
			t.Fatal("word timing/text changed")
		}
	}
	if labelled.Transcript.Language != transcript.Language {
		t.Fatal("language changed")
	}
	r, err = s.OpenCheckpoint(context.Background(), job.ID, "speaker-vtt")
	if output := readAll(t, r, err); !strings.Contains(output, "Nemotron") || !strings.Contains(output, "<v SPEAKER_00>") {
		t.Fatal(output)
	}
	if len(progress) != 4 || progress[0].Completed != 0 || progress[3].Completed != 176000 {
		t.Fatal(progress)
	}
	assertNoScratch(t, s, job.ID)
	t.Logf("CPU trained provider smoke: cancellation + fresh retry, retained 3 checkpoints, turns=%v, labelled_words=%d, experimental=%v, elapsed=%s", d.Turns, labelled.LabelledWords, labelled.Experimental, time.Since(start))
}
