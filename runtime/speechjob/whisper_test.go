//go:build linux && amd64

package speechjob

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/models/whisper"
)

func fixturePCMStage(samples int) Stage {
	return stage("decode", func(ctx context.Context, _ *Input, w io.Writer) error {
		return writeContext(ctx, w, testWAV(make([]int16, samples), ""))
	})
}
func fixtureWindows(firstSeen *[]int64, failAt int64) windowInfer {
	return func(ctx context.Context, source whisper.SampleReader, total, first int64, emit func(whisper.WindowTranscript) error) error {
		*firstSeen = append(*firstSeen, first)
		plan, e := whisper.NewWindowPlan(total, 320, 160)
		if e != nil {
			return e
		}
		scratch := make([]float32, 320)
		for i := first; i < plan.Count(); i++ {
			if i == failAt {
				return io.ErrUnexpectedEOF
			}
			w, e := plan.ReadWindow(ctx, source, i, scratch)
			if e != nil {
				return e
			}
			end := min(w.End, w.Start+80)
			if e = emit(whisper.WindowTranscript{Window: w, Segments: []whisper.Segment{{Start: float64(w.Start) / 16000, End: float64(end) / 16000, Text: fmt.Sprintf("window%d", i), Tokens: []int{42}}}}); e != nil {
				return e
			}
		}
		return nil
	}
}
func fixtureWhisperStage(firsts *[]int64, failAt int64) Stage {
	return whisperWindowStage(hash([]byte("fixture-whisper-v1")), 320, 160, 4096, 128<<10, 448, 51865, fixtureWindows(firsts, failAt))
}

func TestWhisperJournalResumeAcrossReopen(t *testing.T) {
	s, dir := openTest(t)
	job := createTest(t, s)
	var starts []int64
	job, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, 2)}, nil)
	if !errors.Is(e, io.ErrUnexpectedEOF) || job.Status != Failed || len(job.Checkpoints) != 1 {
		t.Fatal(job, e)
	}
	if e = s.Close(); e != nil {
		t.Fatal(e)
	}
	s, e = Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	later := stage("later", func(context.Context, *Input, io.Writer) error { return io.ErrClosedPipe })
	job, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, -1), later}, nil)
	if !errors.Is(e, io.ErrClosedPipe) || len(job.Checkpoints) != 2 || fmt.Sprint(starts) != "[0 2]" {
		t.Fatal(job, starts, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), job.ID, "asr-windows")
	raw := readAll(t, r, e)
	rows := strings.Split(strings.TrimSpace(raw), "\n")
	plan, _ := whisper.NewWindowPlan(1001, 320, 160)
	if len(rows) != int(plan.Count()) {
		t.Fatal("incomplete windows", len(rows))
	}
	for i, row := range rows {
		var record windowRecord
		if e = json.Unmarshal([]byte(row), &record); e != nil {
			t.Fatal(e)
		}
		expected, _ := plan.At(int64(i))
		if record.Result.Window != expected || record.Result.Segments[0].Text != fmt.Sprintf("window%d", i) {
			t.Fatal(record)
		}
	}
	later.Run = func(context.Context, *Input, io.Writer) error { return nil }
	job, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, -1), later}, nil)
	if e != nil || job.Status != Complete || len(starts) != 2 {
		t.Fatal("repeated ASR", job, starts, e)
	}
	assertNoScratch(t, s, job.ID)
	if e = s.Delete(context.Background(), job.ID); e != nil {
		t.Fatal(e)
	}
}

func TestWhisperJournalRejectsCorruptionAndGaps(t *testing.T) {
	for _, kind := range []string{"payload", "ack", "gap", "ack-geometry"} {
		t.Run(kind, func(t *testing.T) {
			s, _ := openTest(t)
			job := createTest(t, s)
			var starts []int64
			st := fixtureWhisperStage(&starts, 2)
			job, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), st}, nil)
			if e == nil {
				t.Fatal("fixture")
			}
			key := checkpointKey(job, st)
			base := filepath.Join(s.root.Name(), job.ID, windowBase(key, 0))
			switch kind {
			case "payload":
				if e = os.WriteFile(base, []byte("corrupt"), 0600); e != nil {
					t.Fatal(e)
				}
			case "ack":
				if e = os.WriteFile(base+".ack", []byte("{}"), 0600); e != nil {
					t.Fatal(e)
				}
			case "gap":
				if e = os.Remove(base + ".ack"); e != nil {
					t.Fatal(e)
				}
			case "ack-geometry":
				b, e := os.ReadFile(base + ".ack")
				if e != nil {
					t.Fatal(e)
				}
				var a windowAck
				json.Unmarshal(b, &a)
				a.Index = 1
				b, _ = json.Marshal(a)
				if e = os.WriteFile(base+".ack", b, 0600); e != nil {
					t.Fatal(e)
				}
			}
			before := len(starts)
			_, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, -1)}, nil)
			if !errors.Is(e, ErrCorrupt) || len(starts) != before {
				t.Fatal("corrupt resumed", kind, starts, e)
			}
		})
	}
}

func TestWhisperWindowPublicationRetryAndConflict(t *testing.T) {
	for _, point := range []string{"window-payload-published", "window-ack-published"} {
		t.Run(point, func(t *testing.T) {
			s, _ := openTest(t)
			job := createTest(t, s)
			var starts []int64
			failure := errors.New("publication barrier")
			s.fault = func(p string) error {
				if p == point {
					return failure
				}
				return nil
			}
			_, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, -1)}, nil)
			if !errors.Is(e, failure) {
				t.Fatal(e)
			}
			s.fault = nil
			job, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, -1)}, nil)
			want := int64(0)
			if point == "window-ack-published" {
				want = 1
			}
			if e != nil || job.Status != Complete || len(starts) != 2 || starts[1] != want {
				t.Fatal(job, starts, e)
			}
		})
	}
	s, _ := openTest(t)
	job := createTest(t, s)
	var starts []int64
	s.fault = func(p string) error {
		if p == "window-payload-published" {
			return io.ErrClosedPipe
		}
		return nil
	}
	if _, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, -1)}, nil); e == nil {
		t.Fatal("fixture")
	}
	s.fault = nil
	original := fixtureWindows(&starts, -1)
	infer := func(ctx context.Context, r whisper.SampleReader, total, first int64, emit func(whisper.WindowTranscript) error) error {
		return original(ctx, r, total, first, func(w whisper.WindowTranscript) error {
			w.Segments[0].Text = "changed under same identity"
			return emit(w)
		})
	}
	st := whisperWindowStage(hash([]byte("fixture-whisper-v1")), 320, 160, 4096, 128<<10, 448, 51865, infer)
	if _, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), st}, nil); !errors.Is(e, ErrCorrupt) {
		t.Fatal("orphan overwritten", e)
	}
}

func TestWhisperJournalAdmissionValidationCancel(t *testing.T) {
	for _, kind := range []string{"quota", "window-limit", "result-limit", "bad-order", "bad-geometry", "bad-time", "bad-token", "incomplete", "cancel"} {
		t.Run(kind, func(t *testing.T) {
			s, _ := openTest(t)
			job := createTest(t, s)
			var starts []int64
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			original := fixtureWindows(&starts, -1)
			infer := func(ctx context.Context, r whisper.SampleReader, total, first int64, emit func(whisper.WindowTranscript) error) error {
				if kind == "incomplete" {
					return nil
				}
				return original(ctx, r, total, first, func(w whisper.WindowTranscript) error {
					switch kind {
					case "bad-order":
						w.Window.Index++
					case "bad-geometry":
						w.Window.EmitEnd++
					case "bad-time":
						w.Segments[0].End = 999
					case "bad-token":
						w.Segments[0].Tokens[0] = -1
					case "cancel":
						cancel()
					}
					return emit(w)
				})
			}
			if kind == "quota" {
				s.limits.MaxBytes = 300000
			}
			windowLimit, resultLimit := int64(4096), int64(128<<10)
			if kind == "window-limit" {
				windowLimit = 1
			}
			if kind == "result-limit" {
				resultLimit = 300
			}
			st := whisperWindowStage(hash([]byte("v1")), 320, 160, windowLimit, resultLimit, 448, 51865, infer)
			job, e := s.Run(ctx, job.ID, config, []Stage{fixturePCMStage(1001), st}, nil)
			if e == nil || job.Status == Complete || len(job.Checkpoints) != 1 {
				t.Fatal(kind, job, e)
			}
			if kind == "quota" && len(starts) > 0 {
				t.Fatal("inferred without admission")
			}
			if kind == "cancel" && !errors.Is(e, context.Canceled) {
				t.Fatal(e)
			}
			assertNoScratch(t, s, job.ID)
		})
	}
}

// Public API binding to actual checked Go frontend/encoder/decoder, with a
// zero-layer two-channel model which emits EOT. No trained weights or quality.
func jobToyWhisper() (*whisper.Whisper, *whisper.Tokenizer) {
	c := whisper.Config{NumMelBins: 80, MaxLength: 2, EncoderDModel: 2, DecoderDModel: 2, EncoderHeads: 1, DecoderHeads: 1, EncoderFFNDim: 2, DecoderFFNDim: 2, HeadDim: 2, VocabSize: 51865, MaxDecoderLength: 8}
	e, d := whisper.NewEncoder(c), whisper.NewDecoder(c)
	e.Conv1Weight = make([]float32, 2*80*3)
	e.Conv1Bias = make([]float32, 2)
	e.Conv2Weight = make([]float32, 2*2*3)
	e.Conv2Bias = make([]float32, 2)
	e.FinalLNWeight = []float32{1, 1}
	e.FinalLNBias = make([]float32, 2)
	d.TokenEmbed = make([]float32, c.VocabSize*2)
	d.TokenEmbed[50257*2] = 10
	d.TokenEmbed[50257*2+1] = -10
	d.PosEmbed = make([]float32, c.MaxDecoderLength*2)
	for i := 0; i < c.MaxDecoderLength; i++ {
		d.PosEmbed[2*i] = 1
		d.PosEmbed[2*i+1] = -1
	}
	d.FinalLNWeight = []float32{1, 1}
	d.FinalLNBias = make([]float32, 2)
	vocab := make(map[int]string, c.VocabSize)
	for i := 0; i < c.VocabSize; i++ {
		vocab[i] = fmt.Sprintf("text%d", i)
	}
	for id, text := range map[int]string{50257: "<|endoftext|>", 50258: "<|startoftranscript|>", 50259: "<|en|>", 50267: "<|pt|>", 50358: "<|translate|>", 50359: "<|transcribe|>", 50363: "<|notimestamps|>"} {
		vocab[id] = text
	}
	for i := 0; i <= 1500; i++ {
		vocab[50364+i] = fmt.Sprintf("<|%.2f|>", float64(i)/50)
	}
	return &whisper.Whisper{Encoder: e, Decoder: d, Config: c}, &whisper.Tokenizer{Vocab: vocab, VocabSize: c.VocabSize}
}
func TestWhisperJobActualToyModel(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_GRAPH", "0")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_SELF_ATTN", "0")
	model, tok := jobToyWhisper()
	cfg := WhisperStageConfig{ModelSHA256: hash([]byte("toy")), RuntimeSHA256: hash([]byte("host")), Language: "pt", MaxWindowBytes: 4096, MaxResultBytes: 128 << 10}
	st, e := NewWhisperWindowStage(model, tok, cfg)
	if e != nil {
		t.Fatal(e)
	}
	s, _ := openTest(t)
	job := createTest(t, s)
	job, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(321), st}, nil)
	if e != nil || job.Status != Complete {
		t.Fatal(job, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), job.ID, "asr-windows")
	raw := readAll(t, r, e)
	if bytes.Count([]byte(raw), []byte{'\n'}) != 2 {
		t.Fatal(raw)
	}
	for _, row := range strings.Split(strings.TrimSpace(raw), "\n") {
		var record windowRecord
		if e = json.Unmarshal([]byte(row), &record); e != nil || len(record.Result.Segments) != 0 {
			t.Fatal(record, e)
		}
	}
	cfg.SkipDigitalSilence = true
	different, e := NewWhisperWindowStage(model, tok, cfg)
	if e != nil || different.Version == st.Version {
		t.Fatal("identity unchanged", e)
	}
	if _, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(321), different}, nil); !errors.Is(e, ErrConfiguration) {
		t.Fatal("changed options reused", e)
	}
}

func TestWhisperJournalMoreThanHundredWindows(t *testing.T) {
	s, _ := openTest(t)
	var starts []int64
	st := fixtureWhisperStage(&starts, -1)
	var prior string
	for run := 0; run < 2; run++ {
		job := createTest(t, s)
		job, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(320 + 130*160), st}, nil)
		if e != nil {
			t.Fatal(e)
		}
		r, e := s.OpenCheckpoint(context.Background(), job.ID, "asr-windows")
		raw := readAll(t, r, e)
		if len(strings.Split(strings.TrimSpace(raw), "\n")) != 131 {
			t.Fatal("window cutoff")
		}
		if run > 0 && raw != prior {
			t.Fatal("same identity/output changed")
		}
		prior = raw
	}
}

func TestWhisperJournalIgnoredCallbackErrorFailsClosed(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	infer := func(ctx context.Context, r whisper.SampleReader, total, first int64, emit func(whisper.WindowTranscript) error) error {
		_ = emit(whisper.WindowTranscript{})
		var starts []int64
		_ = fixtureWindows(&starts, -1)(ctx, r, total, first, emit)
		return nil
	}
	st := whisperWindowStage(hash([]byte("ignored-error")), 320, 160, 4096, 128<<10, 448, 51865, infer)
	job, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), st}, nil)
	if e == nil || job.Status != Failed || len(job.Checkpoints) != 1 {
		t.Fatal(job, e)
	}
}

func TestWhisperConstructorBounds(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_GRAPH", "0")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_SELF_ATTN", "0")
	model, tok := jobToyWhisper()
	good := WhisperStageConfig{ModelSHA256: hash([]byte("toy")), RuntimeSHA256: hash([]byte("host")), Language: "pt", MaxWindowBytes: 4096, MaxResultBytes: 128 << 10}
	for _, kind := range []string{"hash", "window-limit", "result-limit", "overlap", "language", "tokens", "timestamp", "generation"} {
		c := good
		switch kind {
		case "hash":
			c.ModelSHA256 = "bad"
		case "window-limit":
			c.MaxWindowBytes = 0
		case "result-limit":
			c.MaxResultBytes = 65 << 20
		case "overlap":
			c.OverlapSamples = 161
		case "language":
			c.Language = ""
		case "tokens":
			c.MaxNewTokens = 99
		case "timestamp":
			c.MaxInitialTimestampIndex = -1
		case "generation":
			c.GenerationJSON = []byte(`{"unknown":true}`)
		}
		if _, e := NewWhisperWindowStage(model, tok, c); e == nil {
			t.Fatal("bad config", kind)
		}
	}
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "0")
	if _, e := NewWhisperWindowStage(model, tok, good); e == nil {
		t.Fatal("unrestricted GPU accepted")
	}
}
