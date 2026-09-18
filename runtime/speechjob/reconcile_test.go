//go:build linux && amd64

package speechjob

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/models/whisper"
)

func reconcileConfig() TranscriptStageConfig {
	return TranscriptStageConfig{ASRVersion: hash([]byte("asr version")), Language: "pt", WindowSamples: 320, OverlapSamples: 160}
}
func asrRecords(t *testing.T, total int64, key string, cfg TranscriptStageConfig, segments [][]whisper.Segment) []byte {
	t.Helper()
	plan, e := whisper.NewWindowPlan(total, cfg.WindowSamples, cfg.OverlapSamples)
	if e != nil {
		t.Fatal(e)
	}
	var b bytes.Buffer
	for i := int64(0); i < plan.Count(); i++ {
		w, _ := plan.At(i)
		var seg []whisper.Segment
		if int(i) < len(segments) {
			seg = segments[i]
		}
		record := windowRecord{Schema: 1, Key: key, Result: whisper.WindowTranscript{Window: w, Segments: seg}}
		row, e := json.Marshal(record)
		if e != nil {
			t.Fatal(e)
		}
		b.Write(row)
		b.WriteByte('\n')
	}
	return b.Bytes()
}
func TestReconcileCheckedWordTimings(t *testing.T) {
	ctx := context.Background()
	cfg := reconcileConfig()
	key := hash([]byte("word key"))
	segment := whisper.Segment{Start: .0125, End: .02, Text: "Olá mundo", Tokens: []int{42, 43}}
	words := []whisper.WordTiming{{Word: "Olá", Start: .0125, End: .015, TokenStart: 0, TokenEnd: 1}, {Word: "mundo", Start: .015, End: .02, TokenStart: 1, TokenEnd: 2}}
	raw := asrRecords(t, 480, key, cfg, [][]whisper.Segment{{segment}})
	lines := bytes.Split(bytes.TrimSpace(raw), []byte{'\n'})
	var record windowRecord
	if err := json.Unmarshal(lines[0], &record); err != nil {
		t.Fatal(err)
	}
	record.Result.Words = words
	lines[0], _ = json.Marshal(record)
	raw = append(bytes.Join(lines, []byte{'\n'}), '\n')
	got, err := reconcileASR(ctx, bytes.NewReader(raw), 480, key, cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := []WordCue{{StartSample: 200, EndSample: 240, Speaker: -1, Text: "Olá"}, {StartSample: 240, EndSample: 320, Speaker: -1, Text: "mundo"}}
	if len(got.Cues) != 1 || !reflect.DeepEqual(got.Words, want) {
		t.Fatalf("words=%+v", got.Words)
	}
}

func TestReconcileExactDuplicateAndDisjoint(t *testing.T) {
	ctx := context.Background()
	cfg := reconcileConfig()
	key := hash([]byte("key"))
	same := whisper.Segment{Start: .0125, End: .0175, Text: "Olá <tag>", Tokens: []int{42}}
	segments := [][]whisper.Segment{{{Start: 0, End: .005, Text: "primeiro", Tokens: []int{1}}, same}, {same, {Start: .025, End: .03, Text: "fim", Tokens: []int{2}}}}
	raw := asrRecords(t, 480, key, cfg, segments)
	got, e := reconcileASR(ctx, bytes.NewReader(raw), 480, key, cfg)
	if e != nil {
		t.Fatal(e)
	}
	want := []Cue{{StartSample: 0, EndSample: 80, Speaker: -1, Text: "primeiro"}, {StartSample: 200, EndSample: 280, Speaker: -1, Text: "Olá <tag>"}, {StartSample: 400, EndSample: 480, Speaker: -1, Text: "fim"}}
	if !reflect.DeepEqual(got.Cues, want) {
		t.Fatal(got.Cues)
	}
	var vtt bytes.Buffer
	if e = WriteWebVTT(ctx, &vtt, got); e != nil || !strings.Contains(vtt.String(), "Olá &lt;tag&gt;") {
		t.Fatal(vtt.String(), e)
	}
	// Identical words in separate time intervals are legitimate repetition.
	segments[1][1].Text = same.Text
	if got, e = reconcileASR(ctx, bytes.NewReader(asrRecords(t, 480, key, cfg, segments)), 480, key, cfg); e != nil || len(got.Cues) != 3 {
		t.Fatal(got, e)
	}
}
func TestReconcileConflictsNeverDropText(t *testing.T) {
	cfg := reconcileConfig()
	key := hash([]byte("key"))
	a := whisper.Segment{Start: .0125, End: .0175, Text: "same", Tokens: []int{42}}
	for _, kind := range []string{"text", "tokens", "time", "subsample-difference"} {
		b := a
		b.Tokens = append([]int{}, a.Tokens...)
		switch kind {
		case "text":
			b.Text = "different"
		case "tokens":
			b.Tokens = []int{43}
		case "time":
			b.Start = .015
			b.End = .025
		case "subsample-difference":
			b.Start = math.Nextafter(b.Start, math.Inf(1))
		}
		if _, e := reconcileASR(context.Background(), bytes.NewReader(asrRecords(t, 480, key, cfg, [][]whisper.Segment{{a}, {b}})), 480, key, cfg); !errors.Is(e, ErrTranscriptOverlap) {
			t.Fatal(kind, e)
		}
	}
	// Different ownership intervals do not legitimise throwing either text away.
	a.Start = .009
	a.End = .02
	b := whisper.Segment{Start: .01, End: .025, Text: "other", Tokens: []int{1}}
	if _, e := reconcileASR(context.Background(), bytes.NewReader(asrRecords(t, 480, key, cfg, [][]whisper.Segment{{a}, {b}})), 480, key, cfg); !errors.Is(e, ErrTranscriptOverlap) {
		t.Fatal(e)
	}
}
func TestReconcileRejectsBrokenStreams(t *testing.T) {
	cfg := reconcileConfig()
	key := hash([]byte("key"))
	good := asrRecords(t, 480, key, cfg, nil)
	lines := bytes.SplitAfter(good, []byte{'\n'})
	for _, kind := range []string{"missing", "extra", "order", "key", "version", "geometry", "trailing", "alias", "duplicate", "line-limit", "bad-text", "subsample"} {
		raw := append([]byte{}, good...)
		switch kind {
		case "missing":
			raw = lines[0]
		case "extra":
			raw = append(raw, lines[1]...)
		case "order":
			raw = append(append([]byte{}, lines[1]...), lines[0]...)
		case "key":
			raw = bytes.Replace(raw, []byte(key), []byte(hash([]byte("other"))), 1)
		case "version":
			raw = bytes.Replace(raw, []byte(`"schema":1`), []byte(`"schema":2`), 1)
		case "geometry":
			raw = bytes.Replace(raw, []byte(`"PadSamples":0`), []byte(`"PadSamples":1`), 1)
		case "trailing":
			raw = raw[:len(raw)-1]
		case "alias":
			raw = bytes.Replace(raw, []byte(`"schema":1`), []byte(`"Schema":1`), 1)
		case "duplicate":
			raw = bytes.Replace(raw, []byte(`"schema":1`), []byte(`"schema":1,"schema":1`), 1)
		case "line-limit":
			raw = append(bytes.Repeat([]byte{' '}, 1<<20), '\n')
		case "bad-text":
			raw = asrRecords(t, 480, key, cfg, [][]whisper.Segment{{{Start: 0, End: .001, Text: "\x00"}}})
		case "subsample":
			raw = asrRecords(t, 480, key, cfg, [][]whisper.Segment{{{Start: .00001, End: .001, Text: "x"}}})
		}
		if _, e := reconcileASR(context.Background(), bytes.NewReader(raw), 480, key, cfg); e == nil {
			t.Fatal("accepted", kind)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := reconcileASR(ctx, bytes.NewReader(good), 480, key, cfg); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
}
func TestReconcileSampleGridAndEmptyLongPlan(t *testing.T) {
	for _, sample := range []int64{0, 1, 479999, 4 * 3600 * 16000} {
		got, e := sampleTimestamp(float64(sample)/16000, 4*3600*16000)
		if e != nil || got != sample {
			t.Fatal(sample, got, e)
		}
	}
	for _, sec := range []float64{-1, math.NaN(), math.Inf(1), 1.0 / 32000} {
		if _, e := sampleTimestamp(sec, 16000); !errors.Is(e, ErrTranscriptSampleGrid) {
			t.Fatal(sec, e)
		}
	}
	cfg := reconcileConfig()
	key := hash([]byte("key"))
	total := int64(320 + 130*160)
	tr, e := reconcileASR(context.Background(), bytes.NewReader(asrRecords(t, total, key, cfg, nil)), total, key, cfg)
	if e != nil || len(tr.Cues) != 0 || tr.TotalSamples != total {
		t.Fatal(tr, e)
	}
	if _, e = reconcileASR(context.Background(), strings.NewReader(""), 0, key, cfg); e != nil {
		t.Fatal(e)
	}
}
func timedPCMStage(samples int64, source media.SourceTiming) Stage {
	return stage("decode", func(ctx context.Context, _ *Input, w io.Writer) error {
		plain := testWAV(make([]int16, samples), "")
		chunk, err := media.MarshalSourceTimingWAVChunk(source)
		if err != nil {
			return err
		}
		if len(chunk) == 0 {
			return writeContext(ctx, w, plain)
		}
		data := append(append(append([]byte{}, plain[:36]...), chunk...), plain[36:]...)
		binary.LittleEndian.PutUint32(data[4:8], uint32(len(data)-8))
		return writeContext(ctx, w, data)
	})
}

func textASRStage(t *testing.T, total int64, cfg TranscriptStageConfig, conflict bool, calls *int) Stage {
	return Stage{Name: "asr-windows", Version: cfg.ASRVersion, Run: func(ctx context.Context, in *Input, w io.Writer) error {
		*calls++
		a := whisper.Segment{Start: .0125, End: .0175, Text: "Olá <&>", Tokens: []int{42}}
		b := a
		if conflict {
			b.Text = "conflito"
		}
		key := checkpointKey(in.job, Stage{Name: "asr-windows", Version: cfg.ASRVersion})
		return writeContext(ctx, w, asrRecords(t, total, key, cfg, [][]whisper.Segment{{a}, {b}}))
	}}
}
func TestTranscriptStageRetentionAndIdentity(t *testing.T) {
	cfg := reconcileConfig()
	tr, e := NewTranscriptStage(cfg)
	if e != nil {
		t.Fatal(e)
	}
	s, dir := openTest(t)
	job := createTest(t, s)
	calls := 0
	late := stage("late", func(context.Context, *Input, io.Writer) error { return io.ErrClosedPipe })
	stages := []Stage{fixturePCMStage(480), textASRStage(t, 480, cfg, false, &calls), tr, NewVTTStage(), late}
	job, e = s.Run(context.Background(), job.ID, config, stages, nil)
	if !errors.Is(e, io.ErrClosedPipe) || len(job.Checkpoints) != 4 {
		t.Fatal(job, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), job.ID, "vtt")
	before := readAll(t, r, e)
	if !strings.Contains(before, "Olá &lt;&amp;&gt;") {
		t.Fatal(before)
	}
	s.Close()
	s, e = Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	late.Run = func(context.Context, *Input, io.Writer) error { return nil }
	stages[4] = late
	job, e = s.Run(context.Background(), job.ID, config, stages, nil)
	if e != nil || calls != 1 || job.Status != Complete {
		t.Fatal(job, calls, e)
	}
	r, e = s.OpenCheckpoint(context.Background(), job.ID, "vtt")
	if readAll(t, r, e) != before {
		t.Fatal("changed reuse")
	}
	changed := cfg
	changed.Language = "fr"
	stages[2], e = NewTranscriptStage(changed)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = s.Run(context.Background(), job.ID, config, stages, nil); !errors.Is(e, ErrConfiguration) {
		t.Fatal(e)
	}
}
func TestTranscriptStagePersistsSourceTimingWithoutChangingCanonicalCues(t *testing.T) {
	cfg := reconcileConfig()
	tr, err := NewTranscriptStage(cfg)
	if err != nil {
		t.Fatal(err)
	}
	source := media.SourceTiming{Start: 250 * time.Millisecond, Duration: time.Second, HasEdits: true, SourceRate: 48000, Priming: 1024, Padding: 512, LeadingSilence: 240}
	s, _ := openTest(t)
	job := createTest(t, s)
	calls := 0
	job, err = s.Run(context.Background(), job.ID, config, []Stage{timedPCMStage(480, source), textASRStage(t, 480, cfg, false, &calls), tr}, nil)
	if err != nil || job.Status != Complete {
		t.Fatal(job, err)
	}
	r, err := s.OpenCheckpoint(context.Background(), job.ID, "transcript")
	got, err := ReadTranscriptJSON(context.Background(), r)
	if closeErr := r.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	if got.SourceTiming != source || got.TotalSamples != 480 || got.Cues[0].StartSample != 200 || got.Cues[0].EndSample != 280 {
		t.Fatalf("source/canonical mapping changed: %+v", got)
	}
}

func TestTranscriptStageConflictPreservesRaw(t *testing.T) {
	cfg := reconcileConfig()
	tr, _ := NewTranscriptStage(cfg)
	s, _ := openTest(t)
	job := createTest(t, s)
	calls := 0
	job, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(480), textASRStage(t, 480, cfg, true, &calls), tr}, nil)
	if !errors.Is(e, ErrTranscriptOverlap) || len(job.Checkpoints) != 2 {
		t.Fatal(job, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), job.ID, "asr-windows")
	if !strings.Contains(readAll(t, r, e), "conflito") {
		t.Fatal("raw lost")
	}
	if _, e = s.OpenCheckpoint(context.Background(), job.ID, "transcript"); e == nil {
		t.Fatal("conflict published")
	}
}
