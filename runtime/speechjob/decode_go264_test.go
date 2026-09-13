package speechjob

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

func go264DecodeConfig(extension string) Go264DecodeConfig {
	return Go264DecodeConfig{InputExtension: extension, MaxInputBytes: 1 << 20, MaxOutputBytes: 90000, MaxDuration: time.Second}
}

func TestGo264DecodeStageIdentityAndValidation(t *testing.T) {
	cfg := go264DecodeConfig(".wav")
	stage, err := NewGo264DecodeStage(cfg)
	if err != nil || stage.Name != "decode" || !validHash(stage.Version) {
		t.Fatal(stage, err)
	}
	ff := FFmpegDecodeConfig{FFmpegPath: "/fixture/ffmpeg", FFprobePath: "/fixture/ffprobe", FFmpegSHA256: hash([]byte("ffmpeg")), FFprobeSHA256: hash([]byte("ffprobe")), InputExtension: cfg.InputExtension, MaxInputBytes: cfg.MaxInputBytes, MaxOutputBytes: cfg.MaxOutputBytes, MaxDuration: cfg.MaxDuration}
	adapter := &decodeFixtureAdapter{}
	if stage.Version == newDecodeStage(ff, adapter).Version {
		t.Fatal("provider identity collision")
	}
	for _, kind := range []string{"extension", "input", "output", "duration", "large-duration"} {
		bad := cfg
		switch kind {
		case "extension":
			bad.InputExtension = ".mp3"
		case "input":
			bad.MaxInputBytes = 0
		case "output":
			bad.MaxOutputBytes = 1
		case "duration":
			bad.MaxDuration = time.Nanosecond
		case "large-duration":
			bad.MaxDuration = media.DefaultMaxDuration + time.Second
		}
		if _, err := NewGo264DecodeStage(bad); err == nil {
			t.Fatal("accepted", kind)
		}
	}
}

func TestGo264DecodeStageProviderIdentityPinned(t *testing.T) {
	if go264DecodeProvider != "github.com/rcarmo/go-264@v0.0.0-20260913161458-9ed3d408e05d" {
		t.Fatal(go264DecodeProvider)
	}
}

func TestGo264DecodeStageCanonicalAndResume(t *testing.T) {
	s, dir := openTest(t)
	cfg := go264DecodeConfig(".wav")
	stage, err := NewGo264DecodeStage(cfg)
	if err != nil {
		t.Fatal(err)
	}
	samples := []int16{-32768, -123, 0, 1, 32767}
	source := testWAV(samples, "")
	job, err := s.Create(context.Background(), "source.wav", config, bytes.NewReader(source))
	if err != nil {
		t.Fatal(err)
	}
	later := stageFixture("later", func(context.Context, *Input, io.Writer) error { return errors.New("later failure") })
	job, err = s.Run(context.Background(), job.ID, config, []Stage{stage, later}, nil)
	if err == nil || len(job.Checkpoints) != 1 || job.Checkpoints[0].Stage != "decode" {
		t.Fatal(job, err)
	}
	checkpoint := filepath.Join(dir, job.ID, job.Checkpoints[0].Blob.File)
	pcm, err := media.OpenCanonicalPCM(context.Background(), checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(samples))
	if n, err := pcm.ReadSamplesAt(context.Background(), got, 0); n != len(got) || err != nil {
		t.Fatal(n, err)
	}
	if err := pcm.Close(); err != nil {
		t.Fatal(err)
	}
	for i, want := range samples {
		if got[i] != float32(want)/32768 {
			t.Fatal("PCM mismatch", i, got[i], want)
		}
	}
	first := job.Checkpoints[0].Blob.SHA256
	later.Run = func(_ context.Context, _ *Input, out io.Writer) error { _, err := out.Write([]byte("ok")); return err }
	job, err = s.Run(context.Background(), job.ID, config, []Stage{stage, later}, nil)
	if err != nil || job.Status != Complete || job.Checkpoints[0].Blob.SHA256 != first {
		t.Fatal(job.Status, err)
	}
	assertNoScratch(t, s, job.ID)
}

func TestGo264JobAACIntegration(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_FFMPEG") != "1" {
		t.Skip("FFmpeg is used only to generate the synthetic AAC fixture")
	}
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	const frames = 16000
	samples := make([]int16, frames)
	for i := range samples {
		samples[i] = int16(4000 * math.Sin(float64(i)*2*math.Pi*440/16000))
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "tone.wav")
	if err := os.WriteFile(source, testWAV(samples, ""), 0600); err != nil {
		t.Fatal(err)
	}
	encoded := filepath.Join(dir, "tone.m4a")
	cmd := exec.Command(ffmpeg, "-nostdin", "-v", "error", "-threads", "1", "-i", source, "-ar", "48000", "-c:a", "aac", "-b:a", "96k", "-movie_timescale", "48000", "-y", encoded)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatal(err, string(output))
	}
	input, err := os.ReadFile(encoded)
	if err != nil {
		t.Fatal(err)
	}
	store, root := openTest(t)
	cfg := Go264DecodeConfig{InputExtension: ".m4a", MaxInputBytes: 1 << 20, MaxOutputBytes: 129000, MaxDuration: 2 * time.Second}
	stage, err := NewGo264DecodeStage(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var previous string
	for i := 0; i < 2; i++ {
		job, err := store.Create(context.Background(), "tone.m4a", config, bytes.NewReader(input))
		if err != nil {
			t.Fatal(err)
		}
		job, err = store.Run(context.Background(), job.ID, config, []Stage{stage}, nil)
		if err != nil || job.Status != Complete {
			t.Fatal(job.Status, err)
		}
		reader, err := store.OpenCheckpoint(context.Background(), job.ID, "decode")
		decoded := readAll(t, reader, err)
		if i > 0 && decoded != previous {
			t.Fatal("pure-Go AAC checkpoint not deterministic")
		}
		previous = decoded
		pcm, err := media.OpenCanonicalPCM(context.Background(), filepath.Join(root, job.ID, job.Checkpoints[0].Blob.File))
		if err != nil {
			t.Fatal(err)
		}
		timeline, timing := pcm.Timeline(), pcm.SourceTiming()
		if err := pcm.Close(); err != nil {
			t.Fatal(err)
		}
		if timeline.Samples != frames || timing.SourceRate != 48000 || !timing.Exact || timing.Priming != 1024 || timing.Padding != 128 {
			t.Fatal("pure-Go AAC timing", timeline, timing)
		}
		assertNoScratch(t, store, job.ID)
	}
}

func TestGo264DecodeStageCancellationCleanup(t *testing.T) {
	s, _ := openTest(t)
	cfg := go264DecodeConfig(".wav")
	stage, err := NewGo264DecodeStage(cfg)
	if err != nil {
		t.Fatal(err)
	}
	const frames = 8000
	wav := make([]byte, 44+2*frames)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], uint32(len(wav)-8))
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 1)
	binary.LittleEndian.PutUint32(wav[24:], 16000)
	binary.LittleEndian.PutUint32(wav[28:], 32000)
	binary.LittleEndian.PutUint16(wav[32:], 2)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 2*frames)
	job, err := s.Create(context.Background(), "source.wav", config, bytes.NewReader(wav))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	job, err = s.Run(ctx, job.ID, config, []Stage{stage}, nil)
	if !errors.Is(err, context.Canceled) || len(job.Checkpoints) != 0 {
		t.Fatal(job, err)
	}
	assertNoScratch(t, s, job.ID)
}

func stageFixture(name string, run func(context.Context, *Input, io.Writer) error) Stage {
	return Stage{Name: name, Version: hash([]byte(name + "-fixture")), Run: run}
}
