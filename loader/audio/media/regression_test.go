package media

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSampleArithmeticNoIntermediateOverflow(t *testing.T) {
	for _, rate := range []SampleRate{16000, 44100, 48000, 192000, 768000} {
		samples, err := durationToSamples(4*time.Hour, rate)
		if err != nil || samples != int64(rate)*14400 {
			t.Fatalf("rate %d: %d, %v", rate, samples, err)
		}
		if got := (Timeline{rate, SampleCount(samples)}).Duration(); got != 4*time.Hour {
			t.Fatalf("duration %s", got)
		}
	}
	if got := (Timeline{1, SampleCount(maxInt64)}).Duration(); got != time.Duration(maxInt64) {
		t.Fatalf("did not saturate: %v", got)
	}
	if got := maxFramesForDuration(4 * time.Hour); got != 230400000 {
		t.Fatalf("frames=%d", got)
	}
	if _, err := parsePositiveDuration("9223372036.854776"); err == nil {
		t.Fatal("accepted duration at overflowing float boundary")
	}
}

func TestInvalidLimitsAndProbeMetadata(t *testing.T) {
	for _, cfg := range []Config{{MaxDuration: -1}, {MaxDuration: 5 * time.Hour}, {MaxInputBytes: -1}, {MaxProbeStdoutBytes: 5 << 20}} {
		cfg.FFmpegPath = "ffmpeg"
		cfg.FFprobePath = "ffprobe"
		if _, err := NewFFmpeg(cfg); err == nil {
			t.Fatal("invalid limits accepted")
		}
	}
	for _, stream := range []string{
		`{"codec_type":"audio","codec_name":"aac","duration":"1","sample_rate":"16000","channels":1}`,
		`{"index":-1,"codec_type":"audio","codec_name":"aac","duration":"1","sample_rate":"16000","channels":1}`,
		`{"index":0,"codec_type":"audio","codec_name":"aac","duration":"NaN","sample_rate":"16000","channels":1}`,
		`{"index":0,"codec_type":"audio","codec_name":"aac","duration":"1","sample_rate":"99999999","channels":1}`,
		`{"index":0,"codec_type":"audio","codec_name":"aac","duration":"1","sample_rate":"16000","channels":100}`,
	} {
		if _, err := parseProbeJSON([]byte(`{"streams":[`+stream+`],"format":{"duration":"1"}}`), "x", "mov", 1, time.Hour); err == nil {
			t.Fatalf("accepted malformed metadata %s", stream)
		}
	}
}

func TestProbeOutputBoundAndPrecancel(t *testing.T) {
	src := filepath.Join(t.TempDir(), "a.wav")
	writeSyntheticWAV(t, src, 16000, 16)
	r := &fakeRunner{handlers: []func(context.Context, Command) error{func(_ context.Context, c Command) error {
		_, err := io.WriteString(c.Stdout, strings.Repeat("x", 1024))
		return err
	}}}
	ff := newTestFFmpeg(t, r, Config{MaxProbeStdoutBytes: 32})
	if _, err := ff.Probe(context.Background(), src); !errors.Is(err, ErrInvalidSource) {
		t.Fatalf("err=%v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ff.Probe(ctx, src); !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
	if len(r.Calls()) != 1 {
		t.Fatal("cancelled probe started runner")
	}
}

func TestCanonicalWAVRejectsRiffMismatchAndDuplicateData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.wav")
	writeCanonicalWAV(t, path, 8)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"riff-short", "riff-long", "data-extra", "partial-header", "rate"} {
		t.Run(kind, func(t *testing.T) {
			b := append([]byte(nil), data...)
			switch kind {
			case "riff-short":
				binary.LittleEndian.PutUint32(b[4:8], 12)
			case "riff-long":
				binary.LittleEndian.PutUint32(b[4:8], uint32(len(b)+100))
			case "data-extra":
				b = append(b, []byte("data\x00\x00\x00\x00")...)
				binary.LittleEndian.PutUint32(b[4:8], uint32(len(b)-8))
			case "partial-header":
				b = append(b, 1)
				binary.LittleEndian.PutUint32(b[4:8], uint32(len(b)-8))
			case "rate":
				binary.LittleEndian.PutUint32(b[24:28], 48000)
			}
			if err := os.WriteFile(path, b, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := validateCanonicalWAV(path, 1<<20, 16000); !errors.Is(err, ErrInvalidOutput) {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

func TestDecodePublicationRacePreservesDestination(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "in.wav")
	dst := filepath.Join(dir, "out.wav")
	writeSyntheticWAV(t, src, 16000, 16)
	r := &fakeRunner{handlers: []func(context.Context, Command) error{
		func(_ context.Context, c Command) error {
			return writeProbeJSON(t, c, `{"streams":[{"index":0,"codec_type":"audio","codec_name":"pcm_s16le","duration":"1","sample_rate":"16000","channels":1}]}`)
		},
		func(_ context.Context, c Command) error {
			writeCanonicalWAV(t, c.Args[len(c.Args)-1], 160)
			return os.WriteFile(dst, []byte("keep"), 0600)
		},
	}}
	ff := newTestFFmpeg(t, r, Config{})
	if _, err := ff.DecodeToFile(context.Background(), src, dst); err == nil {
		t.Fatal("overwrote racing destination")
	}
	b, _ := os.ReadFile(dst)
	if string(b) != "keep" {
		t.Fatal("destination corrupted")
	}
	remains, _ := filepath.Glob(filepath.Join(dir, ".out.wav.tmp-*"))
	if len(remains) != 0 {
		t.Fatal("partial output remains")
	}
	if _, err := os.Stat(src); err != nil {
		t.Fatal("source removed")
	}
}

func TestDecodeOutputMonitorCancelsOwnedRunner(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "in.wav")
	writeSyntheticWAV(t, src, 16000, 16)
	r := &fakeRunner{handlers: []func(context.Context, Command) error{
		func(_ context.Context, c Command) error {
			return writeProbeJSON(t, c, `{"streams":[{"index":0,"codec_type":"audio","codec_name":"pcm_s16le","duration":"1","sample_rate":"16000","channels":1}]}`)
		},
		func(ctx context.Context, c Command) error {
			writeCanonicalWAV(t, c.Args[len(c.Args)-1], 1000)
			<-ctx.Done()
			return ctx.Err()
		},
	}}
	ff := newTestFFmpeg(t, r, Config{MaxDecodeOutputBytes: 128})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := ff.DecodeToFile(ctx, src, filepath.Join(dir, "out.wav")); !errors.Is(err, ErrDecodeOutputLimit) {
		t.Fatalf("err=%v", err)
	}
}

// Explicit opt-in: tiny public synthetic fixtures, no service/model/GPU needed.
func TestFFmpegIntegration(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_FFMPEG") != "1" {
		t.Skip("set GO_PHERENCE_TEST_FFMPEG=1 for synthetic FFmpeg integration")
	}
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Fatal(err)
	}
	ff, err := NewFFmpeg(Config{FFmpegPath: ffmpeg, FFprobePath: ffprobe})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	for _, rate := range []int{44100, 48000} {
		dir := t.TempDir()
		src := filepath.Join(dir, "-input with spaces.wav")
		writeSyntheticWAV(t, src, rate, rate/4)
		result, err := ff.DecodeToFile(ctx, src, filepath.Join(dir, "out.wav"))
		if err != nil {
			t.Fatal(err)
		}
		if result.Timeline.Samples != 4000 {
			t.Fatalf("rate %d: frames=%d", rate, result.Timeline.Samples)
		}
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "input.wav")
	writeSyntheticWAV(t, src, 48000, 12000)
	m4a := filepath.Join(dir, "-encoded test.m4a")
	cmd := exec.CommandContext(ctx, ffmpeg, "-nostdin", "-v", "error", "-threads", "1", "-i", src, "-c:a", "aac", "-threads", "1", m4a)
	if err := cmd.Run(); err != nil {
		t.Fatal("synthetic AAC creation failed:", err)
	}
	result, err := ff.DecodeToFile(ctx, m4a, filepath.Join(dir, "decoded.wav"))
	if err != nil {
		t.Fatal(err)
	}
	if result.Timeline.Samples < 4000 || result.Timeline.Samples > 4500 {
		t.Fatalf("unexpected AAC padded frames %d", result.Timeline.Samples)
	}
	// Output frame count is authoritative; don't derive it from container seconds.
	info, err := validateCanonicalWAV(result.Path, 1<<20, 10000)
	if err != nil || info.frames != int64(result.Timeline.Samples) {
		t.Fatalf("bad actual timeline: %+v %v", info, err)
	}
}
