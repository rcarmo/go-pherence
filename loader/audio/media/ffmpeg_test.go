package media

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeRunner struct {
	mu       sync.Mutex
	calls    []Command
	handlers []func(context.Context, Command) error
}

func (r *fakeRunner) Run(ctx context.Context, cmd Command) error {
	r.mu.Lock()
	idx := len(r.calls)
	copied := Command{Path: cmd.Path, Args: append([]string(nil), cmd.Args...), Stdout: cmd.Stdout, Stderr: cmd.Stderr}
	r.calls = append(r.calls, copied)
	var handler func(context.Context, Command) error
	if idx < len(r.handlers) {
		handler = r.handlers[idx]
	}
	r.mu.Unlock()
	if handler == nil {
		return nil
	}
	return handler(ctx, cmd)
}

func (r *fakeRunner) Calls() []Command {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Command, len(r.calls))
	copy(out, r.calls)
	return out
}

func newTestFFmpeg(t *testing.T, runner *fakeRunner, extra Config) *FFmpeg {
	t.Helper()
	cfg := Config{
		FFprobePath: "ffprobe-test",
		FFmpegPath:  "ffmpeg-test",
		Runner:      runner,
	}
	if extra.MaxInputBytes != 0 {
		cfg.MaxInputBytes = extra.MaxInputBytes
	}
	if extra.MaxDuration != 0 {
		cfg.MaxDuration = extra.MaxDuration
	}
	if extra.MaxDecodeOutputBytes != 0 {
		cfg.MaxDecodeOutputBytes = extra.MaxDecodeOutputBytes
	}
	if extra.MaxProbeStdoutBytes != 0 {
		cfg.MaxProbeStdoutBytes = extra.MaxProbeStdoutBytes
	}
	ff, err := NewFFmpeg(cfg)
	if err != nil {
		t.Fatalf("NewFFmpeg: %v", err)
	}
	return ff
}

func writeProbeJSON(t *testing.T, cmd Command, json string) error {
	t.Helper()
	_, err := io.WriteString(cmd.Stdout, json)
	return err
}

func writeCanonicalWAV(t *testing.T, path string, frames int) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create wav: %v", err)
	}
	defer f.Close()
	dataSize := frames * 2
	buf := new(bytes.Buffer)
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(48+dataSize))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(buf, binary.LittleEndian, uint32(CanonicalSampleRate))
	_ = binary.Write(buf, binary.LittleEndian, uint32(int(CanonicalSampleRate)*2))
	_ = binary.Write(buf, binary.LittleEndian, uint16(2))
	_ = binary.Write(buf, binary.LittleEndian, uint16(16))
	buf.WriteString("LIST")
	_ = binary.Write(buf, binary.LittleEndian, uint32(4))
	buf.WriteString("INFO")
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, uint32(dataSize))
	for i := 0; i < frames; i++ {
		s := int16(0)
		if i%7 == 0 {
			s = int16(math.Sin(float64(i)) * 12000)
		}
		_ = binary.Write(buf, binary.LittleEndian, s)
	}
	if _, err := f.Write(buf.Bytes()); err != nil {
		t.Fatalf("write wav: %v", err)
	}
}

func writeSyntheticWAV(t *testing.T, path string, rate int, frames int) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create source wav: %v", err)
	}
	defer f.Close()
	dataSize := frames * 2
	buf := new(bytes.Buffer)
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVEfmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(buf, binary.LittleEndian, uint32(rate))
	_ = binary.Write(buf, binary.LittleEndian, uint32(rate*2))
	_ = binary.Write(buf, binary.LittleEndian, uint16(2))
	_ = binary.Write(buf, binary.LittleEndian, uint16(16))
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, uint32(dataSize))
	for i := 0; i < frames; i++ {
		_ = binary.Write(buf, binary.LittleEndian, int16(0))
	}
	if _, err := f.Write(buf.Bytes()); err != nil {
		t.Fatalf("write source wav: %v", err)
	}
}

func writeFakeM4A(t *testing.T, path string) {
	t.Helper()
	data := []byte{
		0x00, 0x00, 0x00, 0x18,
		'f', 't', 'y', 'p',
		'M', '4', 'A', ' ',
		0x00, 0x00, 0x00, 0x00,
		'i', 's', 'o', 'm',
		'm', 'p', '4', '2',
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write fake m4a: %v", err)
	}
}

func TestParseProbeJSON(t *testing.T) {
	tests := []struct {
		name      string
		json      string
		wantIndex int
		wantRate  SampleRate
		wantErr   error
	}{
		{
			name:      "chooses lowest audio stream index",
			json:      `{"streams":[{"index":2,"codec_type":"audio","codec_name":"aac","duration":"1.5","sample_rate":"48000","channels":2},{"index":1,"codec_type":"audio","codec_name":"aac","duration":"1.25","sample_rate":"44100","channels":1}],"format":{"duration":"2.0"}}`,
			wantIndex: 1,
			wantRate:  44100,
		},
		{
			name:      "falls back to format duration",
			json:      `{"streams":[{"index":3,"codec_type":"audio","codec_name":"aac","duration":"","sample_rate":"44100","channels":2}],"format":{"duration":"0.5"}}`,
			wantIndex: 3,
			wantRate:  44100,
		},
		{
			name:    "no audio streams",
			json:    `{"streams":[],"format":{"duration":"1.0"}}`,
			wantErr: ErrNoAudio,
		},
		{
			name:    "rejects invalid duration",
			json:    `{"streams":[{"index":0,"codec_type":"audio","codec_name":"aac","duration":"0","sample_rate":"44100","channels":2}],"format":{"duration":"0"}}`,
			wantErr: ErrDurationOutOfRange,
		},
		{
			name:    "rejects malformed json",
			json:    `{`,
			wantErr: ErrInvalidSource,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseProbeJSON([]byte(tt.json), "/tmp/in.m4a", "mov", 1234, 4*time.Hour)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("error=%v want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseProbeJSON: %v", err)
			}
			if got.StreamIndex != tt.wantIndex {
				t.Fatalf("stream index=%d want %d", got.StreamIndex, tt.wantIndex)
			}
			if got.Format.SampleRate != tt.wantRate {
				t.Fatalf("sample rate=%d want %d", got.Format.SampleRate, tt.wantRate)
			}
		})
	}
}

func TestDecodeToFileUsesActualOutputSampleCount(t *testing.T) {
	cases := []struct {
		name         string
		rate         int
		duration     string
		outputFrames int
	}{
		{name: "44.1k rounding", rate: 44100, duration: fmt.Sprintf("%.12f", float64(1001)/44100), outputFrames: 364},
		{name: "48k rounding", rate: 48000, duration: fmt.Sprintf("%.12f", float64(1601)/48000), outputFrames: 534},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			src := filepath.Join(dir, "input.wav")
			writeSyntheticWAV(t, src, tt.rate, tt.rate/10)
			dst := filepath.Join(dir, "decoded.wav")
			runner := &fakeRunner{}
			runner.handlers = []func(context.Context, Command) error{
				func(ctx context.Context, cmd Command) error {
					return writeProbeJSON(t, cmd, fmt.Sprintf(`{"streams":[{"index":0,"codec_type":"audio","codec_name":"pcm_s16le","duration":%q,"sample_rate":%q,"channels":1,"bits_per_sample":16}],"format":{"duration":%q}}`, tt.duration, fmt.Sprintf("%d", tt.rate), tt.duration))
				},
				func(ctx context.Context, cmd Command) error {
					writeCanonicalWAV(t, cmd.Args[len(cmd.Args)-1], tt.outputFrames)
					return nil
				},
			}
			ff := newTestFFmpeg(t, runner, Config{})
			got, err := ff.DecodeToFile(context.Background(), src, dst)
			if err != nil {
				t.Fatalf("DecodeToFile: %v", err)
			}
			if int(got.Timeline.Samples) != tt.outputFrames {
				t.Fatalf("samples=%d want %d", got.Timeline.Samples, tt.outputFrames)
			}
		})
	}
}

func TestProbeAndDecodeCommandSafety(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "-odd name.m4a")
	writeFakeM4A(t, src)
	dst := filepath.Join(dir, "-decoded output.wav")
	runner := &fakeRunner{}
	runner.handlers = []func(context.Context, Command) error{
		func(ctx context.Context, cmd Command) error {
			return writeProbeJSON(t, cmd, `{"streams":[{"index":4,"codec_type":"audio","codec_name":"aac","duration":"1.0","sample_rate":"44100","channels":2}],"format":{"duration":"1.0"}}`)
		},
		func(ctx context.Context, cmd Command) error {
			writeCanonicalWAV(t, cmd.Args[len(cmd.Args)-1], 16000)
			return nil
		},
	}
	ff := newTestFFmpeg(t, runner, Config{})
	if _, err := ff.DecodeToFile(context.Background(), src, dst); err != nil {
		t.Fatalf("DecodeToFile: %v", err)
	}
	calls := runner.Calls()
	if len(calls) != 2 {
		t.Fatalf("calls=%d want 2", len(calls))
	}
	probeArgs := calls[0].Args
	decodeArgs := calls[1].Args
	assertContainsPairs(t, probeArgs, []string{"-protocol_whitelist", "file,pipe", "-f", "mov", "-enable_drefs", "0", "-use_absolute_path", "0", "-i", mustAbs(t, src)})
	assertContainsPairs(t, decodeArgs, []string{"-protocol_whitelist", "file,pipe", "-f", "mov", "-enable_drefs", "0", "-use_absolute_path", "0", "-i", mustAbs(t, src), "-map", "0:4", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le"})
	assertContainsTokens(t, decodeArgs, []string{"-vn", "-sn", "-dn"})
	assertLastPair(t, decodeArgs, "-f", "wav")
	if got := decodeArgs[len(decodeArgs)-1]; !strings.HasPrefix(got, dir) {
		t.Fatalf("decode output path=%q not in temp dir %q", got, dir)
	}
}

func TestProbeRejectsLimitsAndNoAudio(t *testing.T) {
	t.Run("size limit", func(t *testing.T) {
		dir := t.TempDir()
		src := filepath.Join(dir, "input.wav")
		writeSyntheticWAV(t, src, 16000, 8)
		ff := newTestFFmpeg(t, &fakeRunner{}, Config{MaxInputBytes: 16})
		_, err := ff.Probe(context.Background(), src)
		if !errors.Is(err, ErrSizeLimit) {
			t.Fatalf("error=%v want ErrSizeLimit", err)
		}
	})

	t.Run("no audio", func(t *testing.T) {
		dir := t.TempDir()
		src := filepath.Join(dir, "input.m4a")
		writeFakeM4A(t, src)
		runner := &fakeRunner{handlers: []func(context.Context, Command) error{
			func(ctx context.Context, cmd Command) error {
				return writeProbeJSON(t, cmd, `{"streams":[],"format":{"duration":"1.0"}}`)
			},
		}}
		ff := newTestFFmpeg(t, runner, Config{})
		_, err := ff.Probe(context.Background(), src)
		if !errors.Is(err, ErrNoAudio) {
			t.Fatalf("error=%v want ErrNoAudio", err)
		}
	})

	t.Run("duration limit", func(t *testing.T) {
		dir := t.TempDir()
		src := filepath.Join(dir, "input.m4a")
		writeFakeM4A(t, src)
		runner := &fakeRunner{handlers: []func(context.Context, Command) error{
			func(ctx context.Context, cmd Command) error {
				return writeProbeJSON(t, cmd, `{"streams":[{"index":0,"codec_type":"audio","codec_name":"aac","duration":"20.0","sample_rate":"44100","channels":2}],"format":{"duration":"20.0"}}`)
			},
		}}
		ff := newTestFFmpeg(t, runner, Config{MaxDuration: time.Second})
		_, err := ff.Probe(context.Background(), src)
		if !errors.Is(err, ErrDurationOutOfRange) {
			t.Fatalf("error=%v want ErrDurationOutOfRange", err)
		}
	})
}

func TestDecodeRejectsMalformedOutputAndCleansTemp(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "input.wav")
	writeSyntheticWAV(t, src, 16000, 16)
	dst := filepath.Join(dir, "decoded.wav")
	runner := &fakeRunner{}
	runner.handlers = []func(context.Context, Command) error{
		func(ctx context.Context, cmd Command) error {
			return writeProbeJSON(t, cmd, `{"streams":[{"index":0,"codec_type":"audio","codec_name":"pcm_s16le","duration":"1.0","sample_rate":"16000","channels":1,"bits_per_sample":16}],"format":{"duration":"1.0"}}`)
		},
		func(ctx context.Context, cmd Command) error {
			return os.WriteFile(cmd.Args[len(cmd.Args)-1], []byte("not-wav"), 0o644)
		},
	}
	ff := newTestFFmpeg(t, runner, Config{})
	_, err := ff.DecodeToFile(context.Background(), src, dst)
	if !errors.Is(err, ErrInvalidOutput) {
		t.Fatalf("error=%v want ErrInvalidOutput", err)
	}
	matches, findErr := filepath.Glob(filepath.Join(dir, ".decoded.wav.tmp-*.wav"))
	if findErr != nil {
		t.Fatalf("glob temp files: %v", findErr)
	}
	if len(matches) != 0 {
		t.Fatalf("temp files remain: %v", matches)
	}
	if _, err := os.Stat(dst); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("destination exists after failure")
	}
}

func TestDecodeCancellationCleanup(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "input.wav")
	writeSyntheticWAV(t, src, 16000, 16)
	dst := filepath.Join(dir, "decoded.wav")
	runner := &fakeRunner{}
	runner.handlers = []func(context.Context, Command) error{
		func(ctx context.Context, cmd Command) error {
			return writeProbeJSON(t, cmd, `{"streams":[{"index":0,"codec_type":"audio","codec_name":"pcm_s16le","duration":"10.0","sample_rate":"16000","channels":1,"bits_per_sample":16}],"format":{"duration":"10.0"}}`)
		},
		func(ctx context.Context, cmd Command) error {
			path := cmd.Args[len(cmd.Args)-1]
			if err := os.WriteFile(path, make([]byte, 256), 0o644); err != nil {
				return err
			}
			<-ctx.Done()
			return ctx.Err()
		},
	}
	ff := newTestFFmpeg(t, runner, Config{})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(40 * time.Millisecond)
		cancel()
	}()
	_, err := ff.DecodeToFile(ctx, src, dst)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error=%v want context.Canceled", err)
	}
	matches, _ := filepath.Glob(filepath.Join(dir, ".decoded.wav.tmp-*.wav"))
	if len(matches) != 0 {
		t.Fatalf("temp files remain after cancel: %v", matches)
	}
}

func TestDecodePreservesExistingDestination(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "input.wav")
	writeSyntheticWAV(t, src, 16000, 16)
	dst := filepath.Join(dir, "decoded.wav")
	orig := []byte("keep me")
	if err := os.WriteFile(dst, orig, 0o644); err != nil {
		t.Fatalf("write dst: %v", err)
	}
	runner := &fakeRunner{}
	ff := newTestFFmpeg(t, runner, Config{})
	_, err := ff.DecodeToFile(context.Background(), src, dst)
	if err == nil {
		t.Fatal("expected destination exists error")
	}
	if len(runner.Calls()) != 0 {
		t.Fatalf("runner was called despite existing destination")
	}
	got, readErr := os.ReadFile(dst)
	if readErr != nil {
		t.Fatalf("read dst: %v", readErr)
	}
	if !reflect.DeepEqual(got, orig) {
		t.Fatalf("destination changed: %q", got)
	}
}

func TestDecodeOutputLimit(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "input.wav")
	writeSyntheticWAV(t, src, 16000, 16)
	dst := filepath.Join(dir, "decoded.wav")
	runner := &fakeRunner{}
	runner.handlers = []func(context.Context, Command) error{
		func(ctx context.Context, cmd Command) error {
			return writeProbeJSON(t, cmd, `{"streams":[{"index":0,"codec_type":"audio","codec_name":"pcm_s16le","duration":"1.0","sample_rate":"16000","channels":1,"bits_per_sample":16}],"format":{"duration":"1.0"}}`)
		},
		func(ctx context.Context, cmd Command) error {
			writeCanonicalWAV(t, cmd.Args[len(cmd.Args)-1], 4096)
			return nil
		},
	}
	ff := newTestFFmpeg(t, runner, Config{MaxDecodeOutputBytes: 128})
	_, err := ff.DecodeToFile(context.Background(), src, dst)
	if !errors.Is(err, ErrDecodeOutputLimit) {
		t.Fatalf("error=%v want ErrDecodeOutputLimit", err)
	}
}

func assertContainsPairs(t *testing.T, args []string, want []string) {
	t.Helper()
	for i := 0; i < len(want); i += 2 {
		k, v := want[i], want[i+1]
		found := false
		for j := 0; j+1 < len(args); j++ {
			if args[j] == k && args[j+1] == v {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("args %v missing pair %q %q", args, k, v)
		}
	}
}

func mustAbs(t *testing.T, path string) string {
	t.Helper()
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("Abs(%q): %v", path, err)
	}
	return abs
}

func assertContainsTokens(t *testing.T, args []string, want []string) {
	t.Helper()
	for _, token := range want {
		found := false
		for _, arg := range args {
			if arg == token {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("args %v missing token %q", args, token)
		}
	}
}

func assertLastPair(t *testing.T, args []string, k, v string) {
	t.Helper()
	for i := len(args) - 2; i >= 0; i-- {
		if args[i] == k {
			if args[i+1] != v {
				t.Fatalf("args %v pair %q has value %q want %q", args, k, args[i+1], v)
			}
			return
		}
	}
	t.Fatalf("args %v missing pair %q %q", args, k, v)
}
