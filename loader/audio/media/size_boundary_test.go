package media

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestCanonicalReaderSizeLimitInclusive(t *testing.T) {
	path := pcmFixture(t, 16)
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := validateCanonicalWAV(path, stat.Size(), 16); err != nil {
		t.Fatalf("valid WAV exactly at its byte/frame limit: %v", err)
	}
	if _, err := validateCanonicalWAV(path, stat.Size()-1, 16); !errors.Is(err, ErrDecodeOutputLimit) {
		t.Fatalf("accepted WAV above its byte limit: %v", err)
	}
	if _, err := validateCanonicalWAV(path, stat.Size(), 15); !errors.Is(err, ErrDecodeOutputLimit) {
		t.Fatalf("accepted WAV above its frame limit: %v", err)
	}
}

func TestFFmpegOutputAtCeilingRemainsRejected(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "input.wav")
	dst := filepath.Join(dir, "output.wav")
	writeSyntheticWAV(t, src, 16000, 16)
	// Test fixture: 56-byte RIFF+fmt+LIST+data header plus 32 bytes of PCM.
	// The fake process returns success, just as ffmpeg can when it reaches -fs.
	const ceiling = 56 + 16*2
	runner := &fakeRunner{handlers: []func(context.Context, Command) error{
		func(_ context.Context, cmd Command) error {
			return writeProbeJSON(t, cmd, `{"streams":[{"index":0,"codec_type":"audio","codec_name":"pcm_s16le","duration":"0.001","sample_rate":"16000","channels":1}]}`)
		},
		func(_ context.Context, cmd Command) error {
			writeCanonicalWAV(t, cmd.Args[len(cmd.Args)-1], 16)
			return nil
		},
	}}
	ff := newTestFFmpeg(t, runner, Config{MaxDecodeOutputBytes: ceiling})
	if _, err := ff.DecodeToFile(context.Background(), src, dst); !errors.Is(err, ErrDecodeOutputLimit) {
		t.Fatalf("published a possibly truncated -fs output: %v", err)
	}
	if _, err := os.Stat(dst); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("destination exists despite output ceiling")
	}
	partials, err := filepath.Glob(filepath.Join(dir, ".output.wav.tmp-*.wav"))
	if err != nil || len(partials) != 0 {
		t.Fatalf("temporary output remains: %v %v", partials, err)
	}
	if _, err := os.Stat(src); err != nil {
		t.Fatal("source was removed")
	}
}
