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
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

type decodeFixtureAdapter struct {
	calls    int
	mutate   func(context.Context, string, string) (media.DecodeResult, error)
	metadata string
}

func (a *decodeFixtureAdapter) Probe(context.Context, string) (media.ProbeResult, error) {
	panic("unused: real adapter probes inside DecodeToFile")
}
func (a *decodeFixtureAdapter) DecodeToFile(ctx context.Context, src, dst string) (media.DecodeResult, error) {
	a.calls++
	if a.mutate != nil {
		return a.mutate(ctx, src, dst)
	}
	raw, e := os.ReadFile(src)
	if e != nil {
		return media.DecodeResult{}, e
	}
	if !strings.HasSuffix(src, ".wav") || string(raw) != "source audio" {
		return media.DecodeResult{}, errors.New("materialised source changed")
	}
	samples := []int16{-32768, -123, 0, 1, 32767}
	b := testWAV(samples, a.metadata)
	if e = os.WriteFile(dst, b, 0600); e != nil {
		return media.DecodeResult{}, e
	}
	return media.DecodeResult{Path: dst, SizeBytes: int64(len(b)), Format: media.AudioFormat{Container: "wav", Encoding: "pcm_s16le", SampleRate: 16000, Channels: 1, BitsPerSample: 16}, Timeline: media.Timeline{SampleRate: 16000, Samples: media.SampleCount(len(samples))}}, nil
}
func testWAV(samples []int16, junk string) []byte {
	b := make([]byte, 44)
	copy(b, "RIFF")
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], 16000)
	binary.LittleEndian.PutUint32(b[28:], 32000)
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	if junk != "" {
		b = b[:36]
		j := []byte(junk)
		n := len(j)
		size := make([]byte, 4)
		binary.LittleEndian.PutUint32(size, uint32(n))
		b = append(b, []byte("JUNK")...)
		b = append(b, size...)
		b = append(b, j...)
		if n%2 != 0 {
			b = append(b, 0)
		}
		b = append(b, make([]byte, 8)...)
	}
	copy(b[len(b)-8:], "data")
	binary.LittleEndian.PutUint32(b[len(b)-4:], uint32(len(samples)*2))
	for _, s := range samples {
		b = append(b, byte(s), byte(uint16(s)>>8))
	}
	binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
	return b
}
func decodeConfig(t *testing.T) FFmpegDecodeConfig {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "fixture-executable")
	if e := os.WriteFile(bin, []byte("fixture only"), 0700); e != nil {
		t.Fatal(e)
	}
	return FFmpegDecodeConfig{FFmpegPath: bin, FFprobePath: bin, FFmpegSHA256: hash([]byte("fixture only")), FFprobeSHA256: hash([]byte("fixture only")), InputExtension: ".wav", MaxInputBytes: 1 << 20, MaxOutputBytes: 90000, MaxDuration: time.Second}
}
func assertNoScratch(t *testing.T, s *Store, id string) {
	t.Helper()
	entries, e := os.ReadDir(filepath.Join(s.root.Name(), id))
	if e != nil {
		t.Fatal(e)
	}
	for _, x := range entries {
		if strings.HasPrefix(x.Name(), ".work-") || strings.HasPrefix(x.Name(), ".payload-") {
			t.Fatal("scratch leak", x.Name())
		}
	}
}
func TestDecodeStageCanonicalAndResume(t *testing.T) {
	s, dir := openTest(t)
	cfg := decodeConfig(t)
	if _, e := NewFFmpegDecodeStage(cfg); e != nil {
		t.Fatal(e)
	}
	adapter := &decodeFixtureAdapter{metadata: "irrelevant ffmpeg metadata/path/time"}
	decode := newDecodeStage(cfg, adapter)
	m := createTest(t, s)
	fail := stage("transcript", func(context.Context, *Input, io.Writer) error { return errors.New("later model failure") })
	m, e := s.Run(context.Background(), m.ID, config, []Stage{decode, fail}, nil)
	if e == nil || m.Status != Failed || len(m.Checkpoints) != 1 {
		t.Fatal(m, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), m.ID, "decode")
	payload := readAll(t, r, e)
	if payload != string(testWAV([]int16{-32768, -123, 0, 1, 32767}, "")) {
		t.Fatal("canonical PCM changed")
	}
	assertNoScratch(t, s, m.ID)
	s.Close()
	s, e = Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	fail.Run = func(_ context.Context, _ *Input, w io.Writer) error { _, e := w.Write([]byte("success")); return e }
	m, e = s.Run(context.Background(), m.ID, config, []Stage{decode, fail}, nil)
	if e != nil || m.Status != Complete || adapter.calls != 1 {
		t.Fatal("decode repeated", adapter.calls, e)
	}
	// Metadata changes cannot alter the checkpoint: fresh job, same canonical PCM.
	adapter.metadata = "different"
	other := createTest(t, s)
	other, e = s.Run(context.Background(), other.ID, config, []Stage{decode}, nil)
	if e != nil || other.Checkpoints[0].Blob.SHA256 != m.Checkpoints[0].Blob.SHA256 {
		t.Fatal("metadata nondeterminism", e)
	}
	changed := cfg
	changed.FFmpegSHA256 = hash([]byte("newbinary"))
	different := newDecodeStage(changed, adapter)
	if _, e = s.Run(context.Background(), other.ID, config, []Stage{different}, nil); !errors.Is(e, ErrConfiguration) {
		t.Fatal("changed identity reused", e)
	}
}
func TestDecodeStageValidationAndCleanup(t *testing.T) {
	for _, kind := range []string{"adapter-error", "wrong-path", "bad-format", "bad-count", "bad-size", "bad-timing", "malformed", "symlink", "cancel", "panic", "changed-executable", "upload-cap", "quota"} {
		t.Run(kind, func(t *testing.T) {
			s, _ := openTest(t)
			cfg := decodeConfig(t)
			m := createTest(t, s)
			good := &decodeFixtureAdapter{}
			sentinel := errors.New("decoder failure")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			adapter := &decodeFixtureAdapter{mutate: func(ctx context.Context, src, dst string) (media.DecodeResult, error) {
				if kind == "adapter-error" {
					os.WriteFile(dst, []byte("partial"), 0600)
					return media.DecodeResult{}, sentinel
				}
				if kind == "panic" {
					panic("worker")
				}
				r, e := good.DecodeToFile(ctx, src, dst)
				if e != nil {
					return r, e
				}
				switch kind {
				case "wrong-path":
					r.Path = src
				case "bad-format":
					r.Format.SampleRate = 8000
				case "bad-count":
					r.Timeline.Samples++
				case "bad-size":
					r.SizeBytes++
				case "bad-timing":
					r.Source.Start = -time.Second
				case "malformed":
					os.WriteFile(dst, []byte("not wav"), 0600)
					r.SizeBytes = 7
				case "symlink":
					os.Remove(dst)
					os.Symlink(src, dst)
				case "cancel":
					cancel()
				}
				return r, nil
			}}
			if kind == "changed-executable" {
				os.WriteFile(cfg.FFmpegPath, []byte("changed"), 0700)
			}
			if kind == "upload-cap" {
				cfg.MaxInputBytes = 1
			}
			if kind == "quota" {
				s.limits.MaxBytes = 200000
			}
			result, e := s.Run(ctx, m.ID, config, []Stage{newDecodeStage(cfg, adapter)}, nil)
			if e == nil || result.Status == Complete || len(result.Checkpoints) != 0 {
				t.Fatal("accepted", kind, e)
			}
			if kind == "adapter-error" && !errors.Is(e, sentinel) {
				t.Fatal("error lost", e)
			}
			if kind == "cancel" && !errors.Is(e, context.Canceled) {
				t.Fatal("cancel lost", e)
			}
			if (kind == "changed-executable" || kind == "upload-cap" || kind == "quota") && adapter.calls != 0 {
				t.Fatal("invalid admitted")
			}
			assertNoScratch(t, s, m.ID)
		})
	}
	cfg := decodeConfig(t)
	for _, kind := range []string{"path", "hash", "type", "duration", "input", "output"} {
		c := cfg
		switch kind {
		case "path":
			c.FFmpegPath = "ffmpeg"
		case "hash":
			c.FFprobeSHA256 = "bad"
		case "type":
			c.InputExtension = ".mp3"
		case "duration":
			c.MaxDuration = 5 * time.Hour
		case "input":
			c.MaxInputBytes = 0
		case "output":
			c.MaxOutputBytes = 1
		}
		if _, e := NewFFmpegDecodeStage(c); e == nil {
			t.Fatal("bad config", kind)
		}
	}
}
func TestCopyExactContract(t *testing.T) {
	for _, size := range []int64{0, 1, 32769} {
		var out bytes.Buffer
		b := bytes.Repeat([]byte{'x'}, int(size))
		n, e := copyExact(context.Background(), &out, bytes.NewReader(b), size)
		if e != nil || n != size || !bytes.Equal(out.Bytes(), b) {
			t.Fatal(n, e)
		}
	}
	for _, size := range []int64{1, 3} {
		_, e := copyExact(context.Background(), io.Discard, strings.NewReader("xx"), size)
		if e == nil {
			t.Fatal("extent")
		}
	}
	cc, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := copyExact(cc, io.Discard, strings.NewReader("x"), 1); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
}

// Real subprocess test is explicit and uses only generated public-free PCM.
func TestFFmpegJobDecodeIntegration(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_FFMPEG") != "1" {
		t.Skip("explicit FFmpeg integration")
	}
	ff, e := exec.LookPath("ffmpeg")
	if e != nil {
		t.Fatal(e)
	}
	fp, e := exec.LookPath("ffprobe")
	if e != nil {
		t.Fatal(e)
	}
	ff, _ = filepath.Abs(ff)
	fp, _ = filepath.Abs(fp)
	ffbytes, e := os.ReadFile(ff)
	if e != nil {
		t.Fatal(e)
	}
	fpbytes, e := os.ReadFile(fp)
	if e != nil {
		t.Fatal(e)
	}
	cfg := FFmpegDecodeConfig{FFmpegPath: ff, FFprobePath: fp, FFmpegSHA256: hash(ffbytes), FFprobeSHA256: hash(fpbytes), InputExtension: ".wav", MaxInputBytes: 1 << 20, MaxOutputBytes: 90000, MaxDuration: time.Second}
	decode, e := NewFFmpegDecodeStage(cfg)
	if e != nil {
		t.Fatal(e)
	}
	s, _ := openTest(t)
	samples := make([]int16, 1600)
	for i := range samples {
		samples[i] = int16(i%32000 - 16000)
	}
	original := testWAV(samples, "generated")
	var prior string
	for run := 0; run < 2; run++ {
		m, e := s.Create(context.Background(), "../../; evil name.wav", config, bytes.NewReader(original))
		if e != nil {
			t.Fatal(e)
		}
		m, e = s.Run(context.Background(), m.ID, config, []Stage{decode}, nil)
		if e != nil {
			t.Fatal(e)
		}
		r, e := s.OpenCheckpoint(context.Background(), m.ID, "decode")
		decoded := readAll(t, r, e)
		if decoded != string(testWAV(samples, "")) {
			t.Fatal("FFmpeg PCM differs")
		}
		if run == 1 && prior != decoded {
			t.Fatal("nondeterminism")
		}
		prior = decoded
		assertNoScratch(t, s, m.ID)
	}
	t.Logf("FFmpeg hashes %s / %s;1600 samples exact, two repeats", cfg.FFmpegSHA256, cfg.FFprobeSHA256)
}

// An injected adapter waits for cancellation and returns only after owned work
// has stopped. The stage must wait too, then clean its scratch before returning.
func TestDecodeStageCancelDrain(t *testing.T) {
	s, _ := openTest(t)
	m := createTest(t, s)
	cfg := decodeConfig(t)
	entered := make(chan struct{})
	release := make(chan struct{})
	var alive atomic.Bool
	adapter := &decodeFixtureAdapter{mutate: func(ctx context.Context, _, _ string) (media.DecodeResult, error) {
		alive.Store(true)
		close(entered)
		<-ctx.Done()
		<-release
		alive.Store(false)
		return media.DecodeResult{}, ctx.Err()
	}}
	done := make(chan error, 1)
	go func() {
		_, e := s.Run(context.Background(), m.ID, config, []Stage{newDecodeStage(cfg, adapter)}, nil)
		done <- e
	}()
	<-entered
	if !s.Cancel(m.ID) {
		t.Fatal("cancel")
	}
	select {
	case e := <-done:
		t.Fatal("returned before drain", e)
	default:
	}
	if !alive.Load() {
		t.Fatal("test drain not active")
	}
	close(release)
	if e := <-done; !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if alive.Load() {
		t.Fatal("outlived run")
	}
	assertNoScratch(t, s, m.ID)
}

func TestFFmpegJobAACIntegration(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_FFMPEG") != "1" {
		t.Skip("explicit FFmpeg integration")
	}
	ff, e := exec.LookPath("ffmpeg")
	if e != nil {
		t.Fatal(e)
	}
	fp, e := exec.LookPath("ffprobe")
	if e != nil {
		t.Fatal(e)
	}
	ff, _ = filepath.Abs(ff)
	fp, _ = filepath.Abs(fp)
	ffbytes, e := os.ReadFile(ff)
	if e != nil {
		t.Fatal(e)
	}
	fpbytes, e := os.ReadFile(fp)
	if e != nil {
		t.Fatal(e)
	}
	s, dir := openTest(t)
	fixtures := t.TempDir()
	sourcePath := filepath.Join(fixtures, "tone.wav")
	samples := make([]int16, 16000)
	for i := range samples {
		samples[i] = int16(4000 * math.Sin(float64(i)*2*math.Pi*440/16000))
	}
	if e = os.WriteFile(sourcePath, testWAV(samples, ""), 0600); e != nil {
		t.Fatal(e)
	}
	m4a := filepath.Join(fixtures, "tone.m4a")
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, ff, "-nostdin", "-v", "error", "-threads", "1", "-i", sourcePath, "-c:a", "aac", "-b:a", "96k", "-y", m4a)
	if b, e := cmd.CombinedOutput(); e != nil {
		t.Fatal("synthetic encode", e, string(b))
	}
	input, e := os.ReadFile(m4a)
	if e != nil {
		t.Fatal(e)
	}
	cfg := FFmpegDecodeConfig{FFmpegPath: ff, FFprobePath: fp, FFmpegSHA256: hash(ffbytes), FFprobeSHA256: hash(fpbytes), InputExtension: ".m4a", MaxInputBytes: 1 << 20, MaxOutputBytes: 129000, MaxDuration: 2 * time.Second}
	st, e := NewFFmpegDecodeStage(cfg)
	if e != nil {
		t.Fatal(e)
	}
	var previous string
	for i := 0; i < 2; i++ {
		job, e := s.Create(ctx, "tone.m4a", config, bytes.NewReader(input))
		if e != nil {
			t.Fatal(e)
		}
		job, e = s.Run(ctx, job.ID, config, []Stage{st}, nil)
		if e != nil {
			t.Fatal(e)
		}
		r, e := s.OpenCheckpoint(ctx, job.ID, "decode")
		decoded := readAll(t, r, e)
		if i > 0 && decoded != previous {
			t.Fatal("AAC checkpoint not deterministic")
		}
		previous = decoded
		checkpoint := filepath.Join(dir, job.ID, job.Checkpoints[0].Blob.File)
		pcm, e := media.OpenCanonicalPCM(ctx, checkpoint)
		if e != nil {
			t.Fatal(e)
		}
		timeline := pcm.Timeline()
		pcm.Close()
		if timeline.Samples < 16000 || timeline.Samples > 17024 {
			t.Fatal("AAC count", timeline)
		}
		assertNoScratch(t, s, job.ID)
		t.Logf("AAC frames=%d checkpointSHA=%s", timeline.Samples, job.Checkpoints[0].Blob.SHA256)
	}
}

func TestDecodeDurationFloor(t *testing.T) {
	cfg := decodeConfig(t)
	cfg.MaxDuration = time.Nanosecond
	cfg.MaxOutputBytes = 46
	if _, e := NewFFmpegDecodeStage(cfg); e == nil {
		t.Fatal("subsample duration accepted")
	}
	cfg.MaxDuration = time.Second / 16000
	if _, e := NewFFmpegDecodeStage(cfg); e != nil {
		t.Fatal("single-sample duration", e)
	}
}

func TestDecodeRetainedScratchAdmission(t *testing.T) {
	s, _ := openTest(t)
	retained := createTest(t, s)
	job := createTest(t, s)
	cfg := decodeConfig(t)
	used, _, e := s.Usage()
	if e != nil {
		t.Fatal(e)
	}
	// Enough for a new decode without the old scratch; not enough with it.
	s.limits.MaxBytes = used + job.Input.Bytes + 2*cfg.MaxOutputBytes + maxManifest + 10000
	relative := retained.ID + "/.work-decode-retained"
	if e = s.root.Mkdir(relative, 0700); e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(filepath.Join(s.root.Name(), relative, "input.wav"), make([]byte, 100000), 0600); e != nil {
		t.Fatal(e)
	}
	inv, e := s.Inventory()
	if e != nil || len(inv) != 2 {
		t.Fatal(inv, e)
	}
	adapter := &decodeFixtureAdapter{}
	st := newDecodeStage(cfg, adapter)
	if _, e = s.Run(context.Background(), job.ID, config, []Stage{st}, nil); !errors.Is(e, ErrLimit) || adapter.calls != 0 {
		t.Fatal("retained scratch not charged", adapter.calls, e)
	}
	assertNoScratch(t, s, job.ID)
	if e = s.Delete(context.Background(), retained.ID); e != nil {
		t.Fatal(e)
	}
	job, e = s.Run(context.Background(), job.ID, config, []Stage{st}, nil)
	if e != nil || job.Status != Complete || adapter.calls != 1 {
		t.Fatal("explicit cleanup did not free quota", job, e)
	}
	assertNoScratch(t, s, job.ID)
}
