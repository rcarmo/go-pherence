package media

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type FFmpeg struct {
	cfg Config
}

type sourceKind struct {
	demuxer            string
	container          string
	disableExternalRef bool
}

type probeStream struct {
	Index         *int   `json:"index"`
	CodecType     string `json:"codec_type"`
	CodecName     string `json:"codec_name"`
	Duration      string `json:"duration"`
	SampleRate    string `json:"sample_rate"`
	Channels      int    `json:"channels"`
	BitsPerSample int    `json:"bits_per_sample"`
}

type ffprobeResponse struct {
	Streams []probeStream `json:"streams"`
	Format  struct {
		Duration string `json:"duration"`
	} `json:"format"`
}

func NewFFmpeg(cfg Config) (*FFmpeg, error) {
	if cfg.MaxInputBytes < 0 || cfg.MaxDuration < 0 || cfg.MaxProbeStdoutBytes < 0 || cfg.MaxProbeStderrBytes < 0 || cfg.MaxDecodeStderrBytes < 0 || cfg.MaxDecodeOutputBytes < 0 {
		return nil, fmt.Errorf("negative media limits")
	}
	cfg = withDefaults(cfg)
	// Deliberately limited to the speech service's four-hour RIFF contract.
	if cfg.MaxDuration > DefaultMaxDuration || cfg.MaxProbeStdoutBytes > 4<<20 || cfg.MaxProbeStderrBytes > 1<<20 || cfg.MaxDecodeStderrBytes > 1<<20 {
		return nil, fmt.Errorf("media limits exceed adapter bounds")
	}
	if cfg.FFprobePath == "" || cfg.FFmpegPath == "" {
		return nil, fmt.Errorf("ffmpeg and ffprobe paths are required")
	}
	if cfg.Runner == nil {
		cfg.Runner = execRunner{}
	}
	if cfg.MaxInputBytes <= 0 || cfg.MaxDuration <= 0 || cfg.MaxDecodeOutputBytes <= 0 || cfg.MaxDecodeOutputBytes > canonicalMaxWAVBytes(cfg.MaxDuration) {
		return nil, fmt.Errorf("invalid media limits")
	}
	return &FFmpeg{cfg: cfg}, nil
}

func withDefaults(cfg Config) Config {
	if cfg.MaxInputBytes <= 0 {
		cfg.MaxInputBytes = DefaultMaxInputBytes
	}
	if cfg.MaxDuration <= 0 {
		cfg.MaxDuration = DefaultMaxDuration
	}
	if cfg.MaxProbeStdoutBytes <= 0 {
		cfg.MaxProbeStdoutBytes = DefaultMaxProbeStdoutBytes
	}
	if cfg.MaxProbeStderrBytes <= 0 {
		cfg.MaxProbeStderrBytes = DefaultMaxProbeStderrBytes
	}
	if cfg.MaxDecodeStderrBytes <= 0 {
		cfg.MaxDecodeStderrBytes = DefaultMaxDecodeStderrBytes
	}
	if cfg.MaxDecodeOutputBytes <= 0 {
		cfg.MaxDecodeOutputBytes = canonicalMaxWAVBytes(cfg.MaxDuration)
	}
	return cfg
}

func canonicalMaxWAVBytes(maxDuration time.Duration) int64 {
	frames := maxFramesForDuration(maxDuration)
	return 64*1024 + frames*int64(CanonicalChannels*(CanonicalBitsPerSample/8))
}

func maxFramesForDuration(d time.Duration) int64 {
	if d <= 0 {
		return 0
	}
	frames, ok := multiplyDivide(int64(d), int64(CanonicalSampleRate), int64(time.Second))
	if !ok {
		return maxInt64
	}
	return frames
}

func (f *FFmpeg) Probe(ctx context.Context, path string) (ProbeResult, error) {
	if err := ctx.Err(); err != nil {
		return ProbeResult{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	absPath, kind, size, err := inspectSource(path, f.cfg.MaxInputBytes)
	if err != nil {
		return ProbeResult{}, err
	}
	stdout := &hardLimitBuffer{limit: f.cfg.MaxProbeStdoutBytes}
	stderr := &clippedBuffer{limit: f.cfg.MaxProbeStderrBytes}
	if err := f.cfg.Runner.Run(ctx, Command{
		Path:   f.cfg.FFprobePath,
		Args:   buildProbeArgs(absPath, kind),
		Stdout: stdout,
		Stderr: stderr,
	}); err != nil {
		if errors.Is(err, errWriterLimit) {
			return ProbeResult{}, fmt.Errorf("%w: ffprobe output too large", ErrInvalidSource)
		}
		if ctx.Err() != nil {
			return ProbeResult{}, ctx.Err()
		}
		return ProbeResult{}, fmt.Errorf("%w: ffprobe failed", ErrInvalidSource)
	}
	if err := ctx.Err(); err != nil {
		return ProbeResult{}, err
	}
	return parseProbeJSON(stdout.Bytes(), absPath, kind.container, size, f.cfg.MaxDuration)
}

func (f *FFmpeg) DecodeToFile(ctx context.Context, srcPath, dstPath string) (DecodeResult, error) {
	srcAbs, err := filepath.Abs(srcPath)
	if err != nil {
		return DecodeResult{}, fmt.Errorf("resolve source path: %w", err)
	}
	dstAbs, err := filepath.Abs(dstPath)
	if err != nil {
		return DecodeResult{}, fmt.Errorf("resolve destination path: %w", err)
	}
	if srcAbs == dstAbs {
		return DecodeResult{}, fmt.Errorf("destination must differ from source")
	}
	if _, err := os.Lstat(dstAbs); err == nil {
		return DecodeResult{}, fmt.Errorf("destination already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return DecodeResult{}, fmt.Errorf("stat destination: %w", err)
	}

	probe, err := f.Probe(ctx, srcAbs)
	if err != nil {
		return DecodeResult{}, err
	}
	if probe.Duration <= 0 || probe.Duration > f.cfg.MaxDuration {
		return DecodeResult{}, ErrDurationOutOfRange
	}
	maxFrames := maxFramesForDuration(f.cfg.MaxDuration)

	if err := os.MkdirAll(filepath.Dir(dstAbs), 0o700); err != nil {
		return DecodeResult{}, fmt.Errorf("ensure destination directory: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dstAbs), "."+filepath.Base(dstAbs)+".tmp-*.wav")
	if err != nil {
		return DecodeResult{}, fmt.Errorf("create temp output: %w", err)
	}
	tmpPath := tmp.Name()
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return DecodeResult{}, fmt.Errorf("close temp output: %w", err)
	}
	removeTmp := true
	defer func() {
		if removeTmp {
			_ = os.Remove(tmpPath)
		}
	}()

	deadlineCtx, deadlineCancel := context.WithTimeout(ctx, 8*time.Hour)
	defer deadlineCancel()
	runCtx, cancel := context.WithCancelCause(deadlineCtx)
	defer cancel(nil)
	monitorStop := make(chan struct{})
	monitorDone := make(chan struct{})
	go func() {
		defer close(monitorDone)
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-runCtx.Done():
				return
			case <-monitorStop:
				return
			case <-ticker.C:
				fi, err := os.Stat(tmpPath)
				if err == nil && fi.Size() > f.cfg.MaxDecodeOutputBytes {
					cancel(ErrDecodeOutputLimit)
					return
				}
			}
		}
	}()

	stderr := &clippedBuffer{limit: f.cfg.MaxDecodeStderrBytes}
	err = f.cfg.Runner.Run(runCtx, Command{
		Path:   f.cfg.FFmpegPath,
		Args:   buildDecodeArgs(probe, srcAbs, tmpPath, f.cfg.MaxDecodeOutputBytes),
		Stderr: stderr,
	})
	close(monitorStop)
	<-monitorDone
	if cause := context.Cause(runCtx); cause != nil {
		return DecodeResult{}, cause
	}
	if err != nil {
		if ctx.Err() != nil {
			return DecodeResult{}, ctx.Err()
		}
		return DecodeResult{}, fmt.Errorf("decode failed")
	}
	if ctx.Err() != nil {
		return DecodeResult{}, ctx.Err()
	}
	if _, err := os.Lstat(dstAbs); err == nil {
		return DecodeResult{}, fmt.Errorf("destination already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return DecodeResult{}, fmt.Errorf("stat destination: %w", err)
	}

	info, err := validateCanonicalWAV(tmpPath, f.cfg.MaxDecodeOutputBytes, maxFrames)
	if err != nil {
		return DecodeResult{}, err
	}
	if cause := context.Cause(runCtx); cause != nil {
		return DecodeResult{}, cause
	}
	if err := renameNoReplace(tmpPath, dstAbs); err != nil {
		return DecodeResult{}, fmt.Errorf("publish decoded wav: %w", err)
	}
	removeTmp = false
	return DecodeResult{
		Path:      dstAbs,
		SizeBytes: info.fileSize,
		Format:    canonicalWAV(),
		Timeline: Timeline{
			SampleRate: CanonicalSampleRate,
			Samples:    SampleCount(info.frames),
		},
	}, nil
}

func inspectSource(path string, maxBytes int64) (string, sourceKind, int64, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", sourceKind{}, 0, fmt.Errorf("resolve source path: %w", err)
	}
	fi, err := os.Stat(absPath)
	if err != nil {
		return "", sourceKind{}, 0, fmt.Errorf("stat source: %w", err)
	}
	if !fi.Mode().IsRegular() {
		return "", sourceKind{}, 0, fmt.Errorf("%w: source must be a regular file", ErrInvalidSource)
	}
	if fi.Size() <= 0 {
		return "", sourceKind{}, 0, fmt.Errorf("%w: empty file", ErrInvalidSource)
	}
	if maxBytes > 0 && fi.Size() > maxBytes {
		return "", sourceKind{}, 0, fmt.Errorf("%w: %d bytes", ErrSizeLimit, fi.Size())
	}
	kind, err := sniffSource(absPath)
	if err != nil {
		return "", sourceKind{}, 0, err
	}
	return absPath, kind, fi.Size(), nil
}

func sniffSource(path string) (sourceKind, error) {
	ext := strings.ToLower(filepath.Ext(path))
	buf := make([]byte, 32)
	f, err := os.Open(path)
	if err != nil {
		return sourceKind{}, fmt.Errorf("open source: %w", err)
	}
	defer f.Close()
	n, err := f.Read(buf)
	if err != nil && !errors.Is(err, io.EOF) {
		return sourceKind{}, fmt.Errorf("read source header: %w", err)
	}
	buf = buf[:n]
	switch ext {
	case ".wav":
		if len(buf) < 12 || string(buf[0:4]) != "RIFF" || string(buf[8:12]) != "WAVE" {
			return sourceKind{}, fmt.Errorf("%w: expected RIFF/WAVE content", ErrUnsupportedInput)
		}
		return sourceKind{demuxer: "wav", container: "wav"}, nil
	case ".m4a", ".mp4", ".mov":
		if len(buf) < 12 || string(buf[4:8]) != "ftyp" {
			return sourceKind{}, fmt.Errorf("%w: expected ISO BMFF content", ErrUnsupportedInput)
		}
		return sourceKind{demuxer: "mov", container: "mov", disableExternalRef: true}, nil
	default:
		return sourceKind{}, fmt.Errorf("%w: unsupported extension %q", ErrUnsupportedInput, ext)
	}
}

func parseProbeJSON(data []byte, path string, container string, size int64, maxDuration time.Duration) (ProbeResult, error) {
	var resp ffprobeResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return ProbeResult{}, fmt.Errorf("%w: invalid ffprobe JSON", ErrInvalidSource)
	}
	audio := make([]probeStream, 0, len(resp.Streams))
	for _, stream := range resp.Streams {
		if stream.CodecType == "audio" {
			if stream.Index == nil || *stream.Index < 0 {
				return ProbeResult{}, fmt.Errorf("%w: invalid audio stream index", ErrInvalidSource)
			}
			audio = append(audio, stream)
		}
	}
	if len(audio) == 0 {
		return ProbeResult{}, ErrNoAudio
	}
	sort.Slice(audio, func(i, j int) bool { return *audio[i].Index < *audio[j].Index })
	chosen := audio[0]

	durationText := strings.TrimSpace(chosen.Duration)
	if durationText == "" || durationText == "N/A" {
		durationText = resp.Format.Duration
	}
	duration, err := parsePositiveDuration(durationText)
	if err != nil || duration <= 0 || duration > maxDuration {
		return ProbeResult{}, ErrDurationOutOfRange
	}
	rate, err := parsePositiveInt(chosen.SampleRate)
	if err != nil || rate > 768000 {
		return ProbeResult{}, fmt.Errorf("%w: invalid sample rate", ErrInvalidSource)
	}
	if chosen.Channels <= 0 || chosen.Channels > 64 || chosen.CodecName == "" {
		return ProbeResult{}, fmt.Errorf("%w: invalid channel count", ErrInvalidSource)
	}
	samples, err := durationToSamples(duration, SampleRate(rate))
	if err != nil || samples <= 0 {
		return ProbeResult{}, ErrDurationOutOfRange
	}
	return ProbeResult{
		Path:        path,
		SizeBytes:   size,
		StreamIndex: *chosen.Index,
		Duration:    duration,
		Format: AudioFormat{
			Container:     container,
			Encoding:      chosen.CodecName,
			SampleRate:    SampleRate(rate),
			Channels:      chosen.Channels,
			BitsPerSample: chosen.BitsPerSample,
		},
		Timeline: Timeline{
			SampleRate: SampleRate(rate),
			Samples:    SampleCount(samples),
		},
	}, nil
}

func parsePositiveDuration(s string) (time.Duration, error) {
	if strings.TrimSpace(s) == "" {
		return 0, fmt.Errorf("missing duration")
	}
	seconds, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil || !isFinitePositive(seconds) {
		return 0, fmt.Errorf("invalid duration")
	}
	if seconds >= float64(maxInt64)/float64(time.Second) {
		return 0, fmt.Errorf("duration overflow")
	}
	d := time.Duration(seconds * float64(time.Second))
	if d <= 0 {
		return 0, fmt.Errorf("non-positive duration")
	}
	return d, nil
}

func parsePositiveInt(s string) (int, error) {
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("invalid integer")
	}
	return v, nil
}

func durationToSamples(d time.Duration, rate SampleRate) (int64, error) {
	if d <= 0 || rate <= 0 {
		return 0, fmt.Errorf("invalid sample timeline")
	}
	samples, ok := multiplyDivide(int64(d), int64(rate), int64(time.Second))
	if !ok {
		return 0, fmt.Errorf("sample overflow")
	}
	if samples <= 0 {
		return 0, fmt.Errorf("sample underflow")
	}
	return samples, nil
}

func isFinitePositive(v float64) bool {
	return v > 0 && !math.IsInf(v, 0) && !math.IsNaN(v)
}

func buildProbeArgs(srcPath string, kind sourceKind) []string {
	args := []string{
		"-hide_banner",
		"-loglevel", "error",
		"-protocol_whitelist", "file,pipe",
		"-f", kind.demuxer,
	}
	if kind.disableExternalRef {
		args = append(args, "-enable_drefs", "0", "-use_absolute_path", "0")
	}
	args = append(args,
		"-i", srcPath,
		"-select_streams", "a",
		"-show_entries", "format=duration:stream=index,codec_type,codec_name,duration,sample_rate,channels,bits_per_sample",
		"-of", "json",
	)
	return args
}

func buildDecodeArgs(probe ProbeResult, srcPath, dstPath string, maxBytes int64) []string {
	kind := sourceKind{demuxer: probe.Format.Container, disableExternalRef: probe.Format.Container == "mov"}
	args := []string{
		"-hide_banner",
		"-nostdin",
		"-loglevel", "error",
		"-protocol_whitelist", "file,pipe",
		"-f", kind.demuxer,
	}
	if kind.disableExternalRef {
		args = append(args, "-enable_drefs", "0", "-use_absolute_path", "0")
	}
	args = append(args,
		"-threads", "2",
		"-i", srcPath,
		"-map", fmt.Sprintf("0:%d", probe.StreamIndex),
		"-vn",
		"-sn",
		"-dn",
		"-map_metadata", "-1",
		"-map_chapters", "-1",
		"-ac", strconv.Itoa(CanonicalChannels),
		"-ar", strconv.Itoa(int(CanonicalSampleRate)),
		"-acodec", "pcm_s16le",
		// Coarse subprocess ceiling; reaching it is rejected by strict postvalidation.
		"-fs", strconv.FormatInt(maxBytes, 10),
		"-threads", "2", "-f", "wav",
		"-y", dstPath,
	)
	return args
}
