package media

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	go264audio "github.com/rcarmo/go-264/audio"
	go264mp4 "github.com/rcarmo/go-264/audio/mp4"
	go264pcm "github.com/rcarmo/go-264/audio/pcm"
)

const canonicalWAVHeaderBytes = 44

// Go264Config configures the optional pure-Go audio backend.
type Go264Config struct {
	MaxInputBytes        int64
	MaxDuration          time.Duration
	MaxDecodeOutputBytes int64
}

// Go264 decodes a narrow, explicit WAV/MP4 audio subset without subprocesses.
type Go264 struct {
	cfg Go264Config
}

var _ Adapter = (*Go264)(nil)

// NewGo264 selects the optional provider explicitly; NewFFmpeg/defaults are
// unchanged. Inputs are immutable regular files in caller-owned directories.
func NewGo264(cfg Go264Config) (*Go264, error) {
	if cfg.MaxInputBytes < 0 || cfg.MaxDuration < 0 || cfg.MaxDecodeOutputBytes < 0 {
		return nil, fmt.Errorf("negative media limits")
	}
	if cfg.MaxInputBytes <= 0 {
		cfg.MaxInputBytes = DefaultMaxInputBytes
	}
	if cfg.MaxDuration <= 0 {
		cfg.MaxDuration = DefaultMaxDuration
	}
	if cfg.MaxDuration > DefaultMaxDuration {
		return nil, fmt.Errorf("media limits exceed adapter bounds")
	}
	if cfg.MaxDecodeOutputBytes <= 0 {
		cfg.MaxDecodeOutputBytes = canonicalMaxWAVBytes(cfg.MaxDuration)
	}
	if cfg.MaxInputBytes <= 0 || cfg.MaxDuration <= 0 || cfg.MaxDecodeOutputBytes <= 0 || cfg.MaxDecodeOutputBytes > canonicalMaxWAVBytes(cfg.MaxDuration) {
		return nil, fmt.Errorf("invalid media limits")
	}
	return &Go264{cfg: cfg}, nil
}

func (g *Go264) Probe(ctx context.Context, path string) (ProbeResult, error) {
	if err := contextError(ctx); err != nil {
		return ProbeResult{}, err
	}
	src, err := openGo264Source(path, g.cfg.MaxInputBytes)
	if err != nil {
		return ProbeResult{}, err
	}
	defer src.Close()

	info, err := go264audio.Probe(ctx, src.file, src.size, g.providerLimits())
	if err != nil {
		return ProbeResult{}, mapGo264SourceError(err)
	}
	timeline, duration, err := validateGo264SourceInfo(info, g.cfg.MaxDuration)
	if err != nil {
		return ProbeResult{}, err
	}
	streamIndex, timing, err := g.probeSourceTiming(ctx, src, info)
	if err != nil {
		return ProbeResult{}, err
	}
	return ProbeResult{
		Path:        src.path,
		SizeBytes:   src.size,
		StreamIndex: streamIndex,
		Duration:    duration,
		Format: AudioFormat{
			Container:     src.kind.container,
			Encoding:      go264Encoding(src.kind.container, info.BitsPerSample),
			SampleRate:    SampleRate(info.SampleRate),
			Channels:      info.Channels,
			BitsPerSample: info.BitsPerSample,
		},
		Timeline: timeline,
		Source:   timing,
	}, nil
}

func (g *Go264) DecodeToFile(ctx context.Context, srcPath, dstPath string) (DecodeResult, error) {
	if err := contextError(ctx); err != nil {
		return DecodeResult{}, err
	}
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

	src, err := openGo264Source(srcAbs, g.cfg.MaxInputBytes)
	if err != nil {
		return DecodeResult{}, err
	}
	defer src.Close()

	info, err := go264audio.Probe(ctx, src.file, src.size, g.providerLimits())
	if err != nil {
		return DecodeResult{}, mapGo264SourceError(err)
	}
	_, duration, err := validateGo264SourceInfo(info, g.cfg.MaxDuration)
	if err != nil {
		return DecodeResult{}, err
	}
	_, sourceTiming, err := g.probeSourceTiming(ctx, src, info)
	if err != nil {
		return DecodeResult{}, err
	}
	if sourceTiming.Duration == 0 {
		sourceTiming.Duration = duration
	}

	dec, err := go264audio.Open(ctx, src.file, src.size, go264audio.Options{TargetRate: int(CanonicalSampleRate), TargetChannels: CanonicalChannels, Limits: g.providerLimits()})
	if err != nil {
		return DecodeResult{}, mapGo264SourceError(err)
	}
	defer dec.Close()

	meta := dec.Metadata()
	if meta.Output.SampleRate != int(CanonicalSampleRate) || meta.Output.Channels != CanonicalChannels || meta.Output.BitsPerSample != CanonicalBitsPerSample || meta.Output.Frames <= 0 {
		return DecodeResult{}, fmt.Errorf("%w: invalid provider output metadata", ErrInvalidOutput)
	}
	if meta.PrimingFrames < 0 || meta.PaddingFrames < 0 || meta.LeadingSilenceFrames < 0 {
		return DecodeResult{}, fmt.Errorf("%w: invalid provider source timing", ErrInvalidOutput)
	}
	sourceTiming.SourceRate = SampleRate(meta.Source.SampleRate)
	sourceTiming.Priming = SampleCount(meta.PrimingFrames)
	sourceTiming.Padding = SampleCount(meta.PaddingFrames)
	sourceTiming.LeadingSilence = SampleCount(meta.LeadingSilenceFrames)
	sourceTiming.HasEdits = sourceTiming.HasEdits || meta.LeadingSilenceFrames > 0
	if sourceTiming.Duration <= 0 {
		return DecodeResult{}, fmt.Errorf("%w: invalid provider source duration", ErrInvalidOutput)
	}
	maxFrames := maxFramesForDuration(g.cfg.MaxDuration)
	if meta.Output.Frames > maxFrames {
		return DecodeResult{}, ErrDurationOutOfRange
	}
	if expectedSize, ok := canonicalWAVSizeForFrames(meta.Output.Frames); !ok {
		return DecodeResult{}, fmt.Errorf("%w: canonical wav size overflow", ErrInvalidOutput)
	} else if expectedSize > g.cfg.MaxDecodeOutputBytes {
		return DecodeResult{}, ErrDecodeOutputLimit
	}

	if err := os.MkdirAll(filepath.Dir(dstAbs), 0o700); err != nil {
		return DecodeResult{}, fmt.Errorf("ensure destination directory: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dstAbs), "."+filepath.Base(dstAbs)+".tmp-*.wav")
	if err != nil {
		return DecodeResult{}, fmt.Errorf("create temp output: %w", err)
	}
	tmpPath := tmp.Name()
	removeTmp := true
	defer func() {
		if tmp != nil {
			_ = tmp.Close()
		}
		if removeTmp {
			_ = os.Remove(tmpPath)
		}
	}()

	if err := writeCanonicalWAVHeader(tmp, 0); err != nil {
		return DecodeResult{}, fmt.Errorf("write wav header: %w", err)
	}

	pcmBuf := make([]int16, 2048)
	byteBuf := make([]byte, len(pcmBuf)*2)
	var framesWritten int64
	for {
		if err := contextError(ctx); err != nil {
			return DecodeResult{}, err
		}
		n, _, readErr := dec.ReadPCM(ctx, pcmBuf)
		if n < 0 || n > len(pcmBuf) {
			return DecodeResult{}, fmt.Errorf("%w: invalid provider sample count", ErrInvalidOutput)
		}
		if n > 0 {
			chunkFrames := int64(n)
			if framesWritten > maxFrames-chunkFrames {
				return DecodeResult{}, ErrDurationOutOfRange
			}
			nextSize, ok := canonicalWAVSizeForFrames(framesWritten + chunkFrames)
			if !ok {
				return DecodeResult{}, fmt.Errorf("%w: canonical wav size overflow", ErrInvalidOutput)
			}
			if nextSize > g.cfg.MaxDecodeOutputBytes {
				return DecodeResult{}, ErrDecodeOutputLimit
			}
			for i := 0; i < n; i++ {
				binary.LittleEndian.PutUint16(byteBuf[2*i:2*i+2], uint16(pcmBuf[i]))
			}
			if _, err := tmp.Write(byteBuf[:2*n]); err != nil {
				return DecodeResult{}, fmt.Errorf("write decoded wav data: %w", err)
			}
			framesWritten += chunkFrames
		}
		if readErr == nil {
			if n == 0 {
				return DecodeResult{}, fmt.Errorf("%w: no progress while decoding", ErrInvalidOutput)
			}
			continue
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if err := contextError(ctx); err != nil {
			return DecodeResult{}, err
		}
		return DecodeResult{}, mapGo264ReadError(readErr)
	}
	if err := contextError(ctx); err != nil {
		return DecodeResult{}, err
	}
	if framesWritten != meta.Output.Frames {
		return DecodeResult{}, fmt.Errorf("%w: provider declared/emitted extent mismatch", ErrInvalidOutput)
	}
	if framesWritten <= 0 {
		return DecodeResult{}, fmt.Errorf("%w: empty decoded audio", ErrInvalidOutput)
	}
	dataBytes := framesWritten * 2
	if dataBytes > int64(^uint32(0))-36 {
		return DecodeResult{}, fmt.Errorf("%w: wav data too large", ErrInvalidOutput)
	}
	if err := writeCanonicalWAVHeaderAt(tmp, uint32(dataBytes)); err != nil {
		return DecodeResult{}, fmt.Errorf("update wav header: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return DecodeResult{}, fmt.Errorf("close temp output: %w", err)
	}
	tmp = nil

	if _, err := os.Lstat(dstAbs); err == nil {
		return DecodeResult{}, fmt.Errorf("destination already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return DecodeResult{}, fmt.Errorf("stat destination: %w", err)
	}

	validated, err := validateCanonicalWAV(tmpPath, g.cfg.MaxDecodeOutputBytes, maxFrames)
	if err != nil {
		return DecodeResult{}, err
	}
	if validated.frames != framesWritten {
		return DecodeResult{}, fmt.Errorf("%w: decoded frame count mismatch", ErrInvalidOutput)
	}
	if err := contextError(ctx); err != nil {
		return DecodeResult{}, err
	}
	if err := renameNoReplace(tmpPath, dstAbs); err != nil {
		return DecodeResult{}, fmt.Errorf("publish decoded wav: %w", err)
	}
	removeTmp = false
	return DecodeResult{
		Path:      dstAbs,
		SizeBytes: validated.fileSize,
		Format:    canonicalWAV(),
		Timeline:  Timeline{SampleRate: CanonicalSampleRate, Samples: SampleCount(validated.frames)},
		Source:    sourceTiming,
	}, nil
}

type go264Source struct {
	file *os.File
	path string
	kind sourceKind
	size int64
}

func (s go264Source) Close() error {
	if s.file == nil {
		return nil
	}
	return s.file.Close()
}

func openGo264Source(path string, maxBytes int64) (go264Source, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return go264Source{}, fmt.Errorf("resolve source path: %w", err)
	}
	// Avoid blocking on obvious FIFOs/devices before open. The opened inode
	// is checked again; caller owns the private directory throughout.
	st, err := os.Stat(absPath)
	if err != nil {
		return go264Source{}, fmt.Errorf("%w: source stat", ErrInvalidSource)
	}
	if !st.Mode().IsRegular() {
		return go264Source{}, fmt.Errorf("%w: source must be regular", ErrInvalidSource)
	}
	f, err := os.Open(absPath)
	if err != nil {
		return go264Source{}, fmt.Errorf("open source: %w", err)
	}
	fi, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return go264Source{}, fmt.Errorf("stat source: %w", err)
	}
	if !fi.Mode().IsRegular() {
		_ = f.Close()
		return go264Source{}, fmt.Errorf("%w: source must be a regular file", ErrInvalidSource)
	}
	if fi.Size() <= 0 {
		_ = f.Close()
		return go264Source{}, fmt.Errorf("%w: empty file", ErrInvalidSource)
	}
	if maxBytes > 0 && fi.Size() > maxBytes {
		_ = f.Close()
		return go264Source{}, fmt.Errorf("%w: %d bytes", ErrSizeLimit, fi.Size())
	}
	kind, err := sniffSourceFile(f, absPath)
	if err != nil {
		_ = f.Close()
		return go264Source{}, err
	}
	return go264Source{file: f, path: absPath, kind: kind, size: fi.Size()}, nil
}

func sniffSourceFile(f *os.File, path string) (sourceKind, error) {
	ext := strings.ToLower(filepath.Ext(path))
	buf := make([]byte, 32)
	n, err := f.ReadAt(buf, 0)
	if err != nil && !errors.Is(err, io.EOF) {
		return sourceKind{}, fmt.Errorf("read source header: %w", err)
	}
	buf = buf[:n]
	switch ext {
	case ".wav":
		if len(buf) < 12 || string(buf[0:4]) != "RIFF" || string(buf[8:12]) != "WAVE" {
			return sourceKind{}, fmt.Errorf("%w: expected RIFF/WAVE content", ErrUnsupportedInput)
		}
		return sourceKind{container: "wav"}, nil
	case ".m4a", ".mp4", ".mov":
		if len(buf) < 12 || string(buf[4:8]) != "ftyp" {
			return sourceKind{}, fmt.Errorf("%w: expected ISO BMFF content", ErrUnsupportedInput)
		}
		return sourceKind{container: "mov"}, nil
	default:
		return sourceKind{}, fmt.Errorf("%w: unsupported extension %q", ErrUnsupportedInput, ext)
	}
}

func (g *Go264) providerLimits() go264pcm.Limits {
	return go264pcm.Limits{
		MaxBytes:           g.cfg.MaxInputBytes,
		MaxDurationSeconds: int64(DefaultMaxDuration / time.Second),
		MaxChunks:          4096,
	}
}

func validateGo264SourceInfo(info go264pcm.Info, maxDuration time.Duration) (Timeline, time.Duration, error) {
	if info.SampleRate <= 0 || info.Channels <= 0 || info.Frames <= 0 {
		return Timeline{}, 0, fmt.Errorf("%w: invalid provider metadata", ErrInvalidSource)
	}
	maxFrames, ok := multiplyDivide(int64(maxDuration), int64(info.SampleRate), int64(time.Second))
	if !ok {
		return Timeline{}, 0, fmt.Errorf("%w: duration overflow", ErrInvalidSource)
	}
	if info.Frames > maxFrames {
		return Timeline{}, 0, ErrDurationOutOfRange
	}
	timeline := Timeline{SampleRate: SampleRate(info.SampleRate), Samples: SampleCount(info.Frames)}
	if !timeline.Valid() {
		return Timeline{}, 0, fmt.Errorf("%w: invalid source timeline", ErrInvalidSource)
	}
	return timeline, timeline.Duration(), nil
}

func (g *Go264) probeSourceTiming(ctx context.Context, src go264Source, info go264pcm.Info) (int, SourceTiming, error) {
	duration := (Timeline{SampleRate: SampleRate(info.SampleRate), Samples: SampleCount(info.Frames)}).Duration()
	if src.kind.container != "mov" {
		return 0, SourceTiming{Duration: duration, Exact: true, SourceRate: SampleRate(info.SampleRate)}, nil
	}
	r, err := go264mp4.Open(ctx, src.file, src.size, go264mp4.Limits{MaxBytes: g.cfg.MaxInputBytes, MaxDurationSeconds: int64(DefaultMaxDuration / time.Second), MaxBoxes: 4096})
	if err != nil {
		return 0, SourceTiming{}, mapGo264SourceError(err)
	}
	track := r.Track()
	if !track.Accepted || track.Index < 0 || track.SampleRate != info.SampleRate || track.Channels != info.Channels || track.SampleCount <= 0 {
		return 0, SourceTiming{}, fmt.Errorf("%w: invalid selected MP4 track metadata", ErrInvalidSource)
	}
	plan, err := track.TimingPlan(info.Frames)
	if err != nil {
		return 0, SourceTiming{}, mapGo264SourceError(err)
	}
	outputDuration := (Timeline{SampleRate: SampleRate(info.SampleRate), Samples: SampleCount(plan.OutputFrames)}).Duration()
	return track.Index, SourceTiming{Duration: outputDuration, Exact: true, HasEdits: plan.HasEdit, SourceRate: SampleRate(info.SampleRate), Priming: SampleCount(plan.PrimingFrames), Padding: SampleCount(plan.PaddingFrames), LeadingSilence: SampleCount(plan.LeadingSilenceFrames)}, nil
}

func go264Encoding(container string, bitsPerSample int) string {
	switch container {
	case "mov":
		return "aac_lc"
	case "wav":
		switch bitsPerSample {
		case 8:
			return "pcm_u8"
		case 16:
			return "pcm_s16le"
		case 24:
			return "pcm_s24le"
		case 32:
			return "pcm_s32le"
		default:
			return "pcm"
		}
	default:
		return ""
	}
}

func mapGo264SourceError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	switch {
	case errors.Is(err, go264pcm.ErrUnsupported):
		return ErrUnsupportedInput
	case errors.Is(err, go264pcm.ErrMalformed), errors.Is(err, go264pcm.ErrLimit), errors.Is(err, go264pcm.ErrClosed):
		return ErrInvalidSource
	default:
		return ErrInvalidSource
	}
}

func mapGo264ReadError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	switch {
	case errors.Is(err, go264pcm.ErrUnsupported):
		return ErrUnsupportedInput
	case errors.Is(err, go264pcm.ErrMalformed), errors.Is(err, go264pcm.ErrLimit), errors.Is(err, go264pcm.ErrClosed), errors.Is(err, io.ErrNoProgress):
		return ErrInvalidOutput
	default:
		return ErrInvalidOutput
	}
}

func canonicalWAVSizeForFrames(frames int64) (int64, bool) {
	if frames < 0 || frames > (maxInt64-canonicalWAVHeaderBytes)/2 {
		return 0, false
	}
	return canonicalWAVHeaderBytes + 2*frames, true
}

func writeCanonicalWAVHeader(f *os.File, dataBytes uint32) error {
	header := canonicalWAVHeader(dataBytes)
	_, err := f.Write(header[:])
	return err
}

func writeCanonicalWAVHeaderAt(f *os.File, dataBytes uint32) error {
	header := canonicalWAVHeader(dataBytes)
	_, err := f.WriteAt(header[:], 0)
	return err
}

func canonicalWAVHeader(dataBytes uint32) [canonicalWAVHeaderBytes]byte {
	var header [canonicalWAVHeaderBytes]byte
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], 36+dataBytes)
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1)
	binary.LittleEndian.PutUint16(header[22:24], CanonicalChannels)
	binary.LittleEndian.PutUint32(header[24:28], uint32(CanonicalSampleRate))
	binary.LittleEndian.PutUint32(header[28:32], uint32(int(CanonicalSampleRate)*CanonicalChannels*(CanonicalBitsPerSample/8)))
	binary.LittleEndian.PutUint16(header[32:34], uint16(CanonicalChannels*(CanonicalBitsPerSample/8)))
	binary.LittleEndian.PutUint16(header[34:36], CanonicalBitsPerSample)
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], dataBytes)
	return header
}

func contextError(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	return ctx.Err()
}
