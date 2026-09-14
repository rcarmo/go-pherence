// Package media probes local audio files and decodes them to a canonical speech
// WAV suitable for downstream loaders.
package media

import (
	"context"
	"errors"
	"io"
	"math/bits"
	"time"
)

const (
	CanonicalSampleRate    SampleRate = 16000
	CanonicalChannels                 = 1
	CanonicalBitsPerSample            = 16

	DefaultMaxInputBytes        int64         = 512 << 20
	DefaultMaxProbeStdoutBytes  int64         = 256 << 10
	DefaultMaxProbeStderrBytes  int64         = 16 << 10
	DefaultMaxDecodeStderrBytes int64         = 32 << 10
	DefaultMaxDuration          time.Duration = 4 * time.Hour
	maxInt64                                  = int64(^uint64(0) >> 1)
)

var (
	ErrInvalidSource      = errors.New("invalid media source")
	ErrUnsupportedInput   = errors.New("unsupported media input")
	ErrNoAudio            = errors.New("media has no audio stream")
	ErrDurationOutOfRange = errors.New("media duration out of range")
	ErrSizeLimit          = errors.New("media exceeds size limit")
	ErrDecodeOutputLimit  = errors.New("decoded output exceeds limit")
	ErrInvalidOutput      = errors.New("invalid decoded WAV output")
)

// SampleRate is an audio sample rate in Hz.
type SampleRate int

// SampleCount is a frame count at a specific sample rate.
type SampleCount int64

// Timeline identifies a concrete audio span in whole samples.
type Timeline struct {
	SampleRate SampleRate
	Samples    SampleCount
}

// SourceTiming records how canonical PCM maps onto the selected source stream.
// Start is the source timestamp corresponding to canonical sample zero. It is
// explicit even when zero. Exact=false means the adapter cannot prove edit-list/
// priming/padding mapping and callers must not claim source-time alignment.
// Priming/Padding/LeadingSilence use SourceRate frames.
type SourceTiming struct {
	Start          time.Duration `json:"start_ns"`
	Duration       time.Duration `json:"duration_ns"`
	Exact          bool          `json:"exact"`
	HasEdits       bool          `json:"has_edits"`
	SourceRate     SampleRate    `json:"source_rate"`
	Priming        SampleCount   `json:"priming"`
	Padding        SampleCount   `json:"padding"`
	LeadingSilence SampleCount   `json:"leading_silence"`
}

// Valid reports whether the timeline has a positive sample rate and a
// non-negative sample count.
func (t Timeline) Valid() bool {
	return t.SampleRate > 0 && t.Samples >= 0
}

// Duration returns the truncated duration, saturated at time.Duration maximum.
// Sample counts, not nanoseconds, are the authoritative timeline.
func (t Timeline) Duration() time.Duration {
	if t.SampleRate <= 0 || t.Samples <= 0 {
		return 0
	}
	ns, ok := multiplyDivide(int64(t.Samples), int64(time.Second), int64(t.SampleRate))
	if !ok {
		return time.Duration(maxInt64)
	}
	return time.Duration(ns)
}

// AudioFormat describes an audio stream or decoded output contract.
type AudioFormat struct {
	Container     string
	Encoding      string
	SampleRate    SampleRate
	Channels      int
	BitsPerSample int
}

// canonicalWAV returns the immutable decode format by value.
func canonicalWAV() AudioFormat {
	return AudioFormat{
		Container:     "wav",
		Encoding:      "pcm_s16le",
		SampleRate:    CanonicalSampleRate,
		Channels:      CanonicalChannels,
		BitsPerSample: CanonicalBitsPerSample,
	}
}

// ProbeResult describes the selected source audio stream. Timeline.Samples is
// estimated from reported duration; only DecodeResult counts actual PCM frames.
type ProbeResult struct {
	Path        string
	SizeBytes   int64
	StreamIndex int
	Duration    time.Duration
	Format      AudioFormat
	Timeline    Timeline
	Source      SourceTiming
}

// DecodeResult describes a successfully written canonical WAV file.
type DecodeResult struct {
	Path      string
	SizeBytes int64
	Format    AudioFormat
	Timeline  Timeline
	Source    SourceTiming
}

// Command describes a single external process invocation.
type Command struct {
	Path   string
	Args   []string
	Stdout io.Writer
	Stderr io.Writer
}

// Runner executes and waits for an owned command, honouring context cancellation.
// Tests may inject trusted runners; custom runners must enforce the same contract.
type Runner interface {
	Run(ctx context.Context, cmd Command) error
}

// Adapter is the minimal audio-media contract shared by future backends.
type Adapter interface {
	Probe(ctx context.Context, path string) (ProbeResult, error)
	DecodeToFile(ctx context.Context, srcPath, dstPath string) (DecodeResult, error)
}

// Config configures the temporary FFmpeg/ffprobe adapter.
type Config struct {
	FFprobePath string
	FFmpegPath  string
	Runner      Runner

	MaxInputBytes        int64
	MaxDuration          time.Duration
	MaxProbeStdoutBytes  int64
	MaxProbeStderrBytes  int64
	MaxDecodeStderrBytes int64
	MaxDecodeOutputBytes int64
}

// multiplyDivide computes floor(a*b/d) using a 128-bit intermediate.
func multiplyDivide(a, b, d int64) (int64, bool) {
	if a < 0 || b < 0 || d <= 0 {
		return 0, false
	}
	hi, lo := bits.Mul64(uint64(a), uint64(b))
	if hi >= uint64(d) {
		return 0, false
	}
	q, _ := bits.Div64(hi, lo, uint64(d))
	return int64(q), q <= uint64(maxInt64)
}
