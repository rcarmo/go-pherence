package speechjob

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

// FFmpegDecodeConfig is the complete decode-stage identity. Executable hashes
// are required and checked before each execution; a detected change fails closed.
// The binaries/parent paths MUST be administrator-controlled and immutable for
// the entire job: execution is by path after hashing, not from a pinned inode.
// This check does not prevent a concurrent path/symlink replacement attack.
// Hashing the executables does not pin their shared libraries/OS: include that
// environment identity in the job's configuration too. Paths must be absolute.
// InputExtension is one of .wav/.m4a/.mp4/.mov and must describe the upload type.
// The display name never becomes a filesystem path. All limits are positive.
// MaxOutputBytes bounds the adapter WAV including its metadata, not just PCM.
type FFmpegDecodeConfig struct {
	FFmpegPath, FFprobePath       string
	FFmpegSHA256, FFprobeSHA256   string
	InputExtension                string
	MaxInputBytes, MaxOutputBytes int64
	MaxDuration                   time.Duration
}

// NewFFmpegDecodeStage constructs an explicit temporary-FFmpeg "decode" stage.
// No executable runs at construction. It verifies upload and output caps,
// uses job-owned private scratch, revalidates actual decoded PCM extents, and
// publishes only a fixed-header mono16k S16 WAV through Store.Run's writer.
// Scratch is removed on normal exit/error, retained and quota-accounted after
// process death. Stage admission reserves configured input+scratch+checkpoint
// file caps. FFmpeg's coarse -fs/monitor may overshoot during buffering; this
// reservation does not impose subprocess memory/CPU or hard OS disk quotas.
func NewFFmpegDecodeStage(cfg FFmpegDecodeConfig) (Stage, error) {
	if !filepath.IsAbs(cfg.FFmpegPath) || !filepath.IsAbs(cfg.FFprobePath) || !validHash(cfg.FFmpegSHA256) || !validHash(cfg.FFprobeSHA256) {
		return Stage{}, fmt.Errorf("decode requires absolute paths and executable SHA256s")
	}
	if cfg.MaxInputBytes < 1 || cfg.MaxInputBytes > media.DefaultMaxInputBytes || cfg.MaxOutputBytes < 46 || cfg.MaxOutputBytes > 8<<30 || cfg.MaxDuration < time.Second/16000 || cfg.MaxDuration > media.DefaultMaxDuration {
		return Stage{}, fmt.Errorf("invalid decode limits")
	}
	switch cfg.InputExtension {
	case ".wav", ".m4a", ".mp4", ".mov":
	default:
		return Stage{}, media.ErrUnsupportedInput
	}
	adapter, e := media.NewFFmpeg(media.Config{FFmpegPath: cfg.FFmpegPath, FFprobePath: cfg.FFprobePath, MaxInputBytes: cfg.MaxInputBytes, MaxDecodeOutputBytes: cfg.MaxOutputBytes, MaxDuration: cfg.MaxDuration})
	if e != nil {
		return Stage{}, e
	}
	return newDecodeStage(cfg, adapter), nil
}

// Internal injection is only for contract tests. No hidden alternative backend.
func newDecodeStage(cfg FFmpegDecodeConfig, adapter media.Adapter) Stage {
	identity, _ := json.Marshal(struct {
		Schema string
		Config FFmpegDecodeConfig
	}{"ffmpeg-canonical-wav-v1", cfg})
	return Stage{Name: "decode", Version: hash(identity), Run: func(ctx context.Context, in *Input, out io.Writer) (err error) {
		if e := ctx.Err(); e != nil {
			return e
		}
		if in.job.Input.Bytes > cfg.MaxInputBytes {
			return media.ErrSizeLimit
		}
		for _, b := range []struct{ path, digest string }{{cfg.FFmpegPath, cfg.FFmpegSHA256}, {cfg.FFprobePath, cfg.FFprobeSHA256}} {
			if e := verifyExecutable(ctx, b.path, b.digest); e != nil {
				return e
			}
		}
		used, _, e := in.store.usage()
		if e != nil {
			return e
		}
		// input materialisation + maximum adapter output + maximum published payload
		// + manifest headroom. Validate by subtraction to avoid integer overflow.
		remaining := in.store.limits.MaxBytes - used
		for _, needed := range []int64{in.job.Input.Bytes, cfg.MaxOutputBytes, cfg.MaxOutputBytes, maxManifest} {
			if needed > remaining {
				return ErrLimit
			}
			remaining -= needed
		}
		if cfg.MaxOutputBytes > in.store.limits.MaxArtifactBytes {
			return ErrLimit
		}
		token, e := token()
		if e != nil {
			return e
		}
		relative := in.job.ID + "/.work-decode-" + token
		if e = in.store.root.Mkdir(relative, 0700); e != nil {
			return e
		}
		defer func() { err = errors.Join(err, in.store.root.RemoveAll(relative)) }()
		workspace := filepath.Join(in.store.root.Name(), filepath.FromSlash(relative))
		inputPath, outputPath := filepath.Join(workspace, "input"+cfg.InputExtension), filepath.Join(workspace, "output.wav")
		source, e := in.OpenSource(ctx)
		if e != nil {
			return e
		}
		input, e := os.OpenFile(inputPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if e != nil {
			source.Close()
			return e
		}
		_, copyErr := copyExact(ctx, input, source, in.job.Input.Bytes)
		if e = errors.Join(copyErr, source.Close(), input.Close()); e != nil {
			return e
		}
		if e = in.store.hit("decode-input-ready"); e != nil {
			return e
		}
		decoded, e := adapter.DecodeToFile(ctx, inputPath, outputPath)
		if e != nil {
			return e
		}
		if e = ctx.Err(); e != nil {
			return e
		}
		if decoded.Path != outputPath {
			return fmt.Errorf("%w: adapter changed output path", media.ErrInvalidOutput)
		}
		st, e := os.Lstat(outputPath)
		if e != nil {
			return e
		}
		if !st.Mode().IsRegular() || st.Size() < 44 || st.Size() > cfg.MaxOutputBytes || decoded.SizeBytes != st.Size() {
			return media.ErrInvalidOutput
		}
		pcm, e := media.OpenCanonicalPCM(ctx, outputPath)
		if e != nil {
			return e
		}
		defer func() { err = errors.Join(err, pcm.Close()) }()
		timeline := pcm.Timeline()
		expected := media.AudioFormat{Container: "wav", Encoding: "pcm_s16le", SampleRate: 16000, Channels: 1, BitsPerSample: 16}
		maxSamples := int64(cfg.MaxDuration)/int64(time.Second)*16000 + (int64(cfg.MaxDuration)%int64(time.Second))*16000/int64(time.Second)
		if decoded.Format != expected || decoded.Timeline != timeline || timeline.Samples <= 0 || int64(timeline.Samples) > maxSamples || decoded.Source.Start < 0 || decoded.Source.Duration < 0 || decoded.Source.SourceRate < 0 || decoded.Source.Priming < 0 || decoded.Source.Padding < 0 || decoded.Source.LeadingSilence < 0 {
			return media.ErrInvalidOutput
		}
		if e = in.store.hit("decode-output-ready"); e != nil {
			return e
		}
		return writeCanonicalCheckpoint(ctx, out, pcm, int64(timeline.Samples), cfg.MaxOutputBytes)
	}}
}
func verifyExecutable(ctx context.Context, path, expected string) error {
	if e := ctx.Err(); e != nil {
		return e
	}
	f, e := os.Open(path)
	if e != nil {
		return e
	}
	defer f.Close()
	st, e := f.Stat()
	if e != nil {
		return e
	}
	if !st.Mode().IsRegular() || st.Size() > 1<<30 {
		return ErrConfiguration
	}
	digest := sha256.New()
	buf := make([]byte, 32<<10)
	for {
		if e = ctx.Err(); e != nil {
			return e
		}
		n, re := f.Read(buf)
		if n > 0 {
			digest.Write(buf[:n])
		}
		if re == io.EOF {
			break
		}
		if re != nil {
			return re
		}
	}
	if hex.EncodeToString(digest.Sum(nil)) != expected {
		return fmt.Errorf("%w: decoder executable changed", ErrConfiguration)
	}
	return nil
}

// copyExact never pads a short input or accepts an unaccounted extra byte.
// The reader remains caller-owned; blocking IO needs external cancellation.
func copyExact(ctx context.Context, dst io.Writer, src io.Reader, size int64) (int64, error) {
	if size < 0 {
		return 0, ErrLimit
	}
	var copied int64
	buf := make([]byte, 32<<10)
	for copied < size {
		if e := ctx.Err(); e != nil {
			return copied, e
		}
		n := int(min(int64(len(buf)), size-copied))
		nr, re := io.ReadFull(src, buf[:n])
		if nr > 0 {
			nw, we := dst.Write(buf[:nr])
			copied += int64(nw)
			if we != nil {
				return copied, we
			}
			if nw != nr {
				return copied, io.ErrShortWrite
			}
		}
		if re != nil {
			return copied, re
		}
	}
	if e := ctx.Err(); e != nil {
		return copied, e
	}
	var extra [1]byte
	n, e := io.ReadFull(src, extra[:])
	if n != 0 {
		return copied, ErrLimit
	}
	if e != io.EOF {
		return copied, e
	}
	return copied, nil
}
func writeFull(dst io.Writer, b []byte) error {
	n, e := dst.Write(b)
	if e == nil && n != len(b) {
		e = io.ErrShortWrite
	}
	return e
}

// PCMReader values are exact int16/32768; multiplication by32768 is exact.
// Re-encoding is lossless and removes nondeterministic container metadata.
func writeCanonicalCheckpoint(ctx context.Context, out io.Writer, pcm *media.PCMReader, samples, maxBytes int64) error {
	if samples < 1 || samples > (int64(^uint32(0))-36)/2 || 44+samples*2 > maxBytes {
		return media.ErrDecodeOutputLimit
	}
	var header [44]byte
	copy(header[:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(36+samples*2))
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1)
	binary.LittleEndian.PutUint16(header[22:24], 1)
	binary.LittleEndian.PutUint32(header[24:28], 16000)
	binary.LittleEndian.PutUint32(header[28:32], 32000)
	binary.LittleEndian.PutUint16(header[32:34], 2)
	binary.LittleEndian.PutUint16(header[34:36], 16)
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(samples*2))
	if e := ctx.Err(); e != nil {
		return e
	}
	if e := writeFull(out, header[:]); e != nil {
		return e
	}
	values := make([]float32, 4096)
	raw := make([]byte, 8192)
	for start := int64(0); start < samples; {
		if e := ctx.Err(); e != nil {
			return e
		}
		count := int(min(int64(len(values)), samples-start))
		n, e := pcm.ReadSamplesAt(ctx, values[:count], start)
		if e != nil {
			return e
		}
		if n != count {
			return io.ErrUnexpectedEOF
		}
		for i, v := range values[:n] {
			binary.LittleEndian.PutUint16(raw[i*2:], uint16(int16(v*32768)))
		}
		if e = writeFull(out, raw[:n*2]); e != nil {
			return e
		}
		start += int64(n)
	}
	return ctx.Err()
}
