package media

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// PCMReader reads the canonical mono 16 kHz s16 WAV produced by Adapter without
// loading a whole recording. It owns one file descriptor and a 16 KiB conversion
// buffer. Reads use frame offsets and caller-owned float32 output (-1 <= x < 1).
// No resampling, gapless trimming, downmixing or feature extraction happens here.
//
// The file must be immutable in a caller-owned private directory until Close.
// Validation and reads use the same open inode. Calls are serialised, including
// Close; a caller waiting for another read can cancel. A regular-file ReadAt
// already in progress cannot be interrupted by context until that read returns.
// This reader has no video, model, GPU or external-process dependency.
// A PCMReader must not be copied after construction.
type PCMReader struct {
	file  *os.File
	info  wavInfo
	gate  chan struct{}
	bytes []byte
}

var ErrPCMReaderClosed = errors.New("canonical PCM reader is closed")

// OpenCanonicalPCM validates canonical WAV metadata using fixed speech limits
// (four hours and the corresponding byte bound, including one optional source-
// timing chunk). It does not read audio payloads or invoke FFmpeg. Applications
// must finish publishing the WAV before opening.
func OpenCanonicalPCM(ctx context.Context, path string) (*PCMReader, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Avoid opening an obvious FIFO/device. This is not a hostile-directory
	// sandbox: the caller still owns the path throughout open and validation.
	stat, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat canonical PCM: %w", err)
	}
	if !stat.Mode().IsRegular() {
		return nil, fmt.Errorf("%w: canonical PCM must be a regular file", ErrInvalidOutput)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open canonical PCM: %w", err)
	}
	info, err := validateCanonicalWAVFile(ctx, file, canonicalMaxWAVBytes(DefaultMaxDuration)+sourceTimingChunkBytes, maxFramesForDuration(DefaultMaxDuration))
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	return &PCMReader{file: file, info: info, gate: make(chan struct{}, 1), bytes: make([]byte, 16*1024)}, nil
}

// Timeline reports the actual frame count of the validated canonical WAV.
// It is immutable and remains available after Close. Zero/nil readers return an
// invalid zero timeline. Frame offsets are relative to decoded PCM, not original
// container PTS; the media/job layer must preserve the source timeline mapping.
func (r *PCMReader) Timeline() Timeline {
	if r == nil || r.gate == nil {
		return Timeline{}
	}
	return Timeline{SampleRate: CanonicalSampleRate, Samples: SampleCount(r.info.frames)}
}

// SourceTiming returns the optional durable source mapping embedded by the
// speech-job decode stage. Legacy/plain canonical WAVs return the zero value.
func (r *PCMReader) SourceTiming() SourceTiming {
	if r == nil || r.gate == nil {
		return SourceTiming{}
	}
	return r.info.sourceTiming
}

// ReadSamplesAt reads from an absolute canonical-PCM frame offset. It returns
// io.EOF only when dst extends past the validated data extent or starts at its
// end. A short underlying read *within* that extent is an error (the file changed
// or is truncated), never successful end-of-audio. dst[n:] is left untouched;
// callers must discard partial output on any non-EOF error.
func (r *PCMReader) ReadSamplesAt(ctx context.Context, dst []float32, startSample int64) (int, error) {
	if r == nil || r.gate == nil {
		return 0, ErrPCMReaderClosed
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	select {
	case r.gate <- struct{}{}:
		defer func() { <-r.gate }()
	case <-ctx.Done():
		return 0, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if r.file == nil {
		return 0, ErrPCMReaderClosed
	}
	if startSample < 0 || startSample > r.info.frames {
		return 0, fmt.Errorf("PCM frame offset out of range")
	}
	if len(dst) == 0 {
		return 0, nil
	}
	remaining := r.info.frames - startSample
	requested := int64(len(dst))
	if requested > remaining {
		requested = remaining
	}
	read := 0
	for int64(read) < requested {
		if err := ctx.Err(); err != nil {
			return read, err
		}
		n := len(r.bytes) / 2
		if int64(n) > requested-int64(read) {
			n = int(requested - int64(read))
		}
		// Validation bounds dataOffset and frames to a RIFF file <= four hours.
		offset := r.info.dataOffset + 2*(startSample+int64(read))
		bytesRead, err := r.file.ReadAt(r.bytes[:2*n], offset)
		if bytesRead != 2*n || err != nil {
			return read, fmt.Errorf("%w: canonical PCM changed or was truncated", ErrInvalidOutput)
		}
		if err := ctx.Err(); err != nil {
			return read, err
		}
		for i := 0; i < n; i++ {
			dst[read+i] = float32(int16(binary.LittleEndian.Uint16(r.bytes[2*i:2*i+2]))) / 32768
		}
		read += n
	}
	if int64(len(dst)) > requested {
		return read, io.EOF
	}
	return read, nil
}

// Close waits for an owned read to finish and releases the descriptor. It is
// idempotent and never removes the underlying WAV. No background workers exist.
func (r *PCMReader) Close() error {
	if r == nil || r.gate == nil {
		return nil
	}
	r.gate <- struct{}{}
	defer func() { <-r.gate }()
	if r.file == nil {
		return nil
	}
	err := r.file.Close()
	r.file = nil
	r.bytes = nil
	return err
}
