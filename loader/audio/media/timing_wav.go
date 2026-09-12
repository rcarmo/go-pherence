package media

import (
	"encoding/binary"
	"fmt"
	"time"
)

const sourceTimingChunkBytes = 64

// MarshalSourceTimingWAVChunk encodes SourceTiming as a private fixed-size RIFF
// chunk. A zero value emits no chunk for compatibility with pre-mapping WAVs.
// All sample counts use SourceRate frames; durations use signed nanoseconds.
func MarshalSourceTimingWAVChunk(t SourceTiming) ([]byte, error) {
	if t == (SourceTiming{}) {
		return nil, nil
	}
	if err := validateSourceTiming(t); err != nil {
		return nil, err
	}
	b := make([]byte, sourceTimingChunkBytes)
	copy(b[:4], "gptm")
	binary.LittleEndian.PutUint32(b[4:8], sourceTimingChunkBytes-8)
	binary.LittleEndian.PutUint32(b[8:12], 1)
	var flags uint32
	if t.Exact {
		flags |= 1
	}
	if t.HasEdits {
		flags |= 2
	}
	binary.LittleEndian.PutUint32(b[12:16], flags)
	binary.LittleEndian.PutUint32(b[16:20], uint32(t.SourceRate))
	binary.LittleEndian.PutUint64(b[24:32], uint64(t.Start))
	binary.LittleEndian.PutUint64(b[32:40], uint64(t.Duration))
	binary.LittleEndian.PutUint64(b[40:48], uint64(t.Priming))
	binary.LittleEndian.PutUint64(b[48:56], uint64(t.Padding))
	binary.LittleEndian.PutUint64(b[56:64], uint64(t.LeadingSilence))
	return b, nil
}

func parseSourceTimingWAVChunk(b []byte) (SourceTiming, error) {
	var t SourceTiming
	if len(b) != sourceTimingChunkBytes-8 || binary.LittleEndian.Uint32(b[:4]) != 1 || binary.LittleEndian.Uint32(b[12:16]) != 0 {
		return t, fmt.Errorf("%w: invalid source timing chunk", ErrInvalidOutput)
	}
	flags := binary.LittleEndian.Uint32(b[4:8])
	if flags&^uint32(3) != 0 {
		return t, fmt.Errorf("%w: invalid source timing flags", ErrInvalidOutput)
	}
	t = SourceTiming{
		Start:          time.Duration(int64(binary.LittleEndian.Uint64(b[16:24]))),
		Duration:       time.Duration(int64(binary.LittleEndian.Uint64(b[24:32]))),
		Exact:          flags&1 != 0,
		HasEdits:       flags&2 != 0,
		SourceRate:     SampleRate(binary.LittleEndian.Uint32(b[8:12])),
		Priming:        SampleCount(int64(binary.LittleEndian.Uint64(b[32:40]))),
		Padding:        SampleCount(int64(binary.LittleEndian.Uint64(b[40:48]))),
		LeadingSilence: SampleCount(int64(binary.LittleEndian.Uint64(b[48:56]))),
	}
	if err := validateSourceTiming(t); err != nil {
		return SourceTiming{}, err
	}
	return t, nil
}

func validateSourceTiming(t SourceTiming) error {
	if t.Start < 0 || t.Duration <= 0 || t.SourceRate <= 0 || t.SourceRate > 768000 || t.Priming < 0 || t.Padding < 0 || t.LeadingSilence < 0 {
		return fmt.Errorf("%w: invalid source timing", ErrInvalidOutput)
	}
	// Reject duration overflow and sample-count combinations that cannot be
	// represented in source time. The fields remain descriptive mapping data;
	// no equality with canonical PCM duration is inferred here.
	const maxDuration = time.Duration(1<<63 - 1)
	if t.Start > maxDuration-t.Duration {
		return fmt.Errorf("%w: overflowing source timing", ErrInvalidOutput)
	}
	maxFrames := SampleCount((uint64(maxDuration) / uint64(time.Second)) * uint64(t.SourceRate))
	if t.Priming > maxFrames || t.Padding > maxFrames || t.LeadingSilence > maxFrames || t.Priming > maxFrames-t.Padding || t.Priming+t.Padding > maxFrames-t.LeadingSilence {
		return fmt.Errorf("%w: overflowing source sample timing", ErrInvalidOutput)
	}
	return nil
}
