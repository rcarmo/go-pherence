package media

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

type wavInfo struct {
	fileSize      int64
	sampleRate    int
	channels      int
	bitsPerSample int
	blockAlign    int
	frames        int64
	dataOffset    int64
	sourceTiming  SourceTiming
}

func validateCanonicalWAV(path string, maxBytes int64, maxFrames int64) (wavInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return wavInfo{}, fmt.Errorf("open decoded wav: %w", err)
	}
	defer f.Close()
	return validateCanonicalWAVFile(context.Background(), f, maxBytes, maxFrames)
}

// Scan the already-open file so validation and PCM reads use the same inode.
// Seeking here is confined to construction; PCMReader uses positional reads.
func validateCanonicalWAVFile(ctx context.Context, f *os.File, maxBytes int64, maxFrames int64) (wavInfo, error) {
	if err := ctx.Err(); err != nil {
		return wavInfo{}, err
	}
	fi, err := f.Stat()
	if err != nil {
		return wavInfo{}, fmt.Errorf("stat decoded wav: %w", err)
	}
	if !fi.Mode().IsRegular() {
		return wavInfo{}, fmt.Errorf("%w: decoded WAV must be a regular file", ErrInvalidOutput)
	}
	if fi.Size() <= 0 {
		return wavInfo{}, fmt.Errorf("%w: empty decoded file", ErrInvalidOutput)
	}
	if maxBytes > 0 && fi.Size() > maxBytes {
		return wavInfo{}, fmt.Errorf("%w: %d bytes", ErrDecodeOutputLimit, fi.Size())
	}

	header := make([]byte, 12)
	if _, err := io.ReadFull(f, header); err != nil {
		return wavInfo{}, fmt.Errorf("%w: read riff header: %v", ErrInvalidOutput, err)
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return wavInfo{}, fmt.Errorf("%w: not RIFF/WAVE", ErrInvalidOutput)
	}
	riffSize := int64(binary.LittleEndian.Uint32(header[4:8])) + 8
	if riffSize != fi.Size() || riffSize < 12 {
		return wavInfo{}, fmt.Errorf("%w: inconsistent RIFF size", ErrInvalidOutput)
	}

	var (
		foundFmt  bool
		foundData bool
		format    uint16
		rate      uint32
		channels  uint16
		byteRate  uint32
		block     uint16
		bits      uint16
		dataSize  uint32
		dataStart int64
		timing    SourceTiming
		hasTiming bool
		offset    int64 = 12
	)

	for chunks := 0; offset < riffSize; chunks++ {
		if err := ctx.Err(); err != nil {
			return wavInfo{}, err
		}
		if chunks >= 4096 {
			return wavInfo{}, fmt.Errorf("%w: too many WAV chunks", ErrInvalidOutput)
		}
		if riffSize-offset < 8 {
			return wavInfo{}, fmt.Errorf("%w: partial chunk header", ErrInvalidOutput)
		}
		chunkHeader := make([]byte, 8)
		if _, err := io.ReadFull(f, chunkHeader); err != nil {
			return wavInfo{}, fmt.Errorf("%w: read chunk header: %v", ErrInvalidOutput, err)
		}
		chunkSize := int64(binary.LittleEndian.Uint32(chunkHeader[4:8]))
		if chunkSize < 0 || offset+8+chunkSize > riffSize {
			return wavInfo{}, fmt.Errorf("%w: truncated chunk", ErrInvalidOutput)
		}
		switch string(chunkHeader[0:4]) {
		case "fmt ":
			if foundFmt {
				return wavInfo{}, fmt.Errorf("%w: duplicate fmt chunk", ErrInvalidOutput)
			}
			if chunkSize < 16 {
				return wavInfo{}, fmt.Errorf("%w: short fmt chunk", ErrInvalidOutput)
			}
			fmtData := make([]byte, 16)
			if _, err := io.ReadFull(f, fmtData); err != nil {
				return wavInfo{}, fmt.Errorf("%w: read fmt chunk: %v", ErrInvalidOutput, err)
			}
			if _, err := f.Seek(chunkSize-16, io.SeekCurrent); err != nil {
				return wavInfo{}, fmt.Errorf("%w: skip fmt extension", ErrInvalidOutput)
			}
			format = binary.LittleEndian.Uint16(fmtData[0:2])
			channels = binary.LittleEndian.Uint16(fmtData[2:4])
			rate = binary.LittleEndian.Uint32(fmtData[4:8])
			byteRate = binary.LittleEndian.Uint32(fmtData[8:12])
			block = binary.LittleEndian.Uint16(fmtData[12:14])
			bits = binary.LittleEndian.Uint16(fmtData[14:16])
			foundFmt = true
		case "data":
			if foundData {
				return wavInfo{}, fmt.Errorf("%w: duplicate data chunk", ErrInvalidOutput)
			}
			if _, err := f.Seek(chunkSize, io.SeekCurrent); err != nil {
				return wavInfo{}, fmt.Errorf("%w: seek data chunk: %v", ErrInvalidOutput, err)
			}
			dataSize = uint32(chunkSize)
			dataStart = offset + 8
			foundData = true
		case "gptm":
			if hasTiming || chunkSize != sourceTimingChunkBytes-8 {
				return wavInfo{}, fmt.Errorf("%w: duplicate/invalid source timing", ErrInvalidOutput)
			}
			payload := make([]byte, int(chunkSize))
			if _, err := io.ReadFull(f, payload); err != nil {
				return wavInfo{}, fmt.Errorf("%w: read source timing: %v", ErrInvalidOutput, err)
			}
			timing, err = parseSourceTimingWAVChunk(payload)
			if err != nil {
				return wavInfo{}, err
			}
			hasTiming = true
		default:
			if _, err := f.Seek(chunkSize, io.SeekCurrent); err != nil {
				return wavInfo{}, fmt.Errorf("%w: skip chunk: %v", ErrInvalidOutput, err)
			}
		}
		offset += 8 + chunkSize
		if chunkSize%2 == 1 {
			if offset >= riffSize {
				return wavInfo{}, fmt.Errorf("%w: truncated pad byte", ErrInvalidOutput)
			}
			if _, err := f.Seek(1, io.SeekCurrent); err != nil {
				return wavInfo{}, fmt.Errorf("%w: skip pad byte: %v", ErrInvalidOutput, err)
			}
			offset++
		}
	}
	if !foundFmt || !foundData {
		return wavInfo{}, fmt.Errorf("%w: missing required chunks", ErrInvalidOutput)
	}
	if format != 1 {
		return wavInfo{}, fmt.Errorf("%w: audio format %d", ErrInvalidOutput, format)
	}
	if int(rate) != int(CanonicalSampleRate) || int(channels) != CanonicalChannels || int(bits) != CanonicalBitsPerSample {
		return wavInfo{}, fmt.Errorf("%w: unexpected wav format", ErrInvalidOutput)
	}
	expectedBlock := CanonicalChannels * (CanonicalBitsPerSample / 8)
	if int(block) != expectedBlock {
		return wavInfo{}, fmt.Errorf("%w: block align %d", ErrInvalidOutput, block)
	}
	if int(byteRate) != int(rate)*int(block) {
		return wavInfo{}, fmt.Errorf("%w: byte rate %d", ErrInvalidOutput, byteRate)
	}
	if dataSize == 0 {
		return wavInfo{}, fmt.Errorf("%w: empty audio data", ErrInvalidOutput)
	}
	if int(dataSize)%expectedBlock != 0 {
		return wavInfo{}, fmt.Errorf("%w: unaligned data chunk", ErrInvalidOutput)
	}
	frames := int64(dataSize / uint32(expectedBlock))
	if maxFrames > 0 && frames > maxFrames {
		return wavInfo{}, fmt.Errorf("%w: %d frames", ErrDecodeOutputLimit, frames)
	}
	return wavInfo{
		fileSize:      fi.Size(),
		sampleRate:    int(rate),
		channels:      int(channels),
		bitsPerSample: int(bits),
		blockAlign:    int(block),
		frames:        frames,
		dataOffset:    dataStart,
		sourceTiming:  timing,
	}, nil
}
