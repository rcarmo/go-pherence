// Package audio provides audio I/O, resampling, and mel spectrogram computation
// for speech models (Whisper).
package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// WAV reads a WAV file and returns mono float32 samples at the file's native sample rate.
func WAV(path string) (samples []float32, sampleRate int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	return ReadWAV(f)
}

// ReadWAV reads a WAV stream and returns mono float32 samples.
func ReadWAV(r io.Reader) (samples []float32, sampleRate int, err error) {
	// Read RIFF header
	var riffID [4]byte
	if err := binary.Read(r, binary.LittleEndian, &riffID); err != nil {
		return nil, 0, fmt.Errorf("read RIFF: %w", err)
	}
	if string(riffID[:]) != "RIFF" {
		return nil, 0, fmt.Errorf("not a RIFF file: %q", riffID)
	}

	var fileSize uint32
	if err := binary.Read(r, binary.LittleEndian, &fileSize); err != nil {
		return nil, 0, err
	}
	if fileSize < 4 {
		return nil, 0, fmt.Errorf("invalid RIFF size")
	}
	var waveID [4]byte
	if err := binary.Read(r, binary.LittleEndian, &waveID); err != nil {
		return nil, 0, err
	}
	if string(waveID[:]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a WAVE file: %q", waveID)
	}

	remaining := int64(fileSize) - 4
	r = &io.LimitedReader{R: r, N: remaining}
	// Parse chunks
	var (
		audioFormat   uint16
		numChannels   uint16
		rate          uint32
		bitsPerSample uint16
		dataSize      uint32
		foundFmt      bool
		foundData     bool
	)

	for chunks := 0; !foundData && remaining > 0; chunks++ {
		if chunks >= 4096 || remaining < 8 {
			return nil, 0, fmt.Errorf("invalid WAV chunk count/header")
		}
		var chunkID [4]byte
		if err := binary.Read(r, binary.LittleEndian, &chunkID); err != nil {
			return nil, 0, fmt.Errorf("read chunk: %w", err)
		}
		var chunkSize uint32
		if err := binary.Read(r, binary.LittleEndian, &chunkSize); err != nil {
			return nil, 0, err
		}
		padded := int64(chunkSize) + int64(chunkSize%2)
		if padded > remaining-8 {
			return nil, 0, fmt.Errorf("WAV chunk exceeds RIFF payload")
		}
		remaining -= 8 + padded

		switch string(chunkID[:]) {
		case "fmt ":
			if foundFmt || chunkSize < 16 {
				return nil, 0, fmt.Errorf("invalid/duplicate fmt chunk")
			}
			var raw [16]byte
			if _, err := io.ReadFull(r, raw[:]); err != nil {
				return nil, 0, err
			}
			audioFormat = binary.LittleEndian.Uint16(raw[0:2])
			numChannels = binary.LittleEndian.Uint16(raw[2:4])
			rate = binary.LittleEndian.Uint32(raw[4:8])
			bitsPerSample = binary.LittleEndian.Uint16(raw[14:16])
			byteRate, align := binary.LittleEndian.Uint32(raw[8:12]), binary.LittleEndian.Uint16(raw[12:14])
			if numChannels == 0 || rate == 0 || (audioFormat != 1 && audioFormat != 3) || (audioFormat == 1 && bitsPerSample != 16 && bitsPerSample != 32) || (audioFormat == 3 && bitsPerSample != 32) {
				return nil, 0, fmt.Errorf("invalid/unsupported WAV format")
			}
			frameBytes := uint64(numChannels) * uint64(bitsPerSample/8)
			if uint64(align) != frameBytes || uint64(byteRate) != uint64(rate)*frameBytes {
				return nil, 0, fmt.Errorf("invalid WAV byte rate/alignment")
			}
			if _, err := io.CopyN(io.Discard, r, int64(chunkSize)-16); err != nil {
				return nil, 0, err
			}
			foundFmt = true

		case "data":
			if !foundFmt {
				return nil, 0, fmt.Errorf("data before fmt")
			}
			dataSize = chunkSize
			foundData = true

		default:
			// Skip unknown chunk
			if _, err := io.CopyN(io.Discard, r, int64(chunkSize)); err != nil {
				return nil, 0, err
			}
		}
		if !foundData && chunkSize%2 != 0 {
			if _, err := io.CopyN(io.Discard, r, 1); err != nil {
				return nil, 0, err
			}
		}
	}

	if !foundFmt {
		return nil, 0, fmt.Errorf("no fmt chunk found")
	}
	if !foundData {
		return nil, 0, fmt.Errorf("no data chunk found")
	}
	if audioFormat != 1 && audioFormat != 3 {
		return nil, 0, fmt.Errorf("unsupported audio format %d (want PCM=1 or float=3)", audioFormat)
	}

	bytesPerSample := int(bitsPerSample) / 8
	if dataSize == 0 || uint64(dataSize) > uint64(int(^uint(0)>>1)) || uint64(dataSize)%uint64(int(numChannels)*bytesPerSample) != 0 {
		return nil, 0, fmt.Errorf("invalid WAV data size/alignment")
	}
	numSamples := int(dataSize) / (int(numChannels) * bytesPerSample)
	ch := int(numChannels)

	// Read the whole data chunk once and decode with direct little-endian byte
	// arithmetic. The previous per-sample binary.Read was reflection-bound and
	// took ~37s on a 26-minute file; this is a single buffered read + tight loop.
	// Grow from actual bytes rather than trusting a potentially forged size.
	raw, err := io.ReadAll(io.LimitReader(r, int64(dataSize)))
	if err != nil {
		return nil, 0, err
	}
	if len(raw) != int(dataSize) {
		return nil, 0, io.ErrUnexpectedEOF
	}
	samples = make([]float32, numSamples)

	switch {
	case audioFormat == 1 && bitsPerSample == 16:
		if ch == 1 {
			for i := 0; i < numSamples; i++ {
				samples[i] = float32(int16(binary.LittleEndian.Uint16(raw[2*i:]))) / 32768.0
			}
		} else {
			for i := 0; i < numSamples; i++ {
				base := i * ch * 2
				var sum float32
				for c := 0; c < ch; c++ {
					sum += float32(int16(binary.LittleEndian.Uint16(raw[base+2*c:]))) / 32768.0
				}
				samples[i] = sum / float32(ch)
			}
		}

	case audioFormat == 1 && bitsPerSample == 32:
		for i := 0; i < numSamples; i++ {
			base := i * ch * 4
			var sum float32
			for c := 0; c < ch; c++ {
				sum += float32(int32(binary.LittleEndian.Uint32(raw[base+4*c:]))) / float32(math.MaxInt32)
			}
			samples[i] = sum / float32(ch)
		}

	case audioFormat == 3 && bitsPerSample == 32:
		for i := 0; i < numSamples; i++ {
			base := i * ch * 4
			var sum float32
			for c := 0; c < ch; c++ {
				sum += math.Float32frombits(binary.LittleEndian.Uint32(raw[base+4*c:]))
			}
			samples[i] = sum / float32(ch)
		}

	default:
		return nil, 0, fmt.Errorf("unsupported format: audioFormat=%d bits=%d", audioFormat, bitsPerSample)
	}

	for _, v := range samples {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, 0, fmt.Errorf("nonfinite WAV sample")
		}
	}
	return samples, int(rate), nil
}
