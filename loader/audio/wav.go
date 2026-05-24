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
	binary.Read(r, binary.LittleEndian, &fileSize)

	var waveID [4]byte
	binary.Read(r, binary.LittleEndian, &waveID)
	if string(waveID[:]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a WAVE file: %q", waveID)
	}

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

	for !foundData {
		var chunkID [4]byte
		if err := binary.Read(r, binary.LittleEndian, &chunkID); err != nil {
			if err == io.EOF {
				break
			}
			return nil, 0, fmt.Errorf("read chunk: %w", err)
		}
		var chunkSize uint32
		binary.Read(r, binary.LittleEndian, &chunkSize)

		switch string(chunkID[:]) {
		case "fmt ":
			binary.Read(r, binary.LittleEndian, &audioFormat)
			binary.Read(r, binary.LittleEndian, &numChannels)
			binary.Read(r, binary.LittleEndian, &rate)
			var byteRate uint32
			var blockAlign uint16
			binary.Read(r, binary.LittleEndian, &byteRate)
			binary.Read(r, binary.LittleEndian, &blockAlign)
			binary.Read(r, binary.LittleEndian, &bitsPerSample)
			// Skip any extra fmt bytes
			if chunkSize > 16 {
				io.CopyN(io.Discard, r, int64(chunkSize-16))
			}
			foundFmt = true

		case "data":
			dataSize = chunkSize
			foundData = true

		default:
			// Skip unknown chunk
			io.CopyN(io.Discard, r, int64(chunkSize))
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

	numSamples := int(dataSize) / (int(numChannels) * int(bitsPerSample) / 8)
	samples = make([]float32, numSamples)

	switch {
	case audioFormat == 1 && bitsPerSample == 16:
		buf := make([]int16, int(numChannels))
		for i := 0; i < numSamples; i++ {
			if err := binary.Read(r, binary.LittleEndian, buf); err != nil {
				samples = samples[:i]
				break
			}
			// Mix to mono
			var sum float32
			for _, s := range buf {
				sum += float32(s) / 32768.0
			}
			samples[i] = sum / float32(numChannels)
		}

	case audioFormat == 1 && bitsPerSample == 32:
		buf := make([]int32, int(numChannels))
		for i := 0; i < numSamples; i++ {
			if err := binary.Read(r, binary.LittleEndian, buf); err != nil {
				samples = samples[:i]
				break
			}
			var sum float32
			for _, s := range buf {
				sum += float32(s) / float32(math.MaxInt32)
			}
			samples[i] = sum / float32(numChannels)
		}

	case audioFormat == 3 && bitsPerSample == 32:
		buf := make([]float32, int(numChannels))
		for i := 0; i < numSamples; i++ {
			if err := binary.Read(r, binary.LittleEndian, buf); err != nil {
				samples = samples[:i]
				break
			}
			var sum float32
			for _, s := range buf {
				sum += s
			}
			samples[i] = sum / float32(numChannels)
		}

	default:
		return nil, 0, fmt.Errorf("unsupported format: audioFormat=%d bits=%d", audioFormat, bitsPerSample)
	}

	return samples, int(rate), nil
}
