package qwen3tts

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// WritePCM16Mono24k writes an exclusive little-endian PCM16 mono WAV. Samples
// must be finite and already bounded to [-1,1]; failures remove partial output.
func WritePCM16Mono24k(path string, samples []float32) error {
	if len(samples) == 0 || len(samples) > (math.MaxUint32-36)/2 {
		return fmt.Errorf("invalid Qwen3-TTS WAV samples=%d", len(samples))
	}
	for i, value := range samples {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < -1 || value > 1 {
			return fmt.Errorf("invalid Qwen3-TTS WAV sample[%d]=%g", i, value)
		}
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	success := false
	defer func() {
		_ = file.Close()
		if !success {
			_ = os.Remove(path)
		}
	}()
	header := make([]byte, 44)
	copy(header, "RIFF")
	binary.LittleEndian.PutUint32(header[4:], uint32(36+2*len(samples)))
	copy(header[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 1)
	binary.LittleEndian.PutUint16(header[22:], 1)
	binary.LittleEndian.PutUint32(header[24:], decoder12HzSampleRate)
	binary.LittleEndian.PutUint32(header[28:], 2*decoder12HzSampleRate)
	binary.LittleEndian.PutUint16(header[32:], 2)
	binary.LittleEndian.PutUint16(header[34:], 16)
	copy(header[36:], "data")
	binary.LittleEndian.PutUint32(header[40:], uint32(2*len(samples)))
	if _, err = file.Write(header); err != nil {
		return err
	}
	buf := make([]byte, 8192)
	for start := 0; start < len(samples); start += len(buf) / 2 {
		n := min(len(buf)/2, len(samples)-start)
		for i := 0; i < n; i++ {
			value := max(float32(-1), min(float32(1), samples[start+i]))
			var pcm int16
			if value <= -1 {
				pcm = math.MinInt16
			} else {
				pcm = int16(math.Round(float64(value) * math.MaxInt16))
			}
			binary.LittleEndian.PutUint16(buf[2*i:], uint16(pcm))
		}
		if _, err = file.Write(buf[:2*n]); err != nil {
			return err
		}
	}
	if err = file.Close(); err != nil {
		return err
	}
	success = true
	return nil
}
