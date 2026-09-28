package pockettts

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

func WritePCM16Mono24k(path string, samples []float32) error {
	if len(samples) == 0 || len(samples) > (math.MaxUint32-36)/2 {
		return fmt.Errorf("invalid Pocket TTS WAV samples=%d", len(samples))
	}
	for i, v := range samples {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < -1 || v > 1 {
			return fmt.Errorf("invalid Pocket TTS WAV sample[%d]=%g", i, v)
		}
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	ok := false
	defer func() {
		_ = f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	h := make([]byte, 44)
	copy(h, "RIFF")
	binary.LittleEndian.PutUint32(h[4:], uint32(36+2*len(samples)))
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1)
	binary.LittleEndian.PutUint16(h[22:], 1)
	binary.LittleEndian.PutUint32(h[24:], SampleRate)
	binary.LittleEndian.PutUint32(h[28:], 2*SampleRate)
	binary.LittleEndian.PutUint16(h[32:], 2)
	binary.LittleEndian.PutUint16(h[34:], 16)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], uint32(2*len(samples)))
	if _, err = f.Write(h); err != nil {
		return err
	}
	buf := make([]byte, 8192)
	for start := 0; start < len(samples); start += len(buf) / 2 {
		n := min(len(buf)/2, len(samples)-start)
		for i := 0; i < n; i++ {
			v := samples[start+i]
			var pcm int16
			if v <= -1 {
				pcm = math.MinInt16
			} else {
				pcm = int16(math.Round(float64(v) * math.MaxInt16))
			}
			binary.LittleEndian.PutUint16(buf[2*i:], uint16(pcm))
		}
		if _, err = f.Write(buf[:2*n]); err != nil {
			return err
		}
	}
	if err = f.Close(); err != nil {
		return err
	}
	ok = true
	return nil
}
