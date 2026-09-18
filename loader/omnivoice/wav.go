package omnivoice

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// WriteSyntheticWAV exclusively creates a PCM16 mono WAV, attenuating only if
// needed for .98 peak headroom. Nonfinite/silent outputs are rejected. Returns
// applied gain. The caller must label the recording synthetic in its metadata.
func WriteSyntheticWAV(path string, samples []float32, rate int) (float32, error) {
	if len(samples) == 0 || len(samples) > (math.MaxUint32-36)/2 || rate <= 0 || rate > 192000 {
		return 0, fmt.Errorf("omnivoice: invalid WAV size/rate")
	}
	peak := float32(0)
	for _, v := range samples {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, fmt.Errorf("omnivoice: nonfinite waveform")
		}
		peak = max(peak, float32(math.Abs(float64(v))))
	}
	if peak < 1e-7 {
		return 0, fmt.Errorf("omnivoice: silent waveform")
	}
	gain := float32(1)
	if peak > .98 {
		gain = .98 / peak
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return 0, err
	}
	success := false
	defer func() {
		f.Close()
		if !success {
			os.Remove(path)
		}
	}()
	header := make([]byte, 44)
	copy(header, "RIFF")
	binary.LittleEndian.PutUint32(header[4:], uint32(36+len(samples)*2))
	copy(header[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 1)
	binary.LittleEndian.PutUint16(header[22:], 1)
	binary.LittleEndian.PutUint32(header[24:], uint32(rate))
	binary.LittleEndian.PutUint32(header[28:], uint32(rate*2))
	binary.LittleEndian.PutUint16(header[32:], 2)
	binary.LittleEndian.PutUint16(header[34:], 16)
	copy(header[36:], "data")
	binary.LittleEndian.PutUint32(header[40:], uint32(len(samples)*2))
	if _, err = f.Write(header); err != nil {
		return 0, err
	}
	buf := make([]byte, 8192)
	for start := 0; start < len(samples); start += len(buf) / 2 {
		n := min(len(buf)/2, len(samples)-start)
		for i := 0; i < n; i++ {
			v := int16(math.Round(float64(samples[start+i]*gain) * 32767))
			binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
		}
		if _, err = f.Write(buf[:n*2]); err != nil {
			return 0, err
		}
	}
	if err = f.Close(); err != nil {
		return 0, err
	}
	success = true
	return gain, nil
}
