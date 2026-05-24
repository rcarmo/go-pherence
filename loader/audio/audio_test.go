package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/cmplx"
	"testing"
)

func TestReadWAV16bitMono(t *testing.T) {
	// Generate a minimal 16-bit mono WAV with a 440Hz sine wave
	const sampleRate = 16000
	const duration = 0.1 // seconds
	const numSamples = int(sampleRate * duration)

	var buf bytes.Buffer
	// RIFF header
	buf.WriteString("RIFF")
	dataSize := numSamples * 2
	binary.Write(&buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")
	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))
	binary.Write(&buf, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(&buf, binary.LittleEndian, uint16(1)) // mono
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate*2)) // byte rate
	binary.Write(&buf, binary.LittleEndian, uint16(2))            // block align
	binary.Write(&buf, binary.LittleEndian, uint16(16))           // bits
	// data chunk
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(dataSize))
	for i := 0; i < numSamples; i++ {
		sample := int16(32767 * math.Sin(2*math.Pi*440*float64(i)/sampleRate))
		binary.Write(&buf, binary.LittleEndian, sample)
	}

	samples, rate, err := ReadWAV(&buf)
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if rate != sampleRate {
		t.Fatalf("rate=%d want %d", rate, sampleRate)
	}
	if len(samples) != numSamples {
		t.Fatalf("samples=%d want %d", len(samples), numSamples)
	}
	// Check first non-zero sample is positive (440Hz sine starts at 0 then goes up)
	if samples[1] <= 0 {
		t.Fatalf("expected positive sample[1], got %f", samples[1])
	}
}

func TestResample(t *testing.T) {
	// 1 second of 440Hz at 44100Hz
	const srcRate = 44100
	const dstRate = 16000
	src := make([]float32, srcRate)
	for i := range src {
		src[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / srcRate))
	}

	out := Resample(src, srcRate, dstRate)
	if len(out) != dstRate {
		t.Fatalf("resampled length=%d want %d", len(out), dstRate)
	}
	// Verify the resampled signal still oscillates
	crossings := 0
	for i := 1; i < len(out); i++ {
		if (out[i-1] < 0) != (out[i] < 0) {
			crossings++
		}
	}
	// 440Hz → ~880 zero crossings per second
	if crossings < 800 || crossings > 960 {
		t.Fatalf("unexpected zero crossings=%d (want ~880 for 440Hz)", crossings)
	}
}

func TestResampleSameRate(t *testing.T) {
	src := []float32{1, 2, 3}
	out := Resample(src, 16000, 16000)
	if len(out) != 3 {
		t.Fatalf("same rate resample changed length")
	}
}

func TestMelSpectrogram(t *testing.T) {
	cfg := DefaultMelConfig()

	// Generate 1 second of 1kHz tone at 16kHz
	const dur = 1.0
	samples := make([]float32, int(dur*float64(cfg.SampleRate)))
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / float64(cfg.SampleRate)))
	}

	mel := MelSpectrogram(samples, cfg)
	if mel == nil {
		t.Fatal("MelSpectrogram returned nil")
	}
	if len(mel) != cfg.NumMels {
		t.Fatalf("mel bins=%d want %d", len(mel), cfg.NumMels)
	}

	expectedFrames := (len(samples) - cfg.FFTSize) / cfg.HopLength
	if len(mel[0]) != expectedFrames {
		t.Fatalf("mel frames=%d want %d", len(mel[0]), expectedFrames)
	}

	// The 1kHz tone should produce energy in a specific mel band
	// Find the mel band with maximum energy in the first frame
	maxBin := 0
	maxVal := mel[0][0]
	for m := 1; m < cfg.NumMels; m++ {
		if mel[m][0] > maxVal {
			maxVal = mel[m][0]
			maxBin = m
		}
	}
	// 1kHz should fall roughly in mel bin 15-25 for 80-bin filterbank
	if maxBin < 10 || maxBin > 35 {
		t.Fatalf("1kHz tone peak at mel bin %d (expected ~15-25)", maxBin)
	}
	t.Logf("1kHz tone peak at mel bin %d, value %.2f", maxBin, maxVal)
}

func TestFFT(t *testing.T) {
	// FFT of a simple signal: DC + single frequency
	n := 8
	x := make([]complex128, n)
	// Pure cosine at bin 1
	for i := range x {
		x[i] = complex(math.Cos(2*math.Pi*float64(i)/float64(n)), 0)
	}
	fft(x)

	// Bin 1 should have magnitude n/2 = 4, others ~0 except bin n-1 (conjugate)
	mag1 := cmplx.Abs(x[1])
	if mag1 < 3.5 || mag1 > 4.5 {
		t.Fatalf("FFT bin 1 magnitude=%.2f want ~4", mag1)
	}
	// DC should be ~0
	magDC := cmplx.Abs(x[0])
	if magDC > 0.1 {
		t.Fatalf("FFT DC magnitude=%.2f want ~0", magDC)
	}
}
