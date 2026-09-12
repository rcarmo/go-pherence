package audio

import (
	"math"
	"testing"
)

func directWhisperPower400(input []float64) []float64 {
	out := make([]float64, whisperBins)
	for bin := 0; bin < whisperBins; bin++ {
		var real, imag float64
		for sample, value := range input {
			angle := 2 * math.Pi * float64(bin*sample) / whisperFFTSize
			real += value * math.Cos(angle)
			imag -= value * math.Sin(angle)
		}
		r, i := float64(float32(real)), float64(float32(imag))
		out[bin] = r*r + i*i
	}
	return out
}

func TestWhisperFFT400AgainstDirectDFT(t *testing.T) {
	plan := newWhisperFFT400Plan()
	for _, name := range []string{"impulse", "mixture", "signed-zero"} {
		t.Run(name, func(t *testing.T) {
			input := make([]float64, whisperFFTSize)
			switch name {
			case "impulse":
				input[137] = .75
			case "mixture":
				for i := range input {
					input[i] = .2*math.Sin(2*math.Pi*17*float64(i)/whisperFFTSize) + .03*math.Cos(2*math.Pi*83*float64(i)/whisperFFTSize)
				}
			case "signed-zero":
				for i := range input {
					input[i] = math.Copysign(0, float64(i%2*2-1))
				}
			}
			want := directWhisperPower400(input)
			got := make([]float64, whisperBins)
			if !plan.powerSpectrum400(got, input, make([]complex128, whisperFFTSize)) {
				t.Fatal("fixed shape rejected")
			}
			for i := range got {
				tol := 2e-9 * math.Max(1, want[i])
				if diff := math.Abs(got[i] - want[i]); diff > tol {
					t.Fatalf("bin%d got%.12g want%.12g diff%.3g tol%.3g", i, got[i], want[i], diff, tol)
				}
			}
		})
	}
	if plan.powerSpectrum400(make([]float64, whisperBins-1), make([]float64, whisperFFTSize), make([]complex128, whisperFFTSize)) || plan.powerSpectrum400(make([]float64, whisperBins), make([]float64, whisperFFTSize-1), make([]complex128, whisperFFTSize)) || plan.powerSpectrum400(make([]float64, whisperBins), make([]float64, whisperFFTSize), make([]complex128, whisperFFTSize-1)) {
		t.Fatal("invalid FFT400 shape accepted")
	}
}

func BenchmarkWhisperFFT400(b *testing.B) {
	input := make([]float64, whisperFFTSize)
	for i := range input {
		input[i] = .2*math.Sin(float64(i)*.17) + .03*math.Cos(float64(i)*.83)
	}
	b.Run("direct-dft", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = directWhisperPower400(input)
		}
	})
	plan := newWhisperFFT400Plan()
	power := make([]float64, whisperBins)
	stage := make([]complex128, whisperFFTSize)
	b.Run("mixed-radix", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if !plan.powerSpectrum400(power, input, stage) {
				b.Fatal("rejected")
			}
		}
	})
}
