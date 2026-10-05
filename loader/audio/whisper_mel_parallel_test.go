package audio

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

// denseSerialMel is the previous dense, single-threaded filterbank loop.
func denseSerialMel(samples []float32, numMels int) ([]float32, float32) {
	whisperExactTables.Do(initWhisperExactTables)
	filters := whisperExactTables.filters
	if numMels == 128 {
		filters = whisperExactTables.filters128
	}
	frames := len(samples) / whisperHop
	centered := reflectCenter(samples, whisperFFTSize/2)
	out := make([]float32, numMels*frames)
	power, windowed, scratch := make([]float64, whisperBins), make([]float64, whisperFFTSize), make([]complex128, whisperFFTSize)
	maxLog := float32(-math.MaxFloat32)
	for frame := 0; frame < frames; frame++ {
		for s := range windowed {
			windowed[s] = float64(centered[frame*whisperHop+s]) * whisperExactTables.window[s]
		}
		whisperExactTables.fft400.powerSpectrum400(power, windowed, scratch)
		for mel := 0; mel < numMels; mel++ {
			e := float64(0)
			for bin := 0; bin < whisperBins; bin++ {
				e += filters[bin*numMels+mel] * power[bin]
			}
			if e < 1e-10 {
				e = 1e-10
			}
			v := float32(math.Log10(e))
			out[mel*frames+frame] = v
			maxLog = max(maxLog, v)
		}
	}
	return out, maxLog
}

func TestWhisperMelSparseParallelExact(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	old := whisperMelWorkers
	defer func() { whisperMelWorkers = old }()
	for _, n := range []int{160, 161, 16000, 123457, 480000} {
		s := make([]float32, n)
		for i := range s {
			switch {
			case i%9973 < 50:
				s[i] = 0
			default:
				s[i] = float32(r.NormFloat64() * math.Pow(10, float64(r.Intn(5)-4)))
			}
		}
		for _, mels := range []int{80, 128} {
			want, wantMax := denseSerialMel(s, mels)
			for _, workers := range []int{1, 3, 4} {
				whisperMelWorkers = workers
				got, frames, gotMax, err := whisperLogMelRaw(context.Background(), s, mels)
				if err != nil || frames != n/whisperHop || len(got) != len(want) {
					t.Fatal(n, mels, err, frames)
				}
				if frames > 0 && math.Float32bits(gotMax) != math.Float32bits(wantMax) {
					t.Fatal("max", n, mels, workers)
				}
				for i := range want {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("n=%d mels=%d workers=%d value %d", n, mels, workers, i)
					}
				}
			}
		}
	}
}
