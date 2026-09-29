package audio

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/fft"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// NemotronMelStream owns the 400-sample centered STFT history for one mono
// 16-kHz stream. AppendPCM returns only finalized valid feature rows; Finish
// emits the remaining zero-padded rows and the processor's masked final row.
// State memory is independent of the duration of the recording. Returned
// chunks are owned and remain unchanged by later calls.
type NemotronMelStream struct {
	history            [512]float32 // pre-emphasized samples indexed modulo 512
	previous           float32
	samples, nextFrame int64
	closed             bool
	windowReady        bool
	window             [400]float32
	frame              [512]float32
	spectrum           [514]float32
	real, imag         [512]float64
	power              [257]float32
}

const nemotronStreamMaxChunk = 16000 * 5

// AppendPCM accepts at most five seconds per call to bound returned storage.
// Rejected input leaves the stream unchanged; neither input nor previous
// returned feature chunks are mutated by later calls.
func (s *NemotronMelStream) AppendPCM(samples []float32) ([]float32, error) {
	if s == nil || s.closed || len(samples) == 0 || len(samples) > nemotronStreamMaxChunk || s.samples > math.MaxInt64-200-int64(len(samples)) {
		return nil, fmt.Errorf("invalid Nemotron streaming PCM chunk")
	}
	for _, value := range samples {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron streaming PCM chunk")
		}
	}
	s.initWindow()
	out := make([]float32, 0, (len(samples)/160+2)*128)
	for _, value := range samples {
		pre := value - .97*s.previous
		s.previous = value
		s.history[s.samples%512] = pre
		s.samples++
		for s.nextFrame*160+200 <= s.samples {
			out = append(out, make([]float32, 128)...)
			if err := s.featureRow(out[len(out)-128:]); err != nil {
				return nil, err
			}
			s.nextFrame++
		}
	}
	return out, nil
}

// Finish may be called once after at least one input sample. It returns
// owned features, including one masked all-zero frame at the end.
func (s *NemotronMelStream) Finish() ([]float32, error) {
	if s == nil || s.closed || s.samples == 0 {
		return nil, fmt.Errorf("invalid Nemotron streaming finish")
	}
	if s.nextFrame > s.samples/160 {
		return nil, fmt.Errorf("invalid Nemotron streaming frame state")
	}
	s.initWindow()
	out := make([]float32, 0, int(s.samples/160-s.nextFrame+1)*128)
	for s.nextFrame < s.samples/160 {
		out = append(out, make([]float32, 128)...)
		if err := s.featureRow(out[len(out)-128:]); err != nil {
			return nil, err
		}
		s.nextFrame++
	}
	out = append(out, make([]float32, 128)...)
	s.closed = true
	return out, nil
}

func (s *NemotronMelStream) initWindow() {
	if s.windowReady {
		return
	}
	for i := range s.window {
		s.window[i] = float32(.5 * (1 - math.Cos(2*math.Pi*float64(i)/399)))
	}
	s.windowReady = true
}

func (s *NemotronMelStream) featureRow(out []float32) error {
	clear(s.frame[:])
	start := s.nextFrame*160 - 200
	for i := 0; i < 400; i++ {
		index := start + int64(i)
		if index >= 0 && index < s.samples {
			s.frame[56+i] = s.history[index%512] * s.window[i]
		}
	}
	if !fft.ForwardRealInto(s.spectrum[:], s.frame[:], s.real[:], s.imag[:]) {
		return fmt.Errorf("Nemotron streaming FFT rejected frame")
	}
	for i := range s.power {
		re, im := s.spectrum[2*i], s.spectrum[2*i+1]
		magnitude := float32(math.Sqrt(float64(re*re + im*im)))
		s.power[i] = magnitude * magnitude
	}
	filters := nemotronFilters()
	if simd.HasSgemmAsm {
		if !simd.DenseNNTo(out[:], s.power[:], filters, 1, 128, 257, 1, 257, 128, 128) {
			return fmt.Errorf("Nemotron streaming mel projection rejected frame")
		}
	} else {
		for mel := range out {
			for bin := range s.power {
				out[mel] += s.power[bin] * filters[bin*128+mel]
			}
		}
	}
	for i := range out {
		out[i] = float32(math.Log(float64(out[i] + float32(1.0/(1<<24)))))
	}
	return nil
}
