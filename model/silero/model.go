// Package silero implements native CPU inference for the pinned 16-kHz
// Silero 6.2.0 VAD graph. It does not load files or invoke external engines.
package silero

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
	assets "github.com/rcarmo/go-pherence/loader/silero"
)

const WindowSamples = 512
const HiddenSize = 128

type convLayer struct {
	input, output, stride int
	weight, bias          []float32
}

// Model owns immutable F32 decoded weights, including the source's rounded
// F16 convolution/STFT values. New copies them; the loader can be discarded.
// Fresh streams share these weights, never recurrent or scratch state.
type Model struct {
	basis                                                               []float32
	layers                                                              [4]convLayer
	inputWeight, recurrentWeight, inputBias, recurrentBias, finalWeight []float32
	finalBias                                                           float32
}

func New(file *assets.File) (*Model, error) {
	if file == nil || file.Version != [3]int{6, 2, 0} || file.Window != 512 || file.Context != 64 || len(file.Tensors) != 15 {
		return nil, fmt.Errorf("Silero: unsupported model")
	}
	get := func(name string, shape ...int) ([]float32, error) {
		tensor, ok := file.Tensors[name]
		n := 1
		if !ok || len(tensor.Shape) != len(shape) {
			return nil, fmt.Errorf("Silero: missing/invalid tensor %q", name)
		}
		for i, d := range shape {
			if tensor.Shape[i] != d {
				return nil, fmt.Errorf("Silero: tensor %q shape", name)
			}
			n *= d
		}
		if len(tensor.Values) != n {
			return nil, fmt.Errorf("Silero: tensor %q length", name)
		}
		for _, v := range tensor.Values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("Silero: nonfinite tensor %q", name)
			}
		}
		return append([]float32(nil), tensor.Values...), nil
	}
	m := &Model{}
	var err error
	if m.basis, err = get("_model.stft.forward_basis_buffer", 256, 1, 258); err != nil {
		return nil, err
	}
	channels := [5]int{129, 128, 64, 64, 128}
	strides := [4]int{1, 2, 2, 1}
	for i := range m.layers {
		prefix := fmt.Sprintf("_model.encoder.%d.reparam_conv", i)
		layer := convLayer{input: channels[i], output: channels[i+1], stride: strides[i]}
		if layer.weight, err = get(prefix+".weight", 3, layer.input, layer.output); err != nil {
			return nil, err
		}
		if layer.bias, err = get(prefix+".bias", layer.output); err != nil {
			return nil, err
		}
		m.layers[i] = layer
	}
	for _, item := range []struct {
		name  string
		shape []int
		dst   *[]float32
	}{
		{"_model.decoder.rnn.weight_ih", []int{128, 512}, &m.inputWeight},
		{"_model.decoder.rnn.weight_hh", []int{128, 512}, &m.recurrentWeight},
		{"_model.decoder.rnn.bias_ih", []int{512}, &m.inputBias},
		{"_model.decoder.rnn.bias_hh", []int{512}, &m.recurrentBias},
		{"_model.decoder.decoder.2.weight", []int{128}, &m.finalWeight},
	} {
		if *item.dst, err = get(item.name, item.shape...); err != nil {
			return nil, err
		}
	}
	bias, err := get("_model.decoder.decoder.2.bias")
	if err != nil {
		return nil, err
	}
	m.finalBias = bias[0]
	return m, nil
}

// Stream matches whisper.cpp c44b60b's graph scheduling: one zero-padded
// 512-sample frame with reflect-64 STFT padding and one recurrent update.
// That reference reads Context=64 but does not prepend context in this graph.
// This is not a claim of parity with the different PyTorch 576-sample schedule.
// A stream is serial and must not be used concurrently. Invalid input/errors
// leave recurrent state unchanged. Use a new stream or Reset per recording.
type Stream struct {
	model                *Model
	hidden, cell         [128]float32
	padded               [640]float32
	magnitude            [129 * 4]float32
	convA, convB         [128 * 4]float32
	patch                [129 * 3]float32
	gates                [512]float32
	nextHidden, nextCell [128]float32
	finalInput           [128]float32
}

func (m *Model) NewStream() (*Stream, error) {
	if m == nil || len(m.basis) != 258*256 {
		return nil, fmt.Errorf("Silero: invalid model")
	}
	return &Stream{model: m}, nil
}
func (s *Stream) Reset() {
	if s != nil {
		clear(s.hidden[:])
		clear(s.cell[:])
	}
}

// Probability accepts exactly 512 finite mono16-kHz samples. Caller explicitly
// zero-pads a final tail. Steady-state scratch is stream-owned; returned scalar
// probability is owned. Cancellation is checked between graph operators.
func (s *Stream) Probability(ctx context.Context, pcm []float32) (float32, error) {
	if ctx == nil || s == nil || s.model == nil || len(pcm) != 512 {
		return 0, fmt.Errorf("Silero: invalid frame/context")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	for _, v := range pcm {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, fmt.Errorf("Silero: nonfinite PCM")
		}
	}
	m := s.model
	copy(s.padded[64:576], pcm)
	for i := 0; i < 64; i++ {
		s.padded[i] = pcm[64-i]
		s.padded[576+i] = pcm[510-i]
	}
	// ggml_conv_1d with F16 weights rounds im2col input to binary16 first.
	// Keep that source numerical boundary; do not silently use unrounded PCM.
	for i, v := range s.padded {
		s.padded[i] = half.F16ToF32(half.F32ToF16Even(v))
	}
	for frequency := 0; frequency < 129; frequency++ {
		real := m.basis[frequency*256 : (frequency+1)*256]
		imag := m.basis[(frequency+129)*256 : (frequency+130)*256]
		for frame := 0; frame < 4; frame++ {
			input := s.padded[frame*128 : frame*128+256]
			r := simd.Sdot32(input, real)
			im := simd.Sdot32(input, imag)
			square := r*r + im*im
			if !finite(square) {
				return 0, fmt.Errorf("Silero: nonfinite STFT magnitude")
			}
			s.magnitude[frequency*4+frame] = float32(math.Sqrt(float64(square)))
		}
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	input := s.magnitude[:]
	frames := 4
	for i, layer := range m.layers {
		out := s.convA[:]
		if i%2 == 1 {
			out = s.convB[:]
		}
		nextFrames := (frames+2-3)/layer.stride + 1
		if err := s.convolution(ctx, out, input, frames, nextFrames, layer); err != nil {
			return 0, err
		}
		input = out[:layer.output*nextFrames]
		frames = nextFrames
	}
	if frames != 1 || len(input) != 128 {
		return 0, fmt.Errorf("Silero: invalid encoder output")
	}
	// LSTM gate order is input,forget,cell,output. Preserve the source's
	// separate (Wx+b_ih) and (Wh+b_hh) grouping before adding them together.
	for gate := 0; gate < 512; gate++ {
		a := simd.Sdot32(input, m.inputWeight[gate*128:(gate+1)*128]) + m.inputBias[gate]
		b := simd.Sdot32(s.hidden[:], m.recurrentWeight[gate*128:(gate+1)*128]) + m.recurrentBias[gate]
		s.gates[gate] = a + b
		if !finite(s.gates[gate]) {
			return 0, fmt.Errorf("Silero: nonfinite LSTM gate")
		}
	}
	for i := 0; i < 128; i++ {
		in := sigmoid(s.gates[i])
		forget := sigmoid(s.gates[128+i])
		candidate := float32(math.Tanh(float64(s.gates[256+i])))
		out := sigmoid(s.gates[384+i])
		s.nextCell[i] = forget*s.cell[i] + in*candidate
		s.nextHidden[i] = out * float32(math.Tanh(float64(s.nextCell[i])))
		v := s.nextHidden[i]
		if v < 0 {
			v = 0
		}
		s.finalInput[i] = half.F16ToF32(half.F32ToF16Even(v))
	}
	probability := sigmoid(simd.Sdot32(s.finalInput[:], m.finalWeight) + m.finalBias)
	for i := 0; i < 128; i++ {
		if !finite(s.nextHidden[i]) || !finite(s.nextCell[i]) {
			return 0, fmt.Errorf("Silero: nonfinite recurrent state")
		}
	}
	if !finite(probability) {
		return 0, fmt.Errorf("Silero: nonfinite probability")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	s.hidden = s.nextHidden
	s.cell = s.nextCell
	return probability, nil
}
func (s *Stream) convolution(ctx context.Context, out, input []float32, frames, nextFrames int, layer convLayer) error {
	width := layer.input * 3
	for frame := 0; frame < nextFrames; frame++ {
		for channel := 0; channel < layer.input; channel++ {
			for k := 0; k < 3; k++ {
				pos := frame*layer.stride + k - 1
				v := float32(0)
				if pos >= 0 && pos < frames {
					v = input[channel*frames+pos]
				}
				s.patch[channel*3+k] = half.F16ToF32(half.F32ToF16Even(v))
			}
		}
		for channel := 0; channel < layer.output; channel++ {
			v := simd.Sdot32(s.patch[:width], layer.weight[channel*width:(channel+1)*width]) + layer.bias[channel]
			if !finite(v) {
				return fmt.Errorf("Silero: nonfinite convolution")
			}
			if v < 0 {
				v = 0
			}
			out[channel*nextFrames+frame] = v
		}
	}
	return ctx.Err()
}
func sigmoid(x float32) float32 {
	if x >= 0 {
		return 1 / (1 + float32(math.Exp(float64(-x))))
	}
	e := float32(math.Exp(float64(x)))
	return e / (1 + e)
}
func finite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }
