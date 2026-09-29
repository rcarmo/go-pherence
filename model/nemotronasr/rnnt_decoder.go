package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const rnntBlank = 13087
const rnntGates = 4 * rnntHidden

// RNNTDecoder owns released two-layer LSTM and decoder-projector weights.
// DecoderState belongs to one stream and is not safe for concurrent updates.
type RNNTDecoder struct {
	embedding                    []float32
	inputWeight, recurrentWeight [2][]float32
	inputBias, recurrentBias     [2][]float32
	projectWeight, projectBias   []float32
}

// DecoderState is the owned output and two-layer hidden/cell state of a
// single-stream, single-token RNN-T predictor.
type DecoderState struct {
	output, hidden, cell []float32
	initialized          bool
}

func (s *DecoderState) Snapshot() (output, hidden, cell []float32, initialized bool) {
	if s == nil {
		return nil, nil, nil, false
	}
	return append([]float32(nil), s.output...), append([]float32(nil), s.hidden...), append([]float32(nil), s.cell...), s.initialized
}

func LoadRNNTDecoder(file *safetensors.File) (*RNNTDecoder, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	load := func(name string, dims ...int) ([]float32, error) {
		values, shape, err := file.GetFloat32("decoder." + name)
		if err != nil {
			return nil, err
		}
		if len(shape) != len(dims) {
			return nil, fmt.Errorf("invalid RNNT decoder %s shape %v", name, shape)
		}
		size := 1
		for i, dim := range dims {
			if shape[i] != dim {
				return nil, fmt.Errorf("invalid RNNT decoder %s shape %v", name, shape)
			}
			size *= dim
		}
		if len(values) != size {
			return nil, fmt.Errorf("invalid RNNT decoder %s length", name)
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite RNNT decoder %s", name)
			}
		}
		return values, nil
	}
	m := &RNNTDecoder{}
	var err error
	if m.embedding, err = load("embedding.weight", rnntVocabulary, rnntHidden); err != nil {
		return nil, err
	}
	for layer := 0; layer < 2; layer++ {
		if m.inputWeight[layer], err = load(fmt.Sprintf("lstm.weight_ih_l%d", layer), rnntGates, rnntHidden); err != nil {
			return nil, err
		}
		if m.recurrentWeight[layer], err = load(fmt.Sprintf("lstm.weight_hh_l%d", layer), rnntGates, rnntHidden); err != nil {
			return nil, err
		}
		if m.inputBias[layer], err = load(fmt.Sprintf("lstm.bias_ih_l%d", layer), rnntGates); err != nil {
			return nil, err
		}
		if m.recurrentBias[layer], err = load(fmt.Sprintf("lstm.bias_hh_l%d", layer), rnntGates); err != nil {
			return nil, err
		}
	}
	if m.projectWeight, err = load("decoder_projector.weight", rnntHidden, rnntHidden); err != nil {
		return nil, err
	}
	if m.projectBias, err = load("decoder_projector.bias", rnntHidden); err != nil {
		return nil, err
	}
	return m, nil
}

// Step predicts one token at a time. An initial blank runs the LSTM; later
// blanks return the owned cached output without changing the state. No token
// selection, joint call, or multi-batch cache update is performed here.
func (m *RNNTDecoder) Step(token int, state *DecoderState) ([]float32, error) {
	output, err := m.stepBorrowed(token, state)
	if err != nil {
		return nil, err
	}
	return append([]float32(nil), output...), nil
}

// stepBorrowed returns the current state's output. Only same-package callers
// that finish using it before the next Step may use this borrowed view.
func (m *RNNTDecoder) stepBorrowed(token int, state *DecoderState) ([]float32, error) {
	if m == nil || state == nil || len(m.embedding) != rnntVocabulary*rnntHidden || len(m.projectWeight) != rnntHidden*rnntHidden || len(m.projectBias) != rnntHidden {
		return nil, fmt.Errorf("invalid RNNT decoder")
	}
	if token < 0 || token >= rnntVocabulary {
		return nil, fmt.Errorf("invalid RNNT token")
	}
	if state.initialized {
		if len(state.output) != rnntHidden || len(state.hidden) != 2*rnntHidden || len(state.cell) != 2*rnntHidden {
			return nil, fmt.Errorf("invalid RNNT decoder cache")
		}
		for _, values := range [][]float32{state.output, state.hidden, state.cell} {
			for _, value := range values {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return nil, fmt.Errorf("non-finite RNNT decoder cache")
				}
			}
		}
		if token == rnntBlank {
			return state.output, nil
		}
	} else if len(state.output) != 0 || len(state.hidden) != 0 || len(state.cell) != 0 {
		return nil, fmt.Errorf("invalid uninitialized RNNT decoder cache")
	}
	hidden := make([]float32, 2*rnntHidden)
	cell := make([]float32, 2*rnntHidden)
	input := m.embedding[token*rnntHidden : (token+1)*rnntHidden]
	for layer := 0; layer < 2; layer++ {
		if len(m.inputWeight[layer]) != rnntGates*rnntHidden || len(m.recurrentWeight[layer]) != rnntGates*rnntHidden || len(m.inputBias[layer]) != rnntGates || len(m.recurrentBias[layer]) != rnntGates {
			return nil, fmt.Errorf("invalid RNNT LSTM weights")
		}
		gates := make([]float32, rnntGates)
		if !simd.SgemmNTTo(gates, input, m.inputWeight[layer], 1, rnntGates, rnntHidden, 1, rnntHidden, rnntHidden, rnntGates) {
			return nil, fmt.Errorf("RNNT input gates rejected")
		}
		prevH, prevC := make([]float32, rnntHidden), make([]float32, rnntHidden)
		if state.initialized {
			copy(prevH, state.hidden[layer*rnntHidden:(layer+1)*rnntHidden])
			copy(prevC, state.cell[layer*rnntHidden:(layer+1)*rnntHidden])
		}
		if !simd.SgemmNTTo(gates, prevH, m.recurrentWeight[layer], 1, rnntGates, rnntHidden, 1, rnntHidden, rnntHidden, rnntGates) {
			return nil, fmt.Errorf("RNNT recurrent gates rejected")
		}
		for dim := 0; dim < rnntHidden; dim++ {
			activate := func(offset int) float64 {
				return float64(gates[offset+dim] + m.inputBias[layer][offset+dim] + m.recurrentBias[layer][offset+dim])
			}
			sigmoid := func(x float64) float32 { return float32(1 / (1 + math.Exp(-x))) }
			i, f, g, o := sigmoid(activate(0)), sigmoid(activate(rnntHidden)), float32(math.Tanh(activate(2*rnntHidden))), sigmoid(activate(3*rnntHidden))
			c := f*prevC[dim] + i*g
			cell[layer*rnntHidden+dim] = c
			hidden[layer*rnntHidden+dim] = o * float32(math.Tanh(float64(c)))
			if math.IsNaN(float64(c)) || math.IsInf(float64(c), 0) || math.IsNaN(float64(hidden[layer*rnntHidden+dim])) || math.IsInf(float64(hidden[layer*rnntHidden+dim]), 0) {
				return nil, fmt.Errorf("non-finite RNNT decoder state")
			}
		}
		input = hidden[layer*rnntHidden : (layer+1)*rnntHidden]
	}
	output := make([]float32, rnntHidden)
	if !simd.SgemmNTTo(output, input, m.projectWeight, 1, rnntHidden, rnntHidden, 1, rnntHidden, rnntHidden, rnntHidden) {
		return nil, fmt.Errorf("RNNT decoder projector rejected")
	}
	for i := range output {
		output[i] += m.projectBias[i]
		if math.IsNaN(float64(output[i])) || math.IsInf(float64(output[i]), 0) {
			return nil, fmt.Errorf("non-finite RNNT decoder output")
		}
	}
	state.output, state.hidden, state.cell, state.initialized = output, hidden, cell, true
	return output, nil
}
