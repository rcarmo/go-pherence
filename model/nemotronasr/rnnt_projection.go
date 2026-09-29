package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const rnntPrompts = 128
const rnntHidden = 640
const rnntIntermediate = 2048
const rnntVocabulary = 13088

// RNNTProjection owns the released 3.5 prompt fusion, encoder projector and
// joint head. Input must come from the full encoder in a real transcription;
// no LSTM decoder, token loop or encoder integration is provided here.
type RNNTProjection struct {
	promptFirst, promptFirstBias   []float32
	promptSecond, promptSecondBias []float32
	encoderWeight, encoderBias     []float32
	jointWeight, jointBias         []float32
}

func LoadRNNTProjection(file *safetensors.File) (*RNNTProjection, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	load := func(name string, dims ...int) ([]float32, error) {
		values, shape, err := file.GetFloat32(name)
		if err != nil {
			return nil, err
		}
		if len(shape) != len(dims) {
			return nil, fmt.Errorf("invalid Nemotron ASR RNNT %s shape %v", name, shape)
		}
		size := 1
		for i, dim := range dims {
			if shape[i] != dim {
				return nil, fmt.Errorf("invalid Nemotron ASR RNNT %s shape %v", name, shape)
			}
			size *= dim
		}
		if len(values) != size {
			return nil, fmt.Errorf("invalid Nemotron ASR RNNT %s length", name)
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite Nemotron ASR RNNT %s", name)
			}
		}
		return values, nil
	}
	m := &RNNTProjection{}
	var err error
	if m.promptFirst, err = load("prompt_projector.linear_1.weight", rnntIntermediate, encoderWidth+rnntPrompts); err != nil {
		return nil, err
	}
	if m.promptFirstBias, err = load("prompt_projector.linear_1.bias", rnntIntermediate); err != nil {
		return nil, err
	}
	if m.promptSecond, err = load("prompt_projector.linear_2.weight", encoderWidth, rnntIntermediate); err != nil {
		return nil, err
	}
	if m.promptSecondBias, err = load("prompt_projector.linear_2.bias", encoderWidth); err != nil {
		return nil, err
	}
	if m.encoderWeight, err = load("encoder_projector.weight", rnntHidden, encoderWidth); err != nil {
		return nil, err
	}
	if m.encoderBias, err = load("encoder_projector.bias", rnntHidden); err != nil {
		return nil, err
	}
	if m.jointWeight, err = load("joint.head.weight", rnntVocabulary, rnntHidden); err != nil {
		return nil, err
	}
	if m.jointBias, err = load("joint.head.bias", rnntVocabulary); err != nil {
		return nil, err
	}
	return m, nil
}

// Project applies one prompt ID across owned-or-caller [rows,1024] states,
// returning independently owned [rows,2048], [rows,1024] and [rows,640] stages.
func (m *RNNTProjection) Project(input []float32, rows, prompt int) (first, fused, encoder []float32, err error) {
	if m == nil || len(m.promptFirst) != rnntIntermediate*(encoderWidth+rnntPrompts) || len(m.promptFirstBias) != rnntIntermediate || len(m.promptSecond) != encoderWidth*rnntIntermediate || len(m.promptSecondBias) != encoderWidth || len(m.encoderWeight) != rnntHidden*encoderWidth || len(m.encoderBias) != rnntHidden {
		return nil, nil, nil, fmt.Errorf("invalid Nemotron ASR RNNT projection weights")
	}
	if rows < 1 || rows > 5 || len(input) != rows*encoderWidth || prompt < 0 || prompt >= rnntPrompts {
		return nil, nil, nil, fmt.Errorf("invalid Nemotron ASR RNNT prompt input")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, nil, fmt.Errorf("non-finite Nemotron ASR RNNT prompt input")
		}
	}
	joined := make([]float32, rows*(encoderWidth+rnntPrompts))
	for row := 0; row < rows; row++ {
		copy(joined[row*(encoderWidth+rnntPrompts):], input[row*encoderWidth:(row+1)*encoderWidth])
		joined[row*(encoderWidth+rnntPrompts)+encoderWidth+prompt] = 1
	}
	first = make([]float32, rows*rnntIntermediate)
	if !simd.DenseNTTo(first, joined, m.promptFirst, rows, rnntIntermediate, encoderWidth+rnntPrompts, 1, encoderWidth+rnntPrompts, encoderWidth+rnntPrompts, rnntIntermediate) {
		return nil, nil, nil, fmt.Errorf("Nemotron ASR prompt first projection rejected")
	}
	activated := make([]float32, len(first))
	for i, value := range first {
		first[i] = value + m.promptFirstBias[i%rnntIntermediate]
		if math.IsNaN(float64(first[i])) || math.IsInf(float64(first[i]), 0) {
			return nil, nil, nil, fmt.Errorf("non-finite Nemotron ASR RNNT prompt first output")
		}
		activated[i] = first[i]
		if activated[i] < 0 {
			activated[i] = 0
		}
	}
	fused = make([]float32, rows*encoderWidth)
	if !simd.DenseNTTo(fused, activated, m.promptSecond, rows, encoderWidth, rnntIntermediate, 1, rnntIntermediate, rnntIntermediate, encoderWidth) {
		return nil, nil, nil, fmt.Errorf("Nemotron ASR prompt second projection rejected")
	}
	for i := range fused {
		fused[i] += m.promptSecondBias[i%encoderWidth]
		if math.IsNaN(float64(fused[i])) || math.IsInf(float64(fused[i]), 0) {
			return nil, nil, nil, fmt.Errorf("non-finite Nemotron ASR RNNT prompt output")
		}
	}
	encoder = make([]float32, rows*rnntHidden)
	// This small 5x640x1024 projection is sensitive to reduction order near
	// zero. The checked direct NT path has fewer measured outliers than the
	// blocked-FMA path against the composed prompt fixture.
	if !simd.SgemmNTTo(encoder, fused, m.encoderWeight, rows, rnntHidden, encoderWidth, 1, encoderWidth, encoderWidth, rnntHidden) {
		return nil, nil, nil, fmt.Errorf("Nemotron ASR RNNT encoder projection rejected")
	}
	for i := range encoder {
		encoder[i] += m.encoderBias[i%rnntHidden]
		if math.IsNaN(float64(encoder[i])) || math.IsInf(float64(encoder[i]), 0) {
			return nil, nil, nil, fmt.Errorf("non-finite Nemotron ASR RNNT encoder output")
		}
	}
	return first, fused, encoder, nil
}

// Joint applies the released ReLU and vocabulary projection to one fixed
// decoder vector; the vector is not an LSTM state generated by this operator.
func (m *RNNTProjection) Joint(encoder, decoder []float32, rows int) ([]float32, error) {
	if m == nil || len(m.jointWeight) != rnntVocabulary*rnntHidden || len(m.jointBias) != rnntVocabulary || rows < 1 || rows > 5 || len(encoder) != rows*rnntHidden || len(decoder) != rnntHidden {
		return nil, fmt.Errorf("invalid Nemotron ASR RNNT joint input")
	}
	for _, values := range [][]float32{encoder, decoder} {
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite Nemotron ASR RNNT joint input")
			}
		}
	}
	activated := make([]float32, len(encoder))
	for row := 0; row < rows; row++ {
		for dim := 0; dim < rnntHidden; dim++ {
			value := encoder[row*rnntHidden+dim] + decoder[dim]
			if value > 0 {
				activated[row*rnntHidden+dim] = value
			}
		}
	}
	out := make([]float32, rows*rnntVocabulary)
	if !simd.DenseNTTo(out, activated, m.jointWeight, rows, rnntVocabulary, rnntHidden, 1, rnntHidden, rnntHidden, rnntVocabulary) {
		return nil, fmt.Errorf("Nemotron ASR RNNT joint projection rejected")
	}
	for i := range out {
		out[i] += m.jointBias[i%rnntVocabulary]
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR RNNT joint output")
		}
	}
	return out, nil
}
