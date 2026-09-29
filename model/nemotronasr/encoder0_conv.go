package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const encoderConvKernel = 9

// Encoder0Convolution owns the first ASR encoder block's convolution module.
// The bounded offline path has no padding cache; streaming, subsequent FF2
// and the remaining encoder/RNN-T are separate contracts.
type Encoder0Convolution struct {
	preGamma, preBeta     []float32
	point1                []float32 // [2048,1024], no bias
	depth                 []float32 // [1024,9], causal left padding
	depthGamma, depthBeta []float32
	point2                []float32 // [1024,1024], no bias
}

func LoadEncoder0Convolution(file *safetensors.File) (*Encoder0Convolution, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	prefix := "encoder.layers.0."
	load := func(name string, dims ...int) ([]float32, error) {
		values, shape, err := file.GetFloat32(prefix + name)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if len(shape) != len(dims) {
			return nil, fmt.Errorf("%s: invalid rank %v", name, shape)
		}
		size := 1
		for i, dim := range dims {
			if shape[i] != dim {
				return nil, fmt.Errorf("%s: invalid shape %v", name, shape)
			}
			size *= dim
		}
		if len(values) != size {
			return nil, fmt.Errorf("%s: invalid element count", name)
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("%s: non-finite weight", name)
			}
		}
		return values, nil
	}
	m := &Encoder0Convolution{}
	var err error
	if m.preGamma, err = load("norm_conv.weight", encoderWidth); err != nil {
		return nil, err
	}
	if m.preBeta, err = load("norm_conv.bias", encoderWidth); err != nil {
		return nil, err
	}
	if m.point1, err = load("conv.pointwise_conv1.weight", 2*encoderWidth, encoderWidth, 1); err != nil {
		return nil, err
	}
	if m.depth, err = load("conv.depthwise_conv.weight", encoderWidth, 1, encoderConvKernel); err != nil {
		return nil, err
	}
	if m.depthGamma, err = load("conv.norm.weight", encoderWidth); err != nil {
		return nil, err
	}
	if m.depthBeta, err = load("conv.norm.bias", encoderWidth); err != nil {
		return nil, err
	}
	if m.point2, err = load("conv.pointwise_conv2.weight", encoderWidth, encoderWidth, 1); err != nil {
		return nil, err
	}
	return m, nil
}

// ForwardOffline returns owned convolution output and post-convolution
// residual from a whole five-row unmasked FF1+attention window.
func (m *Encoder0Convolution) ForwardOffline(input []float32, rows int) (output, residual []float32, err error) {
	if m == nil || len(m.preGamma) != encoderWidth || len(m.preBeta) != encoderWidth || len(m.point1) != 2*encoderWidth*encoderWidth || len(m.depth) != encoderWidth*encoderConvKernel || len(m.depthGamma) != encoderWidth || len(m.depthBeta) != encoderWidth || len(m.point2) != encoderWidth*encoderWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 convolution weights")
	}
	if rows < 1 || rows > 5 || len(input) != rows*encoderWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 convolution window")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, fmt.Errorf("non-finite Nemotron ASR encoder-0 convolution input")
		}
	}
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, rows, encoderWidth, m.preGamma, m.preBeta, 1e-5) {
		return nil, nil, fmt.Errorf("Nemotron ASR encoder-0 convolution pre-norm rejected shape")
	}
	point := make([]float32, rows*2*encoderWidth)
	if !simd.DenseNTTo(point, normal, m.point1, rows, 2*encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, 2*encoderWidth) {
		return nil, nil, fmt.Errorf("Nemotron ASR encoder-0 pointwise1 rejected shape")
	}
	glu := make([]float32, len(input))
	for row := 0; row < rows; row++ {
		for ch := 0; ch < encoderWidth; ch++ {
			gate := point[row*2*encoderWidth+encoderWidth+ch]
			glu[row*encoderWidth+ch] = point[row*2*encoderWidth+ch] / (1 + float32(math.Exp(float64(-gate))))
		}
	}
	depth := make([]float32, len(input))
	for row := 0; row < rows; row++ {
		for ch := 0; ch < encoderWidth; ch++ {
			var sum float32
			for tap := 0; tap < encoderConvKernel; tap++ {
				source := row + tap - (encoderConvKernel - 1)
				if source >= 0 {
					sum += glu[source*encoderWidth+ch] * m.depth[ch*encoderConvKernel+tap]
				}
			}
			depth[row*encoderWidth+ch] = sum
		}
	}
	depthNormal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(depthNormal, depth, rows, encoderWidth, m.depthGamma, m.depthBeta, 1e-5) {
		return nil, nil, fmt.Errorf("Nemotron ASR encoder-0 depth norm rejected shape")
	}
	if !simd.SiLUTo(depthNormal, depthNormal) {
		return nil, nil, fmt.Errorf("Nemotron ASR encoder-0 activation rejected shape")
	}
	output = make([]float32, len(input))
	if !simd.DenseNTTo(output, depthNormal, m.point2, rows, encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderWidth) {
		return nil, nil, fmt.Errorf("Nemotron ASR encoder-0 pointwise2 rejected shape")
	}
	residual = make([]float32, len(input))
	for i, value := range output {
		residual[i] = input[i] + value
	}
	return output, residual, nil
}
