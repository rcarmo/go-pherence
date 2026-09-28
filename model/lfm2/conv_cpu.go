package lfm2

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type lfm2Linear struct {
	weight        []float32
	bias          []float32
	inDim, outDim int
}

func loadLFM2Linear(src Float32TensorSource, prefix string, inDim, outDim int, bias bool) (lfm2Linear, error) {
	weight, err := loadLFM2Tensor(src, prefix+".weight", []int{outDim, inDim})
	if err != nil {
		return lfm2Linear{}, err
	}
	l := lfm2Linear{weight: weight, inDim: inDim, outDim: outDim}
	if bias {
		l.bias, err = loadLFM2Tensor(src, prefix+".bias", []int{outDim})
		if err != nil {
			return lfm2Linear{}, err
		}
	}
	return l, nil
}

func (l lfm2Linear) forward(dst, input []float32) error {
	if len(dst) != l.outDim || len(input) != l.inDim || !simd.GemvRows(dst, input, l.weight, l.outDim, l.inDim) {
		return fmt.Errorf("invalid LFM2 linear buffers out/in=%d/%d want %d/%d", len(dst), len(input), l.outDim, l.inDim)
	}
	if len(l.bias) != 0 {
		if len(l.bias) != len(dst) {
			return fmt.Errorf("invalid LFM2 linear bias=%d want %d", len(l.bias), len(dst))
		}
		for i := range dst {
			dst[i] += l.bias[i]
		}
	}
	return nil
}

// ShortConvCPU is one LFM2 depthwise causal-convolution operator. Its weights
// are immutable; ForwardToken returns a new state and never mutates the caller's
// cache.
type ShortConvCPU struct {
	cfg      Config
	layer    int
	in, out  lfm2Linear
	kernel   []float32
	convBias []float32
}

func LoadShortConvCPU(src Float32TensorSource, cfg Config, layer int) (*ShortConvCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil LFM2 convolution tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if layer < 0 || layer >= len(cfg.LayerTypes) || cfg.LayerTypes[layer] != "conv" {
		return nil, fmt.Errorf("LFM2 layer %d is not a convolution layer", layer)
	}
	projectedWidth := sizeProduct(3, cfg.HiddenSize)
	if projectedWidth < 0 {
		return nil, fmt.Errorf("LFM2 convolution projection width overflows")
	}
	prefix := fmt.Sprintf("model.layers.%d.conv", layer)
	m := &ShortConvCPU{cfg: cfg, layer: layer}
	var err error
	if m.in, err = loadLFM2Linear(src, prefix+".in_proj", cfg.HiddenSize, projectedWidth, cfg.ConvBias); err != nil {
		return nil, err
	}
	if m.out, err = loadLFM2Linear(src, prefix+".out_proj", cfg.HiddenSize, cfg.HiddenSize, cfg.ConvBias); err != nil {
		return nil, err
	}
	if m.kernel, err = loadLFM2Tensor(src, prefix+".conv.weight", []int{cfg.HiddenSize, 1, cfg.ConvLCache}); err != nil {
		return nil, err
	}
	if cfg.ConvBias {
		if m.convBias, err = loadLFM2Tensor(src, prefix+".conv.bias", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func (m *ShortConvCPU) NewState() []float32 {
	if m == nil {
		return nil
	}
	n := sizeProduct(m.cfg.HiddenSize, m.cfg.ConvLCache)
	if n < 0 {
		return nil
	}
	return make([]float32, n)
}

// ForwardToken executes the cached single-token branch of Lfm2MoeShortConv.
// State is channel-major [hidden, conv_L_cache] and the returned state includes
// the current projected B*x value.
func (m *ShortConvCPU) ForwardToken(input, state []float32) ([]float32, []float32, error) {
	if m == nil {
		return nil, nil, fmt.Errorf("nil LFM2 CPU convolution")
	}
	h, k := m.cfg.HiddenSize, m.cfg.ConvLCache
	stateLen := sizeProduct(h, k)
	projectedLen := sizeProduct(3, h)
	if stateLen < 0 || projectedLen < 0 || len(input) != h || len(state) != stateLen {
		return nil, nil, fmt.Errorf("invalid LFM2 convolution input/state=%d/%d want %d/%d", len(input), len(state), h, stateLen)
	}
	projected := make([]float32, projectedLen)
	if err := m.in.forward(projected, input); err != nil {
		return nil, nil, err
	}
	b, c, x := projected[:h], projected[h:2*h], projected[2*h:]
	next := append([]float32(nil), state...)
	conv := make([]float32, h)
	for channel := 0; channel < h; channel++ {
		row := next[channel*k : (channel+1)*k]
		copy(row, row[1:])
		row[k-1] = b[channel] * x[channel]
		var sum float32
		weights := m.kernel[channel*k : (channel+1)*k]
		for i := 0; i < k; i++ {
			sum += row[i] * weights[i]
		}
		if len(m.convBias) != 0 {
			sum += m.convBias[channel]
		}
		conv[channel] = c[channel] * sum
	}
	output := make([]float32, h)
	if err := m.out.forward(output, conv); err != nil {
		return nil, nil, err
	}
	return output, next, nil
}
