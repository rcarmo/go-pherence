package tensor

import "github.com/rcarmo/go-pherence/backends/simd/kernels"

// Softmax computes softmax along the last axis.
func (t *Tensor) Softmax() *Tensor {
	if t == nil {
		panic("softmax: nil tensor")
	}
	t.Realize()
	data := t.Data()
	shape := t.Shape()
	ndim := len(shape)
	if ndim == 0 {
		panic("softmax: scalar tensor")
	}
	lastDim := shape[ndim-1]
	if lastDim <= 0 {
		return FromFloat32(nil, shape)
	}
	return FromFloat32(kernels.SoftmaxLastAxis(data, shape), shape)
}

// LayerNorm computes layer normalization along the last axis.
func (t *Tensor) LayerNorm(gamma, beta *Tensor, eps float32) *Tensor {
	if t == nil {
		panic("layernorm: nil tensor")
	}
	t.Realize()
	data := t.Data()
	shape := t.Shape()
	if len(shape) == 0 {
		panic("layernorm: scalar tensor")
	}
	lastDim := shape[len(shape)-1]
	if lastDim <= 0 {
		return FromFloat32(nil, shape)
	}

	var g, b []float32
	if gamma != nil {
		if dims := gamma.Shape(); len(dims) != 1 || dims[0] != lastDim {
			panic("layernorm: gamma shape mismatch")
		}
		g = gamma.Data()
	}
	if beta != nil {
		if dims := beta.Shape(); len(dims) != 1 || dims[0] != lastDim {
			panic("layernorm: beta shape mismatch")
		}
		b = beta.Data()
	}
	return FromFloat32(kernels.LayerNormLastAxis(data, shape, g, b, eps), shape)
}

// GELU computes the GELU activation (tanh approximation).
func (t *Tensor) GELU() *Tensor {
	if t == nil {
		panic("gelu: nil tensor")
	}
	t.Realize()
	data := t.Data()
	return FromFloat32(kernels.GELU(data), t.Shape())
}
