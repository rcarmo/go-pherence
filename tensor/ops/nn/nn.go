package nn

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

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
	total := shapeSize(shape)
	if total < 0 || len(data) < total {
		panic("softmax: invalid backing data")
	}
	outer := total / lastDim
	out := make([]float32, total)
	if !simd.SoftmaxLastAxisTo(out, data, outer, lastDim) {
		panic("softmax: checked SIMD softmax rejected validated tensor")
	}
	return FromFloat32(out, shape)
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
	total := shapeSize(shape)
	if total < 0 || len(data) < total {
		panic("layernorm: invalid backing data")
	}
	outer := total / lastDim
	out := make([]float32, total)
	if !simd.LayerNormLastAxisTo(out, data, outer, lastDim, g, b, eps) {
		panic("layernorm: checked SIMD layernorm rejected validated tensor")
	}
	return FromFloat32(out, shape)
}

// GELU computes the GELU activation (tanh approximation).
func (t *Tensor) GELU() *Tensor {
	if t == nil {
		panic("gelu: nil tensor")
	}
	t.Realize()
	data := t.Data()
	if len(data) == 0 {
		return FromFloat32(nil, t.Shape())
	}
	out, ok := simd.GELUTanhChecked(data)
	if !ok {
		panic("gelu: checked SIMD GELU rejected validated tensor")
	}
	return FromFloat32(out, t.Shape())
}
