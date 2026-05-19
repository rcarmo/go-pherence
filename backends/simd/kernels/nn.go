package kernels

import "math"

func SoftmaxLastAxis(data []float32, shape []int) []float32 {
	ndim := len(shape)
	if ndim == 0 {
		panic("softmax: scalar tensor")
	}
	lastDim := shape[ndim-1]
	if lastDim <= 0 {
		return nil
	}
	total := 1
	for _, d := range shape {
		if d < 0 || (d != 0 && total > int(^uint(0)>>1)/d) {
			panic("softmax: invalid shape")
		}
		total *= d
	}
	if total < 0 || len(data) < total {
		panic("softmax: invalid backing data")
	}
	outerSize := total / lastDim

	out := make([]float32, total)
	for i := 0; i < outerSize; i++ {
		off := i * lastDim
		row := data[off : off+lastDim]

		mx := row[0]
		for _, v := range row[1:] {
			if v > mx {
				mx = v
			}
		}
		sum := float32(0)
		for j, v := range row {
			e := float32(math.Exp(float64(v - mx)))
			out[off+j] = e
			sum += e
		}
		inv := 1.0 / sum
		for j := 0; j < lastDim; j++ {
			out[off+j] *= inv
		}
	}
	return out
}

func LayerNormLastAxis(data []float32, shape []int, gamma, beta []float32, eps float32) []float32 {
	if len(shape) == 0 {
		panic("layernorm: scalar tensor")
	}
	lastDim := shape[len(shape)-1]
	if lastDim <= 0 {
		return nil
	}
	total := 1
	for _, d := range shape {
		if d < 0 || (d != 0 && total > int(^uint(0)>>1)/d) {
			panic("layernorm: invalid shape")
		}
		total *= d
	}
	if total < 0 || len(data) < total {
		panic("layernorm: invalid backing data")
	}
	if (gamma == nil) != (beta == nil) {
		panic("layernorm: gamma and beta must both be present or nil")
	}
	if gamma != nil && (len(gamma) < lastDim || len(beta) < lastDim) {
		panic("layernorm: gamma/beta length mismatch")
	}
	outerSize := total / lastDim
	out := make([]float32, total)
	for i := 0; i < outerSize; i++ {
		off := i * lastDim
		row := data[off : off+lastDim]

		mean := float32(0)
		for _, v := range row {
			mean += v
		}
		mean /= float32(lastDim)

		variance := float32(0)
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(lastDim)
		stdInv := float32(1.0 / math.Sqrt(float64(variance+eps)))

		for j := 0; j < lastDim; j++ {
			v := (row[j] - mean) * stdInv
			if gamma != nil {
				v = gamma[j]*v + beta[j]
			}
			out[off+j] = v
		}
	}
	return out
}

func GELU(data []float32) []float32 {
	out := make([]float32, len(data))
	const c = float32(0.7978845608)
	for i, v := range data {
		arg := c * (v + 0.044715*v*v*v)
		var tanh float32
		if arg < -5 {
			tanh = -1
		} else if arg > 5 {
			tanh = 1
		} else {
			a2 := arg * arg
			tanh = arg * (135135 + a2*(17325+a2*(378+a2))) / (135135 + a2*(62370+a2*(3150+a2*28)))
		}
		out[i] = 0.5 * v * (1 + tanh)
	}
	return out
}
