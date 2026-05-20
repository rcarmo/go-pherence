package kernels

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
