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
	total := shapeSizeChecked("softmax", shape)
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
