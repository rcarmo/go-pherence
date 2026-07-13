package simd

func ddotGo(x, y []float64) float64 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += x[i] * y[i]
	}
	return sum
}
