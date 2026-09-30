package simd

func attentionValueRowScalar(dst, probabilities, values []float32, rows, width int) {
	for dim := 0; dim < width; dim++ {
		var sum float32
		for source := 0; source < rows; source++ {
			product := float32(probabilities[source] * values[source*width+dim])
			sum += product
		}
		dst[dim] = sum
	}
}
