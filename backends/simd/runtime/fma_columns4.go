package simd

// FMAColumns4F32Checked computes four independent output rows from one packed
// column tile. For cols=len(dst)/4 and k=len(weight)/4:
//
//	dst[row*cols+j] = sum_p x[p*cols+j] * weight[row*k+p]
//
// Each output uses one float32 FMA per p in ascending order, starting at
// positive zero. SIMD lanes span columns and the four accumulators span output
// rows, so the scalar and assembly paths have identical reduction order.
// All extents, finite inputs and output non-overlap are checked before writes.
// Empty output is accepted only when all slices are empty. No allocation.
func FMAColumns4F32Checked(dst, x, weight []float32) bool {
	if len(dst) == 0 {
		return len(x) == 0 && len(weight) == 0
	}
	if len(dst)%4 != 0 || len(weight)%4 != 0 {
		return false
	}
	cols, k := len(dst)/4, len(weight)/4
	if cols == 0 || k == 0 || len(x) != cols*k || fmaColumnsOverlap(dst, x) || fmaColumnsOverlap(dst, weight) {
		return false
	}
	for _, values := range [][]float32{x, weight} {
		for _, value := range values {
			if !affineFinite(value) {
				return false
			}
		}
	}
	return fmaColumns4F32(dst, x, weight)
}

func fmaColumns4Scalar(dst, x, weight []float32) {
	cols, k := len(dst)/4, len(weight)/4
	for row := 0; row < 4; row++ {
		for column := 0; column < cols; column++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum = FMA32Scalar(x[p*cols+column], weight[row*k+p], sum)
			}
			dst[row*cols+column] = sum
		}
	}
}
