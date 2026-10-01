package simd

// Sdot32 preserves a four-accumulator, eight-lane FMA reduction: vector groups
// combine as (a+c)+(b+d), then lane halves and adjacent pairs. This is the AVX
// reduction contract used by the pinned GGML F32/F16 dot kernels. Callers of
// F16 kernels must round inputs before this F32 operation. Unlike Sdot, scalar
// leftovers use separately rounded F32 products accumulated in F64.
// Mismatched lengths panic rather than silently truncate model geometry.
func Sdot32(x, y []float32) float32 {
	if len(x) != len(y) {
		panic("Sdot32: mismatched lengths")
	}
	return sdot32(x, y)
}

func sdot32Scalar(x, y []float32) float32 {
	var acc [4][8]float32
	n := len(x) &^ 31
	for i := 0; i < n; i += 32 {
		for j := 0; j < 4; j++ {
			for k := 0; k < 8; k++ {
				p := i + j*8 + k
				acc[j][k] = FMA32Scalar(x[p], y[p], acc[j][k])
			}
		}
	}
	var v [8]float32
	for k := range v {
		v[k] = float32(acc[0][k]+acc[2][k]) + float32(acc[1][k]+acc[3][k])
	}
	var half [4]float32
	for k := range half {
		half[k] = v[k] + v[k+4]
	}
	sum := float64(float32(half[0]+half[1]) + float32(half[2]+half[3]))
	for i := n; i < len(x); i++ {
		product := x[i] * y[i]
		sum += float64(product)
	}
	return float32(sum)
}
