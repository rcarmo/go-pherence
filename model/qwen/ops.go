package qwen

import (
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func rmsNormInPlace(x, weight []float32, eps float32) { simd.RMSNorm(x, weight, eps) }
func rmsNormQwen35InPlace(x, weight []float32, eps float32, zeroCentered bool) {
	if !zeroCentered {
		rmsNormInPlace(x, weight, eps)
		return
	}
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	scale := float32(1 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := range x {
		x[i] *= scale * (1 + weight[i])
	}
}

func roundBF16InPlace(x []float32) {
	for i, v := range x {
		bits := math.Float32bits(v)
		bias := uint32(0x7fff) + ((bits >> 16) & 1)
		x[i] = math.Float32frombits((bits + bias) & 0xffff0000)
	}
}

func roundQwen35InPlace(x []float32, meta loaderconfig.QwenNativeMTPMetadata) {
	if meta.BF16Trajectory {
		roundBF16InPlace(x)
	}
}
