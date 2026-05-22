//go:build !riscv64

package q4

func dequantAccelerated(out []float32, qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) {
	dequantToScalar(out, qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym)
}
