//go:build riscv64

package q4

func dequantAccelerated(out []float32, qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) {
	for j := 0; j < outFeatures; j++ {
		row := out[j*inFeatures : (j+1)*inFeatures]
		if sym {
			dequantSymOutputRow(row, qweight, gIdx, scales, inFeatures, outFeatures, j)
		} else {
			dequantOutputRow(row, qweight, qzeros, gIdx, scales, inFeatures, outFeatures, j)
		}
	}
}
