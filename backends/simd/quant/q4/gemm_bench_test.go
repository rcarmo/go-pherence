package q4

import "testing"

func makeBenchmarkGemmSymInputs(batch, inDim, outDim int) ([]float32, []int32, []int32, []float32) {
	x := make([]float32, batch*inDim)
	for b := 0; b < batch; b++ {
		row := x[b*inDim : (b+1)*inDim]
		for i := 0; i < inDim; i++ {
			v := float32(((b+1)*(i%31+1))%37-18) * 0.0625
			if (b+i)&1 == 1 {
				v = -v
			}
			row[i] = v
		}
	}
	qweight := make([]int32, (inDim/8)*outDim)
	for pack := 0; pack < inDim/8; pack++ {
		for j := 0; j < outDim; j++ {
			qweight[pack*outDim+j] = packQ4(
				int32((pack+j+0)&0xF),
				int32((pack+j+1)&0xF),
				int32((pack+j+2)&0xF),
				int32((pack+j+3)&0xF),
				int32((pack+j+4)&0xF),
				int32((pack+j+5)&0xF),
				int32((pack+j+6)&0xF),
				int32((pack+j+7)&0xF),
			)
		}
	}
	const groupSize = 128
	groups := inDim / groupSize
	gIdx := make([]int32, inDim)
	for i := range gIdx {
		gIdx[i] = int32(i / groupSize)
	}
	scales := make([]float32, groups*outDim)
	for g := 0; g < groups; g++ {
		row := scales[g*outDim : (g+1)*outDim]
		for j := 0; j < outDim; j++ {
			v := float32(((g+3)*(j%17+1))%29-14) * 0.0625
			if v == 0 {
				v = 0.125
			}
			row[j] = v
		}
	}
	return x, qweight, gIdx, scales
}

func gemmSymRepeatedScalar(out, x []float32, batch int, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	for b := 0; b < batch; b++ {
		gemvSymScalar(out[b*outDim:(b+1)*outDim], x[b*inDim:(b+1)*inDim], qweight, gIdx, scales, inDim, outDim)
	}
}

func benchmarkGemmSymPortableVsRepeated(b *testing.B, batch int) {
	const inDim, outDim = 1536, 2048
	x, qweight, gIdx, scales := makeBenchmarkGemmSymInputs(batch, inDim, outDim)
	bytes := int64(batch * inDim * outDim * 4)

	b.Run("portable", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gemmSymBatchedPortable(out, x, batch, qweight, gIdx, scales, inDim, outDim)
		}
	})

	b.Run("repeated_scalar_gemv", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gemmSymRepeatedScalar(out, x, batch, qweight, gIdx, scales, inDim, outDim)
		}
	})
}

func BenchmarkGemmSymBatch4_1536x2048(b *testing.B) { benchmarkGemmSymPortableVsRepeated(b, 4) }
func BenchmarkGemmSymBatch8_1536x2048(b *testing.B) { benchmarkGemmSymPortableVsRepeated(b, 8) }
