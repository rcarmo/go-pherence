package nvfp4

import "testing"

func makeBenchmarkGemmNVFP4Inputs(batch, inDim, outDim int) ([]float32, *NVFP4Weight) {
	groups := inDim / 16
	qw := &NVFP4Weight{
		Weight:       make([]byte, outDim*(inDim/2)),
		WeightScale:  make([]byte, outDim*groups),
		WeightScale2: 0.625,
		OutDim:       outDim,
		InDim:        inDim,
		Groups:       groups,
		GroupSize:    16,
	}
	x := make([]float32, batch*inDim)
	for b := 0; b < batch; b++ {
		row := x[b*inDim : (b+1)*inDim]
		for i := 0; i < inDim; i++ {
			v := float32(((b+3)*(i%29+1))%41-20) * 0.0625
			if (b+i)&1 != 0 {
				v = -v
			}
			row[i] = v
		}
	}
	scalePattern := []byte{0x28, 0x30, 0x34, 0x38, 0x3c, 0x40, 0x44, 0xb0, 0xb8}
	packedPerRow := inDim / 2
	for row := 0; row < outDim; row++ {
		packed := qw.Weight[row*packedPerRow : (row+1)*packedPerRow]
		for i := 0; i < packedPerRow; i++ {
			lo := byte((row + i*3) & 0x0f)
			hi := byte((row*5 + i + 7) & 0x0f)
			packed[i] = lo | (hi << 4)
		}
		scales := qw.WeightScale[row*groups : (row+1)*groups]
		for g := 0; g < groups; g++ {
			scales[g] = scalePattern[(row+g)%len(scalePattern)]
		}
	}
	return x, qw
}

func gemmNVFP4RepeatedScalarRows(out, x []float32, batch int, qw *NVFP4Weight) {
	for b := 0; b < batch; b++ {
		gemvNVFP4Rows(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw, 0, qw.OutDim)
	}
}

func benchmarkGemmNVFP4PortableVsRepeated(b *testing.B, batch int) {
	const inDim, outDim = 1536, 2048
	x, qw := makeBenchmarkGemmNVFP4Inputs(batch, inDim, outDim)
	bytes := int64(batch * inDim * outDim * 4)

	b.Run("batched", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !GemmNVFP4(out, x, batch, qw) {
				b.Fatal("GemmNVFP4 returned false")
			}
		}
	})

	b.Run("repeated_scalar_rows", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gemmNVFP4RepeatedScalarRows(out, x, batch, qw)
		}
	})
}

func BenchmarkGemmNVFP4Batch4_1536x2048(b *testing.B) { benchmarkGemmNVFP4PortableVsRepeated(b, 4) }
func BenchmarkGemmNVFP4Batch8_1536x2048(b *testing.B) { benchmarkGemmNVFP4PortableVsRepeated(b, 8) }
