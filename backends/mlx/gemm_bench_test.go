package mlx

import "testing"

func benchmarkGemmVsRepeatedGemvTo(b *testing.B, batch int) {
	const (
		inDim     = 1536
		outDim    = 2048
		groupSize = 64
	)
	qw := makeBenchMLXWeight(outDim, inDim, groupSize)
	x := makeGemmInput(batch, inDim)
	bytes := int64(batch * inDim * outDim * 4)

	b.Run("batched_q4", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !Gemm(out, x, batch, qw) {
				b.Fatal("Gemm returned false")
			}
		}
	})

	b.Run("repeated_gemv_to", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !repeatedGemvTo(out, x, batch, qw) {
				b.Fatal("repeated GemvTo returned false")
			}
		}
	})
}

func BenchmarkGemmBatch4_1536x2048(b *testing.B) { benchmarkGemmVsRepeatedGemvTo(b, 4) }
func BenchmarkGemmBatch8_1536x2048(b *testing.B) { benchmarkGemmVsRepeatedGemvTo(b, 8) }
