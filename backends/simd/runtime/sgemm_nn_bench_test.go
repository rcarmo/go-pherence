package simd

import "testing"

func benchmarkSgemmNNShape(b *testing.B, parallel bool, m, n, k int) {
	a := make([]float32, m*k)
	w := make([]float32, k*n)
	c := make([]float32, m*n)
	for i := range a {
		a[i] = float32(i%31-15) / 31
	}
	for i := range w {
		w[i] = float32(i%29-14) / 29
	}
	b.SetBytes(int64(2 * m * n * k * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if parallel {
			SgemmNNParallelTo(c, a, w, m, n, k, 1, k, n, n)
		} else {
			SgemmNNTo(c, a, w, m, n, k, 1, k, n, n)
		}
	}
}
func BenchmarkArticleNN1x1024x1024(b *testing.B)   { benchmarkSgemmNNShape(b, false, 1, 1024, 1024) }
func BenchmarkArticleNN32x1024x1024(b *testing.B)  { benchmarkSgemmNNShape(b, false, 32, 1024, 1024) }
func BenchmarkArticleNN227x1024x1024(b *testing.B) { benchmarkSgemmNNShape(b, false, 227, 1024, 1024) }
func BenchmarkArticleNN1500x64x1500(b *testing.B)  { benchmarkSgemmNNShape(b, false, 1500, 64, 1500) }
func BenchmarkArticleNN32x3072x1024(b *testing.B)  { benchmarkSgemmNNShape(b, false, 32, 3072, 1024) }
func BenchmarkArticleNN32x1024x3072(b *testing.B)  { benchmarkSgemmNNShape(b, false, 32, 1024, 3072) }
func BenchmarkArticleNNParallel227x1024x1024(b *testing.B) {
	benchmarkSgemmNNShape(b, true, 227, 1024, 1024)
}
