package simd

import (
	"testing"
	"unsafe"
)

func benchmarkSgemmNTShape(b *testing.B, blocked bool, m, n, k int) {
	a := make([]float32, m*k)
	w := make([]float32, n*k)
	c := make([]float32, m*n)
	for i := range a {
		a[i] = float32((i%31)-15) / 31
	}
	for i := range w {
		w[i] = float32((i%29)-14) / 29
	}
	b.SetBytes(int64(2 * m * n * k * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if blocked {
			SgemmNTBlockedFMA(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&w[0]), unsafe.Pointer(&c[0]), k, k, n)
		} else {
			SgemmNT(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&w[0]), unsafe.Pointer(&c[0]), k, k, n)
		}
	}
}

func BenchmarkArticleNT1x1024x1024Serial(b *testing.B) {
	benchmarkSgemmNTShape(b, false, 1, 1024, 1024)
}
func BenchmarkArticleNT1x1024x1024Blocked(b *testing.B) {
	benchmarkSgemmNTShape(b, true, 1, 1024, 1024)
}
func BenchmarkArticleNT32x1024x1024Serial(b *testing.B) {
	benchmarkSgemmNTShape(b, false, 32, 1024, 1024)
}
func BenchmarkArticleNT32x1024x1024Blocked(b *testing.B) {
	benchmarkSgemmNTShape(b, true, 32, 1024, 1024)
}
func BenchmarkArticleNT227x1024x1024Serial(b *testing.B) {
	benchmarkSgemmNTShape(b, false, 227, 1024, 1024)
}
func BenchmarkArticleNT227x1024x1024Blocked(b *testing.B) {
	benchmarkSgemmNTShape(b, true, 227, 1024, 1024)
}

func BenchmarkArticleGemmRowsPrefill(b *testing.B) {
	benchmarkArticleGemmRows(b, 227, 1024, 1024, true)
}

func BenchmarkArticleGemmRowsPrefillLegacy(b *testing.B) {
	benchmarkArticleGemmRows(b, 227, 1024, 1024, false)
}

func BenchmarkArticleGemmRowsDecode(b *testing.B) {
	benchmarkArticleGemmRows(b, 1, 1024, 1024, true)
}

func BenchmarkArticleGemmRowsWhisper(b *testing.B) {
	benchmarkArticleGemmRows(b, 1500, 1024, 1024, true)
}

func benchmarkArticleGemmRows(b *testing.B, batch, rows, cols int, dispatch bool) {
	x := make([]float32, batch*cols)
	w := make([]float32, rows*cols)
	out := make([]float32, batch*rows)
	b.SetBytes(int64(2 * batch * rows * cols * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if dispatch {
			GemmRowsParallel(out, x, w, batch, rows, cols)
		} else {
			gemmRowsParallelDots(out, x, w, batch, rows, cols)
		}
	}
}
