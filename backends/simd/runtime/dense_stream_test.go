package simd

import (
	"math"
	"runtime"
	"sync"
	"testing"
	"unsafe"
)

func TestDenseStreamingParallelKeepsBlockedBits(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("requires SIMD blocked kernel")
	}
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	for _, shape := range [][3]int{{4, 1024, 1024}, {4, 4096, 1024}, {4, 1024, 4096}, {5, 1031, 1027}, {2, 2048, 1024}} {
		m, n, k := shape[0], shape[1], shape[2]
		a := make([]float32, m*k)
		b := make([]float32, n*k)
		c := make([]float32, m*n)
		for i := range a {
			a[i] = float32(i%19-9) / 17
		}
		for i := range b {
			b[i] = float32(i%31-15) / 29
		}
		for i := range c {
			c[i] = float32(i%7-3) / 9
		}
		want := append([]float32(nil), c...)
		SgemmNTBlockedFMA(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&want[0]), k, k, n)
		if !DenseNTTo(c, a, b, m, n, k, 1, k, k, n) {
			t.Fatal(shape)
		}
		for i := range c {
			if math.Float32bits(c[i]) != math.Float32bits(want[i]) {
				t.Fatalf("shape=%v idx=%d got=%g want=%g", shape, i, c[i], want[i])
			}
		}
	}
}
func TestDenseStreamingParallelIndependentCalls(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("requires SIMD")
	}
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	const m, n, k = 4, 1024, 1024
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	for i := range a {
		a[i] = .2
	}
	for i := range b {
		b[i] = .1
	}
	want := make([]float32, m*n)
	SgemmNTBlockedFMA(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&want[0]), k, k, n)
	var wg sync.WaitGroup
	for task := 0; task < 3; task++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c := make([]float32, m*n)
			if !DenseNTTo(c, a, b, m, n, k, 1, k, k, n) {
				t.Error("rejected")
			}
			for i := range c {
				if math.Float32bits(c[i]) != math.Float32bits(want[i]) {
					t.Error("parallel result changed")
					break
				}
			}
		}()
	}
	wg.Wait()
}
func BenchmarkDenseStreamingFFN(b *testing.B) {
	const m, n, k = 4, 4096, 1024
	a := make([]float32, m*k)
	w := make([]float32, n*k)
	c := make([]float32, m*n)
	for i := range a {
		a[i] = .3
	}
	for i := range w {
		w[i] = .2
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clear(c)
		if !DenseNTTo(c, a, w, m, n, k, 1, k, k, n) {
			b.Fatal("dispatch rejected")
		}
	}
}
