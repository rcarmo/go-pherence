package omnivoice

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"testing"
)

func TestCodecGEMMOverwriteDirtyBuffers(t *testing.T) {
	for _, shape := range [][3]int{{1, 1, 1}, {5, 7, 3}, {6, 16, 7}, {7, 17, 9}, {32, 64, 128}, {32, 65, 128}} {
		m, n, k := shape[0], shape[1], shape[2]
		a, b, want := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
		for i := range a {
			a[i] = float32(i%17-8) * .1
		}
		for i := range b {
			b[i] = float32(i%13-6) * .2
		}
		if !simd.SgemmNNTo(want, a, b, m, n, k, 1, k, n, n) {
			t.Fatal("reference shape")
		}
		for _, prepared := range []bool{false, true} {
			d := &CodecDecoder{}
			if prepared {
				d.scratch = &codecScratch{gemmPanel: make([]float32, k*16)}
			}
			got := make([]float32, m*n+1)
			got[m*n] = 123
			for repeat := 0; repeat < 3; repeat++ {
				for i := 0; i < m*n; i++ {
					got[i] = float32(math.NaN())
				}
				if !d.gemm(got[:m*n], a, b, m, n, k) {
					t.Fatal("shape")
				}
				assertFloat32Exact(t, got[:m*n], want)
				if got[m*n] != 123 {
					t.Fatal("tail sentinel")
				}
			}
			if allocs := testing.AllocsPerRun(3, func() { d.gemm(got[:m*n], a, b, m, n, k) }); allocs != 0 {
				t.Fatal(allocs)
			}
		}
	}
}
