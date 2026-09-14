//go:build linux && amd64

package simd

import (
	"math"
	"syscall"
	"testing"
	"unsafe"
)

func TestFMAMatrixGuardPages(t *testing.T) {
	if !HasAffineF32Asm() {
		t.Skip("AVX2/FMA unavailable")
	}
	page := syscall.Getpagesize()
	alloc := func() []byte {
		m, e := syscall.Mmap(-1, 0, 3*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(func() { syscall.Munmap(m) })
		for _, p := range [][]byte{m[:page], m[2*page:]} {
			if e := syscall.Mprotect(p, syscall.PROT_NONE); e != nil {
				t.Fatal(e)
			}
		}
		return m
	}
	am, bm, cm := alloc(), alloc(), alloc()
	for _, m := range []int{1, 2, 3} {
		for n := 1; n <= 65; n++ {
			for _, k := range []int{1, 3, 7} {
				for _, end := range []bool{false, true} {
					view := func(mem []byte, size int) []float32 {
						start := page
						if end {
							start = 2*page - 4*size
						}
						return unsafe.Slice((*float32)(unsafe.Pointer(&mem[start])), size)
					}
					a, b, c := view(am, m*k), view(bm, k*n), view(cm, m*n)
					for i := range a {
						a[i] = float32(i%13) - 6
					}
					for i := range b {
						b[i] = float32(i%17) * .25
					}
					want := make([]float32, m*n)
					fmaMatrixScalar(want, a, b, m, n, k)
					if !FMAMatrixF32Checked(c, a, b, m, n, k) {
						t.Fatal("admission")
					}
					for i, v := range c {
						if math.Float32bits(v) != math.Float32bits(want[i]) {
							t.Fatal("guard output")
						}
					}
				}
			}
		}
	}
}
