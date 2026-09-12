//go:build linux && amd64

package simd

import (
	"syscall"
	"testing"
	"unsafe"
)

func TestFMAColumnsGuardPages(t *testing.T) {
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
	dm, xm, wm := alloc(), alloc(), alloc()
	for cols := 1; cols <= 65; cols++ {
		for _, k := range []int{1, 3, 7} {
			for _, atEnd := range []bool{false, true} {
				view := func(m []byte, n int) []float32 {
					start := page
					if atEnd {
						start = 2*page - 4*n
					}
					return unsafe.Slice((*float32)(unsafe.Pointer(&m[start])), n)
				}
				d, x, w := view(dm, cols), view(xm, cols*k), view(wm, k)
				for i := range x {
					x[i] = float32(i%13) - 6
				}
				for i := range w {
					w[i] = float32(i) + .5
				}
				want := make([]float32, cols)
				fmaColumnsScalar(want, x, w)
				if !FMAColumnsF32Checked(d, x, w) {
					t.Fatal("rejected")
				}
				for i := range d {
					if d[i] != want[i] {
						t.Fatal("result")
					}
				}
			}
		}
	}
}
