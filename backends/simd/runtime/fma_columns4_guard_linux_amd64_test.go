//go:build linux && amd64

package simd

import (
	"syscall"
	"testing"
	"unsafe"
)

func TestFMAColumns4GuardPages(t *testing.T) {
	if !HasAffineF32Asm() {
		t.Skip("AVX2/FMA unavailable")
	}
	page := syscall.Getpagesize()
	alloc := func() []byte {
		m, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = syscall.Munmap(m) })
		for _, guard := range [][]byte{m[:page], m[2*page:]} {
			if err := syscall.Mprotect(guard, syscall.PROT_NONE); err != nil {
				t.Fatal(err)
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
				d, x, w := view(dm, 4*cols), view(xm, cols*k), view(wm, 4*k)
				for i := range x {
					x[i] = float32(i%13) - 6
				}
				for i := range w {
					w[i] = float32(i%17) + .5
				}
				want := make([]float32, len(d))
				fmaColumns4Scalar(want, x, w)
				if !FMAColumns4F32Checked(d, x, w) {
					t.Fatal("rejected")
				}
				for i := range d {
					if d[i] != want[i] {
						t.Fatal("result", cols, k, atEnd, i)
					}
				}
			}
		}
	}
}
