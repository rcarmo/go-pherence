//go:build linux && amd64

package simd

import (
	"syscall"
	"testing"
	"unsafe"
)

func TestAffineF32GuardPages(t *testing.T) {
	if !HasAffineF32Asm() {
		t.Skip("AVX2/FMA unavailable")
	}
	page := syscall.Getpagesize()
	memory, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(memory)
	if err := syscall.Mprotect(memory[:page], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mprotect(memory[2*page:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	for n := 1; n <= 65; n++ {
		for _, start := range []int{page, 2*page - n*4} {
			values := unsafe.Slice((*float32)(unsafe.Pointer(&memory[start])), n)
			for i := range values {
				values[i] = float32(i)
			}
			if !AffineF32InPlaceChecked(values, 2, 1) {
				t.Fatal("guard page call failed")
			}
			for i, v := range values {
				if v != 2*float32(i)+1 {
					t.Fatal("guard page result")
				}
			}
		}
	}
}
