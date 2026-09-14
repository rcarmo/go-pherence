//go:build linux && amd64

package simd

import (
	"golang.org/x/sys/unix"
	"math"
	"runtime"
	"testing"
	"unsafe"
)

func TestPackBNTGuardedRows(t *testing.T) {
	if !hasAvxPack {
		t.Skip("AVX packing unavailable")
	}
	for _, k := range []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17} {
		page := unix.Getpagesize()
		mapped := make([][]byte, 0, 17)
		alloc := func(size int) []float32 {
			mem, err := unix.Mmap(-1, 0, 2*page, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_PRIVATE|unix.MAP_ANON)
			if err != nil {
				t.Fatal(err)
			}
			mapped = append(mapped, mem)
			t.Cleanup(func() {
				if err := unix.Munmap(mem); err != nil {
					t.Error(err)
				}
			})
			if err := unix.Mprotect(mem[page:], unix.PROT_NONE); err != nil {
				t.Fatal(err)
			}
			return unsafe.Slice((*float32)(unsafe.Pointer(&mem[page-size*4])), size)
		}
		var rows [16][]float32
		var p [16]uintptr
		for i := range rows {
			rows[i] = alloc(k)
			for j := range rows[i] {
				rows[i][j] = math.Float32frombits(0x3f800000 + uint32(i*k+j))
			}
			p[i] = uintptr(unsafe.Pointer(&rows[i][0]))
		}
		out := alloc(k * 16)
		packBNTAsm(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], k, uintptr(unsafe.Pointer(&out[0])))
		for j := 0; j < k; j++ {
			for i := 0; i < 16; i++ {
				if out[j*16+i] != rows[i][j] {
					t.Fatalf("k%d row%d col%d mismatch", k, i, j)
				}
			}
		}
		runtime.KeepAlive(mapped)
	}
}
