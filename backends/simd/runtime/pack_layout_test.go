package simd

import (
	"fmt"
	"math"
	"testing"
)

func TestPackBNTBitwiseLayout(t *testing.T) {
	bits := []uint32{0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f801234, 1, 0x807fffff, 0x3f800000}
	for _, k := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 127, 128, 129, 1024, 3072} {
		for _, nr := range []int{1, 7, 15, 16} {
			for _, pad := range []int{0, 3} {
				t.Run(fmt.Sprintf("k%d/nr%d/pad%d", k, nr, pad), func(t *testing.T) {
					const jj = 2
					ldb := k + pad
					storage := make([]float32, (jj+nr-1)*ldb+k+1)
					src := storage[1:] // deliberately unaligned input, exact final footprint
					for i := range src {
						src[i] = math.Float32frombits(bits[i%len(bits)] ^ uint32(i%3))
					}
					// Offset output ensures unaligned writes and detects over/under-write.
					out := make([]float32, k*gebpNR+2)
					for i := range out {
						out[i] = 12345
					}
					packBNT(src, ldb, jj, nr, k, out[1:len(out)-1])
					for p := 0; p < k; p++ {
						for j := 0; j < gebpNR; j++ {
							var want uint32
							if j < nr {
								want = math.Float32bits(src[(jj+j)*ldb+p])
							}
							if got := math.Float32bits(out[1+p*gebpNR+j]); got != want {
								t.Fatalf("p%d j%d got %08x want %08x", p, j, got, want)
							}
						}
					}
					if out[0] != 12345 || out[len(out)-1] != 12345 {
						t.Fatal("output guard overwritten")
					}
				})
			}
		}
	}
}

func BenchmarkPackBNT(b *testing.B) {
	for _, k := range []int{128, 1024, 3072} {
		src := make([]float32, k*gebpNR)
		dst := make([]float32, k*gebpNR)
		for i := range src {
			src[i] = float32(i)
		}
		b.Run(fmt.Sprintf("scalar/%d", k), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				packBNTScalar(src, k, 0, gebpNR, k, dst)
			}
		})
		b.Run(fmt.Sprintf("dispatch/%d", k), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				packBNT(src, k, 0, gebpNR, k, dst)
			}
		})
	}
}
