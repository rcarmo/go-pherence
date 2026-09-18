package simd

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestF16LittleEndianToF32AllPatterns(t *testing.T) {
	src := make([]byte, 0x10000*2)
	want := make([]float32, 0x10000)
	got := make([]float32, 0x10000)
	for i := range want {
		bits := uint16(i)
		binary.LittleEndian.PutUint16(src[i*2:], bits)
		want[i] = half.F16ToF32(bits)
	}
	if !F16LittleEndianToF32(got, src) {
		t.Fatal("decode rejected valid input")
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("pattern 0x%04x got=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func TestF16LittleEndianToF32Tails(t *testing.T) {
	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			src := make([]byte, n*2)
			want := make([]float32, n)
			got := make([]float32, n)
			patterns := []uint16{0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7c01, 0x7e00, 0xfe01, 0x3555, 0xc155}
			for i := range got {
				bits := patterns[i%len(patterns)] ^ uint16(i<<1)
				binary.LittleEndian.PutUint16(src[i*2:], bits)
				want[i] = half.F16ToF32(bits)
			}
			if !F16LittleEndianToF32(got, src) {
				t.Fatal("decode rejected valid input")
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("i=%d got=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
		})
	}
}

func TestF16LittleEndianToF32RejectsMalformedInput(t *testing.T) {
	dst := []float32{11, 22, 33}
	if F16LittleEndianToF32(dst, []byte{0, 0, 0}) {
		t.Fatal("accepted malformed input")
	}
	if dst[0] != 11 || dst[1] != 22 || dst[2] != 33 {
		t.Fatalf("malformed input mutated dst=%v", dst)
	}
}

func TestF16LittleEndianToF32MixedNaNBlocks(t *testing.T) {
	patterns := []uint16{
		0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0xbc00, 0x3555, 0xc155,
		0x7c01, 0x3c00, 0x7e00, 0x4000, 0xfe01, 0xc000, 0x7d55, 0x0400,
		0x7bff, 0xfbff, 0x0401, 0x3555, 0xc155, 0x0002, 0x83ff, 0x7c00,
		0x7b00,
	}
	src := make([]byte, len(patterns)*2)
	want := make([]float32, len(patterns))
	got := make([]float32, len(patterns))
	for i, bits := range patterns {
		binary.LittleEndian.PutUint16(src[i*2:], bits)
		want[i] = half.F16ToF32(bits)
	}
	if !F16LittleEndianToF32(got, src) {
		t.Fatal("decode rejected valid input")
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("i=%d bits=0x%04x got=%08x want=%08x", i, patterns[i], math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func BenchmarkF16LittleEndianToF32(b *testing.B) {
	for _, n := range []int{2816, 3584, 8192} {
		src := make([]byte, n*2)
		dst := make([]float32, n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint16(src[i*2:], half.F32ToF16(float32(i%257)*0.03125-4))
		}
		b.Run(fmt.Sprintf("dispatch/n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(src)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if !F16LittleEndianToF32(dst, src) {
					b.Fatal("decode rejected valid input")
				}
			}
		})
		b.Run(fmt.Sprintf("scalar/n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(src)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f16LittleEndianToF32Scalar(dst, src)
			}
		})
	}
}

func BenchmarkF16LittleEndianToF32FiniteLarge(b *testing.B) {
	const n = 1 << 20
	src := make([]byte, n*2)
	dst := make([]float32, n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(src[i*2:], half.F32ToF16(float32(i%257)*0.03125-4))
	}
	b.Run("dispatch/n=1048576", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(src)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !F16LittleEndianToF32(dst, src) {
				b.Fatal("decode rejected valid input")
			}
		}
	})
	b.Run("scalar/n=1048576", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(src)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			f16LittleEndianToF32Scalar(dst, src)
		}
	})
}
