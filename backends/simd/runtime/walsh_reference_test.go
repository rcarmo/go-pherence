package simd

import (
	"math"
	"math/bits"
	"slices"
	"testing"
)

func TestWalsh128Reference(t *testing.T) {
	var input, output, roundtrip [128]float32
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.71)) * 0.03
	}
	if !Walsh128ReferenceTo(output[:], input[:]) {
		t.Fatal("rejected group")
	}
	for j, v := range output {
		var want float64
		for i, x := range input {
			sign := 1.
			if bits.OnesCount(uint(i&j))%2 != 0 {
				sign = -1
			}
			want += float64(x) * sign / math.Sqrt(128)
		}
		if math.Abs(float64(v)-want) > 2e-7 {
			t.Fatalf("coefficient %d: %g != %g", j, v, want)
		}
	}
	if !Walsh128ReferenceTo(roundtrip[:], output[:]) {
		t.Fatal("inverse rejected")
	}
	for i, v := range roundtrip {
		if math.Abs(float64(v-input[i])) > 2e-7 {
			t.Fatalf("roundtrip %d", i)
		}
	}
	for _, n := range []int{0, 127} {
		dst := slices.Repeat([]float32{7}, 128)
		if Walsh128ReferenceTo(dst, input[:n]) || !slices.Equal(dst, slices.Repeat([]float32{7}, 128)) {
			t.Fatal("short source wrote")
		}
		if Walsh128ReferenceTo(dst[:n], input[:]) {
			t.Fatal("short dest accepted")
		}
	}
	original := input
	if Walsh128ReferenceTo(input[:], input[:]) || input != original {
		t.Fatal("in-place accepted or mutated")
	}
	var overlap [129]float32
	if Walsh128ReferenceTo(overlap[1:], overlap[:128]) {
		t.Fatal("partial overlap accepted")
	}
}

func BenchmarkWalsh128(b *testing.B) {
	var original, out [128]float32
	for i := range original {
		original[i] = float32(i%17-8) / 256
	}
	b.Run("packed-butterfly", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			out = original
			cqWalsh128(out[:])
		}
	})
	b.Run("dense-reference", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if !Walsh128ReferenceTo(out[:], original[:]) {
				b.Fatal("rejected")
			}
		}
	})
}
