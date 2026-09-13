//go:build amd64

package simd

import (
	"encoding/binary"
	"github.com/rcarmo/go-pherence/half"
	"math"
	"testing"
)

func TestF16LittleEndianToF32AsmStopsBeforeNaNBlock(t *testing.T) {
	if !hasF16CConvert {
		t.Skip("F16C unavailable")
	}
	patterns := []uint16{
		0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0xbc00, 0x3555, 0xc155,
		0x7c01, 0x3c00, 0x7e00, 0x4000, 0xfe01, 0xc000, 0x7d55, 0x0400,
		0x7bff, 0xfbff, 0x0401, 0x3555, 0xc155, 0x0002, 0x83ff, 0x7c00,
	}
	src := make([]byte, len(patterns)*2)
	got := make([]float32, len(patterns))
	for i := range got {
		got[i] = -123.5
	}
	for i, bits := range patterns {
		binary.LittleEndian.PutUint16(src[i*2:], bits)
	}
	if n := f16LittleEndianToF32Asm(got, src); n != 8 {
		t.Fatalf("processed=%d want 8", n)
	}
	for i, bits := range patterns[:8] {
		if math.Float32bits(got[i]) != math.Float32bits(half.F16ToF32(bits)) {
			t.Fatalf("prefix i=%d got=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(half.F16ToF32(bits)))
		}
	}
	for i := 8; i < len(got); i++ {
		if got[i] != -123.5 {
			t.Fatalf("suffix i=%d mutated got=%v", i, got[i])
		}
	}
}
