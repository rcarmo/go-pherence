package gguf

import (
	"encoding/binary"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func BenchmarkDotQ4_0Q8_0(b *testing.B) {
	n := qk8_0 * 80 // Gemma4 E4B hidden/attention widths are multiples of 2560.
	nb := n / qk8_0
	raw := make([]byte, nb*18)
	y := make([]q8_0Block, nb)
	for bi := 0; bi < nb; bi++ {
		binary.LittleEndian.PutUint16(raw[bi*18:], half.F32ToF16(float32(0.03125+float32(bi%7)*0.00390625)))
		for j := 0; j < 16; j++ {
			raw[bi*18+2+j] = byte((bi*17 + j*29 + 3) & 0xff)
		}
		y[bi].d = float32(0.015625 + float32(bi%5)*0.001953125)
		for j := 0; j < qk8_0; j++ {
			y[bi].qs[j] = int8((bi*11+j*7)%127 - 63)
		}
	}
	b.SetBytes(int64(n * 2))
	b.ReportAllocs()
	var sink float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := DotQ4_0Q8_0(raw, y, n)
		if err != nil {
			b.Fatal(err)
		}
		sink += v
	}
	_ = sink
}

func BenchmarkDotQ4_0Q8_0Scalar(b *testing.B) {
	n := qk8_0 * 80
	nb := n / qk8_0
	raw := make([]byte, nb*18)
	y := make([]q8_0Block, nb)
	for bi := 0; bi < nb; bi++ {
		binary.LittleEndian.PutUint16(raw[bi*18:], half.F32ToF16(float32(0.03125+float32(bi%7)*0.00390625)))
		for j := 0; j < 16; j++ {
			raw[bi*18+2+j] = byte((bi*17 + j*29 + 3) & 0xff)
		}
		y[bi].d = float32(0.015625 + float32(bi%5)*0.001953125)
		for j := 0; j < qk8_0; j++ {
			y[bi].qs[j] = int8((bi*11+j*7)%127 - 63)
		}
	}
	b.SetBytes(int64(n * 2))
	b.ReportAllocs()
	var sink float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink += dotQ4_0Q8_0Scalar(raw, y, nb)
	}
	_ = sink
}

func BenchmarkDotQ6KQ8K(b *testing.B) {
	n := qkK * 10 // 2560-wide Gemma4 tied LM-head/embedding row-dot size.
	nb := n / qkK
	raw := make([]byte, nb*210)
	y := make([]q8KBlock, nb)
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*210 : (bi+1)*210]
		for i := 0; i < 128; i++ {
			blk[i] = byte((bi*13 + i*19 + 5) & 0xff)
		}
		for i := 0; i < 64; i++ {
			blk[128+i] = byte((bi*7 + i*31 + 7) & 0xff)
		}
		for i := 0; i < 16; i++ {
			blk[192+i] = byte(int8((bi*5+i*9)%63 - 31))
		}
		binary.LittleEndian.PutUint16(blk[208:], half.F32ToF16(0.03125+float32(bi%7)*0.00390625))
		for j := 0; j < qkK; j++ {
			y[bi].qs[j] = int8((bi*11+j*7)%127 - 63)
		}
		for j := 0; j < qkK/16; j++ {
			var sum int16
			for k := 0; k < 16; k++ {
				sum += int16(y[bi].qs[j*16+k])
			}
			y[bi].bsums[j] = sum
		}
		y[bi].d = float32(0.015625 + float32(bi%5)*0.001953125)
	}
	b.SetBytes(int64(n * 2))
	b.ReportAllocs()
	var sink float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := DotQ6KQ8K(raw, y, n)
		if err != nil {
			b.Fatal(err)
		}
		sink += v
	}
	_ = sink
}
