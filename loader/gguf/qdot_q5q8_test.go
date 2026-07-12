package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestDotQ5_0Q8_0MatchesScalarReference(t *testing.T) {
	n := qk8_0 * 4
	raw := make([]byte, n/qk8_0*22)
	y := make([]q8_0Block, n/qk8_0)
	for bi := 0; bi < n/qk8_0; bi++ {
		blk := raw[bi*22 : (bi+1)*22]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(float32(0.03125+float32(bi)*0.0078125)))
		binary.LittleEndian.PutUint32(blk[2:6], uint32(0xa55a00f0^uint32(bi*0x13579b)))
		for i := 0; i < 16; i++ {
			blk[6+i] = byte((bi*17 + i*29 + 5) & 0xff)
		}
		y[bi].d = float32(0.015625 + float32(bi)*0.00390625)
		for j := 0; j < qk8_0; j++ {
			y[bi].qs[j] = int8((bi*19+j*11)%127 - 63)
		}
	}
	got, err := DotQ5_0Q8_0(raw, y, n)
	if err != nil {
		t.Fatal(err)
	}
	want := dotQ5_0Q8_0ScalarReference(raw, y, n)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("dot=%g want scalar %g diff=%g", got, want, got-want)
	}
}

func TestDotQ8_0Q8_0MatchesScalarReference(t *testing.T) {
	n := qk8_0 * 5
	raw := make([]byte, n/qk8_0*34)
	y := make([]q8_0Block, n/qk8_0)
	for bi := 0; bi < n/qk8_0; bi++ {
		blk := raw[bi*34 : (bi+1)*34]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(float32(0.0234375+float32(bi)*0.00390625)))
		for j := 0; j < qk8_0; j++ {
			blk[2+j] = byte(int8((bi*13+j*7)%127 - 63))
			y[bi].qs[j] = int8((bi*5+j*9)%127 - 63)
		}
		y[bi].d = float32(0.01171875 + float32(bi)*0.001953125)
	}
	got, err := DotQ8_0Q8_0(raw, y, n)
	if err != nil {
		t.Fatal(err)
	}
	want := dotQ8_0Q8_0ScalarReference(raw, y, n)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("dot=%g want scalar %g diff=%g", got, want, got-want)
	}
}

func dotQ5_0Q8_0ScalarReference(raw []byte, y []q8_0Block, n int) float32 {
	nb := n / qk8_0
	var sum float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*22 : (bi+1)*22]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2])) * y[bi].d
		qh := binary.LittleEndian.Uint32(blk[2:6])
		qs := blk[6:22]
		blockSum := 0
		for j := 0; j < qk8_0/2; j++ {
			q0 := int(qs[j] & 0x0F)
			q1 := int(qs[j] >> 4)
			if qh&(1<<uint(j)) != 0 {
				q0 |= 16
			}
			if qh&(1<<uint(j+16)) != 0 {
				q1 |= 16
			}
			blockSum += (q0 - 16) * int(y[bi].qs[j])
			blockSum += (q1 - 16) * int(y[bi].qs[j+16])
		}
		sum += float32(blockSum) * d
	}
	return sum
}

func dotQ8_0Q8_0ScalarReference(raw []byte, y []q8_0Block, n int) float32 {
	nb := n / qk8_0
	var sum float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*34 : (bi+1)*34]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2])) * y[bi].d
		blockSum := 0
		for j := 0; j < qk8_0; j++ {
			blockSum += int(int8(blk[2+j])) * int(y[bi].qs[j])
		}
		sum += float32(blockSum) * d
	}
	return sum
}
