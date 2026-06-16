package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestDequantRowQ8_0ToMatchesScaleTimesInt8(t *testing.T) {
	raw := make([]byte, 34*2)
	binary.LittleEndian.PutUint16(raw[0:2], half.F32ToF16(0.25))
	binary.LittleEndian.PutUint16(raw[34:36], half.F32ToF16(0.5))
	for i := 0; i < qk8_0; i++ {
		raw[2+i] = byte(int8(i - 16))
		raw[34+2+i] = byte(int8(16 - i))
	}
	dst := make([]float32, 64)
	if err := dequantRowQ8_0To(dst, raw, 64); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < qk8_0; i++ {
		if want := float32(i-16) * 0.25; dst[i] != want {
			t.Fatalf("dst[%d]=%g want %g", i, dst[i], want)
		}
		if want := float32(16-i) * 0.5; dst[qk8_0+i] != want {
			t.Fatalf("dst[%d]=%g want %g", qk8_0+i, dst[qk8_0+i], want)
		}
	}
}

func TestQuantizeQ8_0UsesRoundAwayFromZeroWithUnroundedScale(t *testing.T) {
	// amax=127 gives d=1 exactly, so these probe C roundf(x) semantics directly:
	// halves round away from zero, not to even.
	x := make([]float32, qk8_0)
	x[0], x[1], x[2], x[3], x[4] = 127, 0.5, -0.5, 1.5, -1.5
	blocks, err := QuantizeQ8_0(x)
	if err != nil {
		t.Fatal(err)
	}
	want := []int8{127, 1, -1, 2, -2}
	for i, w := range want {
		if got := blocks[0].qs[i]; got != w {
			t.Fatalf("qs[%d]=%d want %d", i, got, w)
		}
	}
	if blocks[0].d != 1 {
		t.Fatalf("d=%g want 1", blocks[0].d)
	}
}

func TestDotQ4_0Q8_0MatchesScalarReference(t *testing.T) {
	n := qk8_0 * 3
	nb := n / qk8_0
	raw := make([]byte, nb*18)
	y := make([]q8_0Block, nb)
	for bi := 0; bi < nb; bi++ {
		binary.LittleEndian.PutUint16(raw[bi*18:], half.F32ToF16(float32(0.03125+float32(bi)*0.0078125)))
		for j := 0; j < 16; j++ {
			raw[bi*18+2+j] = byte((bi*17 + j*29 + 3) & 0xff)
		}
		y[bi].d = float32(0.015625 + float32(bi)*0.00390625)
		for j := 0; j < qk8_0; j++ {
			y[bi].qs[j] = int8((bi*11+j*7)%127 - 63)
		}
	}
	got, err := DotQ4_0Q8_0(raw, y, n)
	if err != nil {
		t.Fatal(err)
	}
	want := dotQ4_0Q8_0ScalarReference(raw, y, n)
	if got != want {
		t.Fatalf("dot=%g want scalar %g diff=%g", got, want, got-want)
	}
}

func dotQ4_0Q8_0ScalarReference(raw []byte, y []q8_0Block, n int) float32 {
	nb := n / qk8_0
	var sum float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*18 : (bi+1)*18]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2])) * y[bi].d
		qs := blk[2:18]
		blockSum := 0
		for j := 0; j < qk8_0/2; j++ {
			blockSum += (int(qs[j]&0x0f) - 8) * int(y[bi].qs[j])
			blockSum += (int(qs[j]>>4) - 8) * int(y[bi].qs[j+qk8_0/2])
		}
		sum += float32(blockSum) * d
	}
	return sum
}

func TestDequantRowQ6KToMatchesScalarReference(t *testing.T) {
	raw := make([]byte, 210)
	for i := 0; i < 128; i++ {
		raw[i] = byte((i*19 + 5) & 0xff)
	}
	for i := 0; i < 64; i++ {
		raw[128+i] = byte((i*31 + 7) & 0xff)
	}
	for i := 0; i < 16; i++ {
		raw[192+i] = byte(int8((i*9)%63 - 31))
	}
	binary.LittleEndian.PutUint16(raw[208:], half.F32ToF16(0.25))
	got := make([]float32, qkK)
	if err := dequantRowQ6KTo(got, raw, qkK); err != nil {
		t.Fatal(err)
	}
	want := dequantQ6KScalarReference(raw)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dst[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func dequantQ6KScalarReference(raw []byte) []float32 {
	out := make([]float32, qkK)
	ql, qh, sc := raw[:128], raw[128:192], raw[192:208]
	d := half.F16ToF32(binary.LittleEndian.Uint16(raw[208:210]))
	a := 0
	qlOff, qhOff, scOff := 0, 0, 0
	for y := 0; y < qkK; y += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8((ql[qlOff+l+0]&0x0f)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
			q2 := int8((ql[qlOff+l+32]&0x0f)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
			q3 := int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
			q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
			out[y+l+0] = d * float32(int8(sc[scOff+is+0])) * float32(q1)
			out[y+l+32] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
			out[y+l+64] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
			out[y+l+96] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
		}
		a += 128
		qlOff += 64
		qhOff += 32
		scOff += 8
		_ = a
	}
	return out
}

func TestDotQ6KQ8KMatchesScalarReference(t *testing.T) {
	n := qkK * 2
	nb := n / qkK
	raw := make([]byte, nb*210)
	y := make([]q8KBlock, nb)
	for bi := 0; bi < nb; bi++ {
		base := bi * 210
		for i := 0; i < 128; i++ {
			raw[base+i] = byte((bi*13 + i*19 + 5) & 0xff)
		}
		for i := 0; i < 64; i++ {
			raw[base+128+i] = byte((bi*23 + i*31 + 7) & 0xff)
		}
		for i := 0; i < 16; i++ {
			raw[base+192+i] = byte(int8((bi*5+i*9)%63 - 31))
		}
		binary.LittleEndian.PutUint16(raw[base+208:], half.F32ToF16(float32(0.0234375+float32(bi)*0.0048828125)))
		y[bi].d = float32(0.017578125 + float32(bi)*0.0029296875)
		for i := 0; i < qkK; i++ {
			y[bi].qs[i] = int8((bi*17+i*5)%127 - 63)
		}
		for i := 0; i < qkK/16; i++ {
			s := 0
			for j := 0; j < 16; j++ {
				s += int(y[bi].qs[i*16+j])
			}
			y[bi].bsums[i] = int16(s)
		}
	}
	got, err := DotQ6KQ8K(raw, y, n)
	if err != nil {
		t.Fatal(err)
	}
	want := dotQ6KQ8KScalarReference(raw, y, n)
	if math.Abs(float64(got-want)) > 1e-4 {
		t.Fatalf("dot=%g want scalar %g diff=%g", got, want, got-want)
	}
}

func dotQ6KQ8KScalarReference(raw []byte, y []q8KBlock, n int) float32 {
	nb := n / qkK
	var sum float32
	var aux [qkK]int
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*210 : (bi+1)*210]
		ql, qh, sc := blk[:128], blk[128:192], blk[192:208]
		a := 0
		qlOff, qhOff := 0, 0
		for j := 0; j < qkK; j += 128 {
			_ = j
			for l := 0; l < 32; l++ {
				aux[a+l+0] = int((ql[qlOff+l+0]&0x0f)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				aux[a+l+32] = int((ql[qlOff+l+32]&0x0f)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				aux[a+l+64] = int((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				aux[a+l+96] = int((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
			}
			a += 128
			qlOff += 64
			qhOff += 32
		}
		is, q8 := 0, 0
		blockSum := 0
		for j := 0; j < qkK/16; j++ {
			scale := int(int8(sc[is]))
			is++
			for l := 0; l < 16; l++ {
				blockSum += scale * int(y[bi].qs[q8+l]) * aux[q8+l]
			}
			q8 += 16
		}
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210])) * y[bi].d
		sum += d * float32(blockSum)
	}
	return sum
}
