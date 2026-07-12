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

func TestDotQ4_0Q8_0MatchesAVX2Reference(t *testing.T) {
	n := qk8_0 * 7
	raw, y := syntheticQ4_0Q8_0DotInputs(n)
	got, err := DotQ4_0Q8_0(raw, y, n)
	if err != nil {
		t.Fatal(err)
	}
	want := dotQ4_0Q8_0AVX2Reference(raw, y, n)
	if got != want {
		t.Fatalf("dot=%g want avx2 %g diff=%g", got, want, got-want)
	}
}

func TestDotQ4_0Q8_0MatchesScalarReference(t *testing.T) {
	n := qk8_0 * 3
	raw, y := syntheticQ4_0Q8_0DotInputs(n)
	got, err := DotQ4_0Q8_0(raw, y, n)
	if err != nil {
		t.Fatal(err)
	}
	want := dotQ4_0Q8_0ScalarReference(raw, y, n)
	if got != want {
		t.Fatalf("dot=%g want scalar %g diff=%g", got, want, got-want)
	}
}

func syntheticQ4_0Q8_0DotInputs(n int) ([]byte, []q8_0Block) {
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
	return raw, y
}

func dotQ4_0Q8_0AVX2Reference(raw []byte, y []q8_0Block, n int) float32 {
	nb := n / qk8_0
	var acc [8]float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*18 : (bi+1)*18]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2])) * y[bi].d
		qs := blk[2:18]
		for lane := 0; lane < 4; lane++ {
			j := lane * 4
			s := (int(qs[j+0]&0x0F)-8)*int(y[bi].qs[j+0]) +
				(int(qs[j+1]&0x0F)-8)*int(y[bi].qs[j+1]) +
				(int(qs[j+2]&0x0F)-8)*int(y[bi].qs[j+2]) +
				(int(qs[j+3]&0x0F)-8)*int(y[bi].qs[j+3])
			acc[lane] = float32(math.FMA(float64(d), float64(float32(s)), float64(acc[lane])))
		}
		for lane := 0; lane < 4; lane++ {
			j := lane * 4
			s := (int(qs[j+0]>>4)-8)*int(y[bi].qs[j+16]) +
				(int(qs[j+1]>>4)-8)*int(y[bi].qs[j+17]) +
				(int(qs[j+2]>>4)-8)*int(y[bi].qs[j+18]) +
				(int(qs[j+3]>>4)-8)*int(y[bi].qs[j+19])
			acc[lane+4] = float32(math.FMA(float64(d), float64(float32(s)), float64(acc[lane+4])))
		}
	}
	r0 := acc[0] + acc[4]
	r1 := acc[1] + acc[5]
	r2 := acc[2] + acc[6]
	r3 := acc[3] + acc[7]
	r0 = r0 + r2
	r1 = r1 + r3
	return r0 + r1
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

func TestQuantizeQ8KComputesScaleQuantsAndBlockSums(t *testing.T) {
	x := make([]float32, qkK)
	x[0] = -2
	x[1] = 1
	x[2] = 0.5
	x[15] = -0.25
	x[16] = 2
	x[17] = -1
	blocks, err := QuantizeQ8K(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks=%d want 1", len(blocks))
	}
	b := blocks[0]
	wantD := float32(2.0 / 127.0)
	if b.d != wantD {
		t.Fatalf("d=%g want %g", b.d, wantD)
	}
	wantQS := map[int]int8{0: -127, 1: 64, 2: 32, 15: -16, 16: 127, 17: -64}
	for i, want := range wantQS {
		if got := b.qs[i]; got != want {
			t.Fatalf("qs[%d]=%d want %d", i, got, want)
		}
	}
	var sum0, sum1 int16
	for i := 0; i < 16; i++ {
		sum0 += int16(b.qs[i])
		sum1 += int16(b.qs[16+i])
	}
	if b.bsums[0] != sum0 || b.bsums[1] != sum1 {
		t.Fatalf("bsums[0:2]=%v,%v want %v,%v", b.bsums[0], b.bsums[1], sum0, sum1)
	}
}

func TestDotQ4KQ8KMatchesActualFixtureVecdotOracle(t *testing.T) {
	fixture, q4Raw, acts := loadQ4K8x8ActualBackendFixture(t)
	rowBytes, err := TensorRawBytes(QuantQ4_K, fixture.K)
	if err != nil {
		t.Fatal(err)
	}
	maxGap := float32(0)
	distinct := 0
	for pos := 0; pos < q4KOracleM; pos++ {
		q8, err := QuantizeQ8K(acts[pos*fixture.K : (pos+1)*fixture.K])
		if err != nil {
			t.Fatal(err)
		}
		for row := 0; row < q4KOracleN; row++ {
			raw := q4Raw[row*rowBytes : (row+1)*rowBytes]
			got, err := DotQ4KQ8K(raw, q8, fixture.K)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(got-fixture.Vecdot[row][pos])) > 1e-4 {
				t.Fatalf("pos=%d row=%d got=%g want vecdot %g diff=%g", pos, row, got, fixture.Vecdot[row][pos], got-fixture.Vecdot[row][pos])
			}
			gap := float32(math.Abs(float64(fixture.Output[row][pos] - got)))
			if gap > maxGap {
				maxGap = gap
			}
			if gap > 5e-7 {
				distinct++
			}
		}
	}
	if distinct == 0 {
		t.Fatalf("Q4_K vecdot oracle collapsed to tuned backend output; max gap=%g", maxGap)
	}
}

func TestDotQ4KQ8KDiffersFromDequantF32OnSyntheticBoundary(t *testing.T) {
	m := &QuantMatrix{Name: "synthetic.q4k", QType: QuantQ4_K, InDim: qkK, OutDim: 1}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes)
	blk := m.Raw[:rowBytes]
	binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.03125))
	binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.0078125))
	for i := 0; i < 12; i++ {
		blk[4+i] = byte(1 + (i*5)%17)
	}
	for i := 0; i < 128; i++ {
		blk[16+i] = byte((i*29 + 7) & 0xff)
	}
	x := make([]float32, qkK)
	for i := range x {
		x[i] = float32((i%31)-15)*0.013 + float32((i/31)%3)*0.001
	}
	q8, err := QuantizeQ8K(x)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DotQ4KQ8K(m.Raw, q8, qkK)
	if err != nil {
		t.Fatal(err)
	}
	deq := make([]float32, qkK)
	if err := m.DequantRowTo(deq, 0); err != nil {
		t.Fatal(err)
	}
	wantF32 := dotF32(deq, x)
	if math.Abs(float64(got-wantF32)) <= 1e-6 {
		t.Fatalf("quantized dot unexpectedly matched dequant-F32 dot: got=%g dequant=%g", got, wantF32)
	}
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

func TestDotQ6KQ8KMatchesAVX2Reference(t *testing.T) {
	n := qkK * 2
	raw, y := syntheticQ6KQ8KDotInputs(n)
	got, err := DotQ6KQ8K(raw, y, n)
	if err != nil {
		t.Fatal(err)
	}
	want := dotQ6KQ8KAVX2Reference(raw, y, n)
	if got != want {
		t.Fatalf("dot=%g want avx2 %g diff=%g", got, want, got-want)
	}
}

func TestDotQ6KQ8KMatchesScalarReference(t *testing.T) {
	n := qkK * 2
	raw, y := syntheticQ6KQ8KDotInputs(n)
	got, err := DotQ6KQ8K(raw, y, n)
	if err != nil {
		t.Fatal(err)
	}
	want := dotQ6KQ8KScalarReference(raw, y, n)
	if math.Abs(float64(got-want)) > 1e-4 {
		t.Fatalf("dot=%g want scalar %g diff=%g", got, want, got-want)
	}
}

func syntheticQ6KQ8KDotInputs(n int) ([]byte, []q8KBlock) {
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
	return raw, y
}

func dotQ6KQ8KAVX2Reference(raw []byte, y []q8KBlock, n int) float32 {
	nb := n / qkK
	var sums [8]float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*210 : (bi+1)*210]
		ql, qh, sc := blk[:128], blk[128:192], blk[192:208]
		var sumi [8]int32
		var q8sclsub [8]int32
		for l := 0; l < 8; l++ {
			q8sclsub[l] = int32((int(y[bi].bsums[2*l])*int(int8(sc[2*l])) + int(y[bi].bsums[2*l+1])*int(int8(sc[2*l+1]))) << 5)
		}
		qlOff, qhOff, q8Base, scalePair := 0, 0, 0, 0
		for j := 0; j < qkK/128; j++ {
			for vec := 0; vec < 4; vec++ {
				for lane := 0; lane < 8; lane++ {
					scaleOff := (scalePair + vec) * 2
					scale := int(int8(sc[scaleOff]))
					if lane >= 4 {
						scale = int(int8(sc[scaleOff+1]))
					}
					seg := 0
					for p := 0; p < 4; p++ {
						idx := vec*32 + lane*4 + p
						var q int
						switch vec {
						case 0:
							q = int((ql[qlOff+idx] & 0x0F) | (((qh[qhOff+idx] >> 0) & 3) << 4))
						case 1:
							q = int((ql[qlOff+idx] & 0x0F) | (((qh[qhOff+idx-32] >> 2) & 3) << 4))
						case 2:
							q = int((ql[qlOff+idx-64] >> 4) | (((qh[qhOff+idx-64] >> 4) & 3) << 4))
						case 3:
							q = int((ql[qlOff+idx-64] >> 4) | (((qh[qhOff+idx-96] >> 6) & 3) << 4))
						}
						seg += q * int(y[bi].qs[q8Base+idx])
					}
					sumi[lane] += int32(scale * seg)
				}
			}
			qlOff += 64
			qhOff += 32
			q8Base += 128
			scalePair += 4
		}
		for l := 0; l < 8; l++ {
			sumi[l] -= q8sclsub[l]
		}
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210])) * y[bi].d
		for l := 0; l < 8; l++ {
			sums[l] = float32(math.FMA(float64(d), float64(float32(sumi[l])), float64(sums[l])))
		}
	}
	r0 := sums[0] + sums[4]
	r1 := sums[1] + sums[5]
	r2 := sums[2] + sums[6]
	r3 := sums[3] + sums[7]
	r0 = r0 + r2
	r1 = r1 + r3
	return r0 + r1
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
