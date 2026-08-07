package gguf

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

var q6CoeffBenchSink int32

func TestQ6KCoeffDotMatchesScalar(t *testing.T) {
	var q8 [256]int8
	var coeff [256]int16
	var want int32
	for i := range q8 {
		q8[i] = int8((i*17)%127 - 63)
		coeff[i] = int16((i*29)%3969 - 1984)
		want += int32(q8[i]) * int32(coeff[i])
	}
	if got := q6KCoeffDot(&q8, &coeff); got != want {
		t.Fatalf("q6KCoeffDot=%d want %d", got, want)
	}
}

func BenchmarkQ6KCoeffDot(b *testing.B) {
	var q8 [256]int8
	var coeff [256]int16
	for i := range q8 {
		q8[i] = int8((i*17)%127 - 63)
		coeff[i] = int16((i*29)%3969 - 1984)
	}
	b.ResetTimer()
	var sum int32
	for i := 0; i < b.N; i++ {
		sum += q6KCoeffDot(&q8, &coeff)
	}
	q6CoeffBenchSink += sum
}

func BenchmarkQ6KCoeffDotScalar(b *testing.B) {
	var q8 [256]int8
	var coeff [256]int16
	for i := range q8 {
		q8[i] = int8((i*17)%127 - 63)
		coeff[i] = int16((i*29)%3969 - 1984)
	}
	b.ResetTimer()
	var sum int32
	for i := 0; i < b.N; i++ {
		for lane := range q8 {
			sum += int32(q8[lane]) * int32(coeff[lane])
		}
	}
	q6CoeffBenchSink += sum
}

func TestQ6KCoeffDot8MatchesScalarPartitions(t *testing.T) {
	var q8 [256]int8
	var coeff [256]int16
	var want, got [8]int32
	for i := range q8 {
		q8[i] = int8((i*17)%127 - 63)
		coeff[i] = int16((i*29)%3969 - 1984)
		want[(i/2)%8] += int32(q8[i]) * int32(coeff[i])
	}
	q6KCoeffDot8(&q8, &coeff, &got)
	if got != want {
		t.Fatalf("q6KCoeffDot8=%v want %v", got, want)
	}
}

func TestQ6KExpandCoeffMatchesReference(t *testing.T) {
	raw, _ := syntheticQ6KQ8KDotInputs(256)
	block := (*[210]byte)(raw)
	var got, want [256]int16
	q6KExpandCoeff(block, &got)
	ql, qh, scales := block[:128], block[128:192], block[192:208]
	for halfBlock := 0; halfBlock < 2; halfBlock++ {
		qlOff, qhOff, base := halfBlock*64, halfBlock*32, halfBlock*128
		for l := 0; l < 32; l++ {
			is := l / 16
			want[base+l] = int16(int8(scales[halfBlock*8+is])) * (int16((ql[qlOff+l]&15)|(((qh[qhOff+l]>>0)&3)<<4)) - 32)
			want[base+l+32] = int16(int8(scales[halfBlock*8+is+2])) * (int16((ql[qlOff+l+32]&15)|(((qh[qhOff+l]>>2)&3)<<4)) - 32)
			want[base+l+64] = int16(int8(scales[halfBlock*8+is+4])) * (int16((ql[qlOff+l]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32)
			want[base+l+96] = int16(int8(scales[halfBlock*8+is+6])) * (int16((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32)
		}
	}
	if got != want {
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("coefficient %d=%d want %d", i, got[i], want[i])
			}
		}
	}
}

func TestQ6KBlockDotMatchesExpandedDot(t *testing.T) {
	raw, y := syntheticQ6KQ8KDotInputs(256)
	block := (*[210]byte)(raw)
	var coeff [256]int16
	q6KExpandCoeff(block, &coeff)
	want := q6KCoeffDot(&y[0].qs, &coeff)
	if got := q6KBlockDot(block, &y[0].qs); got != want {
		t.Fatalf("q6KBlockDot=%d want %d", got, want)
	}
	if got, ok := q6KBlockDotVNNI(block, &y[0].qs); ok && got != want {
		t.Fatalf("q6KBlockDotVNNI=%d want %d", got, want)
	}
}

func TestQ6KBlockDotVNNIRandomExact(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6b10c))
	for iteration := 0; iteration < 1000; iteration++ {
		var block [210]byte
		var q8 [256]int8
		for i := range block {
			block[i] = byte(rng.Intn(256))
		}
		for i := range q8 {
			q8[i] = int8(rng.Intn(256) - 128)
		}
		got, ok := q6KBlockDotVNNI(&block, &q8)
		if !ok {
			t.Skip("AVX-VNNI unavailable")
		}
		want := q6KBlockDotAVX2(&block, &q8)
		if got != want {
			t.Fatalf("iteration=%d vnni=%d avx2=%d", iteration, got, want)
		}
	}
}

func BenchmarkQ6KBlockDot(b *testing.B) {
	raw, y := syntheticQ6KQ8KDotInputs(256)
	block := (*[210]byte)(raw)
	b.Run("avx2", func(b *testing.B) {
		var result int32
		for i := 0; i < b.N; i++ {
			result += q6KBlockDotAVX2(block, &y[0].qs)
		}
		q6CoeffBenchSink += result
	})
	b.Run("vnni", func(b *testing.B) {
		var result int32
		if _, ok := q6KBlockDotVNNI(block, &y[0].qs); !ok {
			b.Skip("AVX-VNNI unavailable")
		}
		for i := 0; i < b.N; i++ {
			result += q6KBlockDot(block, &y[0].qs)
		}
		q6CoeffBenchSink += result
	})
}

func TestDotQ6KQ8KGemvVNNIRandomExact(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6f8e))
	for iteration := 0; iteration < 100; iteration++ {
		blocks := 1 + rng.Intn(10)
		raw := make([]byte, blocks*210)
		y := make([]q8KBlock, blocks)
		for bi := 0; bi < blocks; bi++ {
			block := raw[bi*210 : (bi+1)*210]
			for i := 0; i < 208; i++ {
				block[i] = byte(rng.Intn(256))
			}
			binary.LittleEndian.PutUint16(block[208:], half.F32ToF16((rng.Float32()*2-1)*0.25))
			y[bi].d = (rng.Float32()*2 - 1) * 0.25
			for i := range y[bi].qs {
				y[bi].qs[i] = int8(rng.Intn(256) - 128)
			}
			for group := range y[bi].bsums {
				for j := 0; j < 16; j++ {
					y[bi].bsums[group] += int16(y[bi].qs[group*16+j])
				}
			}
		}
		got, ok := dotQ6KQ8KGemvVNNI(raw, y, blocks)
		if !ok {
			t.Skip("AVX-VNNI unavailable")
		}
		if want := dotQ6KQ8KGemvFastLoop(raw, y, blocks); got != want {
			t.Fatalf("iteration=%d blocks=%d vnni=%g loop=%g", iteration, blocks, got, want)
		}
	}
}

func BenchmarkQ6KExpandCoeff(b *testing.B) {
	raw, _ := syntheticQ6KQ8KDotInputs(256)
	block := (*[210]byte)(raw)
	var coeff [256]int16
	for i := 0; i < b.N; i++ {
		q6KExpandCoeff(block, &coeff)
	}
	q6CoeffBenchSink += int32(coeff[0])
}

func BenchmarkDotQ6KQ8KGemvFast(b *testing.B) {
	raw, y := syntheticQ6KQ8KDotInputs(2560)
	b.Run("loop", func(b *testing.B) {
		var result float32
		for i := 0; i < b.N; i++ {
			result += dotQ6KQ8KGemvFastLoop(raw, y, 10)
		}
		q6CoeffBenchSink += int32(result)
	})
	b.Run("fused", func(b *testing.B) {
		var result float32
		for i := 0; i < b.N; i++ {
			result += dotQ6KQ8KGemvFast(raw, y, 10)
		}
		q6CoeffBenchSink += int32(result)
	})
	b.Run("expanded", func(b *testing.B) {
		var result int32
		for i := 0; i < b.N; i++ {
			for bi := 0; bi < 10; bi++ {
				var coeff [256]int16
				q6KExpandCoeff((*[210]byte)(raw[bi*210:]), &coeff)
				result += q6KCoeffDot(&y[bi].qs, &coeff)
			}
		}
		q6CoeffBenchSink += result
	})
}

func TestDotQ6KQ8KGemvFastBoundedDifference(t *testing.T) {
	raw, y := syntheticQ6KQ8KDotInputs(2560)
	got := dotQ6KQ8KGemvFast(raw, y, 10)
	want, err := DotQ6KQ8K(raw, y, 2560)
	if err != nil {
		t.Fatal(err)
	}
	const maxDifference = float32(0.00012207031)
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	if diff > maxDifference {
		t.Fatalf("fast dot=%g exact=%g difference=%g > %g", got, want, diff, maxDifference)
	}
}
