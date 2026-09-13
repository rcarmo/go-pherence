package simd

import (
	"encoding/binary"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func BenchmarkSgemmNTQ8_0VsDequantizedPacked(b *testing.B) {
	const (
		m = 128
		n = 1024
		k = 1024
	)
	a := makeTestQ8GemmA(m, k)
	wq := makeTestQ8_0Raw(n, k)
	wf := dequantQ8_0RowsForBench(wq, n, k)
	packed, err := PackSgemmNTWeights(wf, n, k, k)
	if err != nil {
		b.Fatal(err)
	}
	c := make([]float32, m*n)

	b.Run("q8_0", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			clear(c)
			if !SgemmNTQ8_0To(c, a, wq, m, n, k) {
				b.Fatal("SgemmNTQ8_0To rejected valid inputs")
			}
		}
	})
	b.Run("dequant-prepacked", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			clear(c)
			if !SgemmNTPrepackedTo(c, a, wf, packed, m, n, k, 1, k, k, n) {
				b.Fatal("SgemmNTPrepackedTo rejected valid inputs")
			}
		}
	})
}

func dequantQ8_0RowsForBench(w []byte, n, k int) []float32 {
	blocks := k / q8_0BlockElems
	rowBytes := blocks * q8_0BlockBytes
	out := make([]float32, n*k)
	for j := 0; j < n; j++ {
		row := w[j*rowBytes : (j+1)*rowBytes]
		dst := out[j*k : (j+1)*k]
		for b := 0; b < blocks; b++ {
			blk := row[b*q8_0BlockBytes : (b+1)*q8_0BlockBytes]
			d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
			for i := 0; i < q8_0BlockElems; i++ {
				dst[b*q8_0BlockElems+i] = d * float32(int8(blk[2+i]))
			}
		}
	}
	return out
}
