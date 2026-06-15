package diffusiongemma

import (
	"encoding/binary"
	"fmt"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func syntheticBenchQ8Matrix(b *testing.B, rows int) *gguf.ExpertMatrices {
	b.Helper()
	m := &gguf.ExpertMatrices{Name: "bench.q8", QType: gguf.QuantQ8_0, InDim: 704, OutDim: rows, Experts: 1}
	rowBytes, err := m.RowBytes()
	if err != nil {
		b.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*m.OutDim)
	for r := 0; r < m.OutDim; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for blk := 0; blk < m.InDim/32; blk++ {
			base := blk * 34
			binary.LittleEndian.PutUint16(row[base:base+2], half.F32ToF16(0.07+float32((r+blk)%7)*0.003))
			for i := 0; i < 32; i++ {
				row[base+2+i] = byte(int8((r*3+blk+i)%15 - 7))
			}
		}
	}
	return m
}

func syntheticBenchQ4KMatrix(b *testing.B, rows int) *gguf.ExpertMatrices {
	b.Helper()
	m := &gguf.ExpertMatrices{Name: "bench.q4k", QType: gguf.QuantQ4_K, InDim: 2816, OutDim: rows, Experts: 1}
	rowBytes, err := m.RowBytes()
	if err != nil {
		b.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*m.OutDim)
	for r := 0; r < m.OutDim; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for blkIdx := 0; blkIdx < m.InDim/256; blkIdx++ {
			blk := row[blkIdx*144 : (blkIdx+1)*144]
			binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.025+float32((r+blkIdx)%5)*0.002))
			binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.004+float32((r+blkIdx)%3)*0.001))
			for i := 0; i < 12; i++ {
				blk[4+i] = byte(1 + (i+r+blkIdx)%17)
			}
			for i := 0; i < 128; i++ {
				blk[16+i] = byte((i*7 + r*11 + blkIdx*13) & 0xff)
			}
		}
	}
	return m
}

func BenchmarkGGUFQ8_0RowDotDirectVsDequant(b *testing.B) {
	m := syntheticBenchQ8Matrix(b, 2816)
	x := make([]float32, m.InDim)
	row := make([]float32, m.InDim)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.013
	}
	b.Run("direct", func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			v, err := ggufQ8_0ExpertRowDot(m, 0, i%m.OutDim, x)
			if err != nil {
				b.Fatal(err)
			}
			sink += v
		}
		_ = sink
	})
	b.Run("dequant_sdot", func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			r := i % m.OutDim
			if err := m.DequantExpertRowTo(row, 0, r); err != nil {
				b.Fatal(err)
			}
			sink += simd.Sdot(row, x)
		}
		_ = sink
	})
}

func BenchmarkGGUFQ8_0RowDotBatchDirectVsDequant(b *testing.B) {
	m := syntheticBenchQ8Matrix(b, 2816)
	row := make([]float32, m.InDim)
	for _, nPos := range []int{1, 2, 3, 4, 8, 16} {
		x := make([]float32, nPos*m.InDim)
		directOut := make([]float32, nPos*m.OutDim)
		dequantOut := make([]float32, nPos*m.OutDim)
		for i := range x {
			x[i] = float32((i%17)-8) * 0.013
		}
		b.Run(fmt.Sprintf("direct_npos_%d", nPos), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				r := i % m.OutDim
				if err := ggufQ8_0ExpertRowDotBatchTo(m, 0, r, x, nPos, directOut[r:], m.OutDim, 1); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("dequant_sdot_npos_%d", nPos), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				r := i % m.OutDim
				if err := m.DequantExpertRowTo(row, 0, r); err != nil {
					b.Fatal(err)
				}
				for pos := 0; pos < nPos; pos++ {
					dequantOut[pos*m.OutDim+r] = simd.Sdot(row, x[pos*m.InDim:(pos+1)*m.InDim])
				}
			}
		})
	}
}

func BenchmarkGGUFQ4KRowDotBatchDirectVsDequant(b *testing.B) {
	m := syntheticBenchQ4KMatrix(b, 1408)
	row := make([]float32, m.InDim)
	for _, nPos := range []int{1, 2, 3, 4, 8} {
		x := make([]float32, nPos*m.InDim)
		directOut := make([]float32, nPos*m.OutDim)
		dequantOut := make([]float32, nPos*m.OutDim)
		for i := range x {
			x[i] = float32((i%19)-9) * 0.011
		}
		b.Run(fmt.Sprintf("direct_npos_%d", nPos), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				r := i % m.OutDim
				if err := ggufQ4KExpertRowDotBatchTo(m, 0, r, x, nPos, directOut[r:], m.OutDim); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("dequant_sdot_npos_%d", nPos), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				r := i % m.OutDim
				if err := m.DequantExpertRowTo(row, 0, r); err != nil {
					b.Fatal(err)
				}
				for pos := 0; pos < nPos; pos++ {
					dequantOut[pos*m.OutDim+r] = simd.Sdot(row, x[pos*m.InDim:(pos+1)*m.InDim])
				}
			}
		})
	}
}

func BenchmarkGGUFQ4KRowDotDirectVsDequant(b *testing.B) {
	m := syntheticBenchQ4KMatrix(b, 1408)
	x := make([]float32, m.InDim)
	row := make([]float32, m.InDim)
	for i := range x {
		x[i] = float32((i%19)-9) * 0.011
	}
	b.Run("direct", func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			v, err := ggufQ4KExpertRowDot(m, 0, i%m.OutDim, x)
			if err != nil {
				b.Fatal(err)
			}
			sink += v
		}
		_ = sink
	})
	b.Run("dequant_sdot", func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			r := i % m.OutDim
			if err := m.DequantExpertRowTo(row, 0, r); err != nil {
				b.Fatal(err)
			}
			sink += simd.Sdot(row, x)
		}
		_ = sink
	})
}
