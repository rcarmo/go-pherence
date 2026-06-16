package diffusiongemma

import (
	"encoding/binary"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func syntheticBenchQ6KLMHeadMatrix(b testing.TB, rows int) *gguf.QuantMatrix {
	b.Helper()
	m := &gguf.QuantMatrix{Name: "bench.q6k", QType: gguf.QuantQ6_K, InDim: 2816, OutDim: rows}
	rowBytes, err := m.RowBytes()
	if err != nil {
		b.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*rows)
	for r := 0; r < rows; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for off := 0; off < rowBytes; off += 210 {
			blk := row[off : off+210]
			for i := 0; i < 128; i++ {
				blk[i] = byte((i*13 + r*17 + off) & 0xff)
			}
			for i := 0; i < 64; i++ {
				blk[128+i] = byte((i*7 + r*19 + off) & 0xff)
			}
			for i := 0; i < 16; i++ {
				blk[192+i] = byte(int8((i%9)-4) + int8(r%5))
			}
			binary.LittleEndian.PutUint16(blk[208:210], half.F32ToF16(0.018+float32(r%7)*0.003))
		}
	}
	return m
}

func BenchmarkGGUFQ6KLMHeadRowDotPrequantRows(b *testing.B) {
	m := syntheticBenchQ6KLMHeadMatrix(b, 1024)
	positions := 256
	hidden := make([]float32, positions*m.InDim)
	for i := range hidden {
		hidden[i] = float32((i%23)-11) * 0.017
	}
	q8, err := prequantizeQ8KRowsForLMHead(hidden, positions, m.InDim)
	if err != nil {
		b.Fatal(err)
	}
	out := make([]float32, positions)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ggufQ6KMatrixRowDotPrequantRows(m, i%m.OutDim, q8, out); err != nil {
			b.Fatal(err)
		}
	}
}
