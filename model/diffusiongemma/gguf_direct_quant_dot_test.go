package diffusiongemma

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func scalarDotForTest(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func TestGGUFQ8_0ExpertRowDotMatchesScalarDequantOracle(t *testing.T) {
	m := &gguf.ExpertMatrices{Name: "synthetic.q8", QType: gguf.QuantQ8_0, InDim: 32, OutDim: 3, Experts: 2}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*m.OutDim*m.Experts)
	for r := 0; r < m.OutDim*m.Experts; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		binary.LittleEndian.PutUint16(row[:2], half.F32ToF16(0.07+float32(r)*0.013))
		for i := 0; i < m.InDim; i++ {
			row[2+i] = byte(int8((r*3+i)%15 - 7))
		}
	}
	x := make([]float32, m.InDim)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.017
	}
	row := make([]float32, m.InDim)
	for expert := 0; expert < m.Experts; expert++ {
		for r := 0; r < m.OutDim; r++ {
			if err := m.DequantExpertRowTo(row, expert, r); err != nil {
				t.Fatal(err)
			}
			want := scalarDotForTest(row, x)
			got, err := ggufQ8_0ExpertRowDot(m, expert, r, x)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("expert=%d row=%d got=%g want=%g", expert, r, got, want)
			}
			batchIn := append(append([]float32(nil), x...), x...)
			batchOut := make([]float32, 2*m.OutDim)
			if err := ggufQ8_0ExpertRowDotBatchTo(m, expert, r, batchIn, 2, batchOut[r:], m.OutDim, 1); err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(batchOut[r]-want)) > 1e-5 || math.Abs(float64(batchOut[m.OutDim+r]-want)) > 1e-5 {
				t.Fatalf("batch expert=%d row=%d got=%g/%g want=%g", expert, r, batchOut[r], batchOut[m.OutDim+r], want)
			}
		}
	}
}

func TestGGUFQ5_0ExpertRowDotMatchesScalarDequantOracle(t *testing.T) {
	m := &gguf.ExpertMatrices{Name: "synthetic.q5", QType: gguf.QuantQ5_0, InDim: 64, OutDim: 3, Experts: 2}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, m.Experts*m.OutDim*rowBytes)
	for r := 0; r < m.Experts*m.OutDim; r++ {
		for b := 0; b < m.InDim/32; b++ {
			blk := m.Raw[r*rowBytes+b*22 : r*rowBytes+(b+1)*22]
			binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(0.05+float32(r+b)*0.01))
			binary.LittleEndian.PutUint32(blk[2:6], uint32(0xa5a50000|r<<3|b))
			for i := 0; i < 16; i++ {
				blk[6+i] = byte((r*13 + b*17 + i*7) & 0xff)
			}
		}
	}
	x := make([]float32, m.InDim)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.03
	}
	row := make([]float32, m.InDim)
	for expert := 0; expert < m.Experts; expert++ {
		for r := 0; r < m.OutDim; r++ {
			if err := m.DequantExpertRowTo(row, expert, r); err != nil {
				t.Fatal(err)
			}
			want := scalarDotForTest(row, x)
			got, err := ggufQ5_0ExpertRowDot(m, expert, r, x)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Fatalf("expert=%d row=%d got=%g want=%g", expert, r, got, want)
			}
			batchIn := append(append([]float32(nil), x...), x...)
			batchOut := make([]float32, 2*m.OutDim)
			if err := ggufQ5_0ExpertRowDotBatchTo(m, expert, r, batchIn, 2, batchOut[r:], m.OutDim, 1); err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(batchOut[r]-want)) > 1e-4 || math.Abs(float64(batchOut[m.OutDim+r]-want)) > 1e-4 {
				t.Fatalf("batch expert=%d row=%d got=%g/%g want=%g", expert, r, batchOut[r], batchOut[m.OutDim+r], want)
			}
		}
	}
}

func TestGGUFQ4KExpertRowDotMatchesScalarDequantOracle(t *testing.T) {
	m := &gguf.ExpertMatrices{Name: "synthetic.q4k", QType: gguf.QuantQ4_K, InDim: 256, OutDim: 4, Experts: 2}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*m.OutDim*m.Experts)
	for r := 0; r < m.OutDim*m.Experts; r++ {
		blk := m.Raw[r*rowBytes : (r+1)*rowBytes]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.025+float32(r)*0.002))
		binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.004+float32(r%3)*0.001))
		for i := 0; i < 12; i++ {
			blk[4+i] = byte(1 + (i+r)%17)
		}
		for i := 0; i < 128; i++ {
			blk[16+i] = byte((i*7 + r*11) & 0xff)
		}
	}
	x := make([]float32, m.InDim)
	for i := range x {
		x[i] = float32((i%19)-9) * 0.011
	}
	row := make([]float32, m.InDim)
	for expert := 0; expert < m.Experts; expert++ {
		for r := 0; r < m.OutDim; r++ {
			if err := m.DequantExpertRowTo(row, expert, r); err != nil {
				t.Fatal(err)
			}
			want := scalarDotForTest(row, x)
			got, err := ggufQ4KExpertRowDot(m, expert, r, x)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Fatalf("expert=%d row=%d got=%g want=%g", expert, r, got, want)
			}
			batchIn := append(append([]float32(nil), x...), x...)
			batchOut := make([]float32, 2*m.OutDim)
			if err := ggufQ4KExpertRowDotBatchTo(m, expert, r, batchIn, 2, batchOut[r:], m.OutDim); err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(batchOut[r]-want)) > 1e-4 || math.Abs(float64(batchOut[m.OutDim+r]-want)) > 1e-4 {
				t.Fatalf("batch expert=%d row=%d got=%g/%g want=%g", expert, r, batchOut[r], batchOut[m.OutDim+r], want)
			}
		}
	}
}
