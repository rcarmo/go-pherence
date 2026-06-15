package diffusiongemma

import (
	"encoding/binary"
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestDirectQuantExpertPolicy(t *testing.T) {
	le := ggufLayerExperts{
		gateUp: &gguf.ExpertMatrices{QType: gguf.QuantQ4_K},
		down:   &gguf.ExpertMatrices{QType: gguf.QuantQ8_0},
	}
	for _, n := range []int{1, 3, 8} {
		if got, want := useDirectQ4GateUpRows(le, n), simd.HasDotU4F32SIMD; got != want {
			t.Fatalf("Q4 direct policy n=%d got=%v want SIMD=%v", n, got, want)
		}
		if got, want := useDirectQ8DownRows(le, n), simd.HasDotI8F32SIMD; got != want {
			t.Fatalf("Q8 direct policy n=%d got=%v want SIMD=%v", n, got, want)
		}
	}
	for _, n := range []int{0, 9, 16} {
		if useDirectQ4GateUpRows(le, n) {
			t.Fatalf("Q4 direct policy accepted n=%d", n)
		}
		if useDirectQ8DownRows(le, n) {
			t.Fatalf("Q8 direct policy accepted n=%d", n)
		}
	}
	if useDirectQ4GateUpRows(ggufLayerExperts{gateUp: &gguf.ExpertMatrices{QType: gguf.QuantQ8_0}}, 1) {
		t.Fatal("Q4 direct policy accepted non-Q4 gate/up")
	}
	if useDirectQ8DownRows(ggufLayerExperts{down: &gguf.ExpertMatrices{QType: gguf.QuantQ4_K}}, 1) {
		t.Fatal("Q8 direct policy accepted non-Q8 down")
	}
}

func TestDirectQuantBatchDotsMatchDequantOracleAcrossPolicyBatches(t *testing.T) {
	q4 := syntheticQ4KExpertMatrixForTest(t, 256, 6, 3)
	q8 := syntheticQ8ExpertMatrixForTest(t, 32, 5, 3)
	for _, nPos := range []int{1, 3, diffusionGemmaDirectQuantMaxBatch} {
		batchQ4 := syntheticBatchForTest(nPos, q4.InDim, 0.011)
		batchQ8 := syntheticBatchForTest(nPos, q8.InDim, -0.019)
		assertQ4BatchDotsMatchOracle(t, q4, batchQ4, nPos)
		assertQ8BatchDotsMatchOracle(t, q8, batchQ8, nPos, 0.75)
	}
}

func syntheticQ4KExpertMatrixForTest(t *testing.T, inDim, outDim, experts int) *gguf.ExpertMatrices {
	t.Helper()
	m := &gguf.ExpertMatrices{Name: "synthetic.policy.q4", QType: gguf.QuantQ4_K, InDim: inDim, OutDim: outDim, Experts: experts}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*outDim*experts)
	for r := 0; r < outDim*experts; r++ {
		blk := m.Raw[r*rowBytes : (r+1)*rowBytes]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.019+float32(r)*0.001))
		binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.003+float32(r%5)*0.0007))
		for i := 0; i < 12; i++ {
			blk[4+i] = byte(3 + (i*5+r)%19)
		}
		for i := 0; i < 128; i++ {
			blk[16+i] = byte((i*13 + r*17) & 0xff)
		}
	}
	return m
}

func syntheticQ8ExpertMatrixForTest(t *testing.T, inDim, outDim, experts int) *gguf.ExpertMatrices {
	t.Helper()
	m := &gguf.ExpertMatrices{Name: "synthetic.policy.q8", QType: gguf.QuantQ8_0, InDim: inDim, OutDim: outDim, Experts: experts}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*outDim*experts)
	for r := 0; r < outDim*experts; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		binary.LittleEndian.PutUint16(row[:2], half.F32ToF16(0.04+float32(r)*0.002))
		for i := 0; i < inDim; i++ {
			row[2+i] = byte(int8((r*7+i*3)%31 - 15))
		}
	}
	return m
}

func syntheticBatchForTest(nPos, dim int, scale float32) []float32 {
	out := make([]float32, nPos*dim)
	for i := range out {
		out[i] = float32((i%23)-11) * scale
	}
	return out
}

func assertQ4BatchDotsMatchOracle(t *testing.T, m *gguf.ExpertMatrices, batch []float32, nPos int) {
	t.Helper()
	row := make([]float32, m.InDim)
	for expert := 0; expert < m.Experts; expert++ {
		for r := 0; r < m.OutDim; r++ {
			if err := m.DequantExpertRowTo(row, expert, r); err != nil {
				t.Fatal(err)
			}
			out := make([]float32, nPos*m.OutDim)
			if err := ggufQ4KExpertRowDotBatchTo(m, expert, r, batch, nPos, out[r:], m.OutDim); err != nil {
				t.Fatal(err)
			}
			for pos := 0; pos < nPos; pos++ {
				want := simd.Sdot(row, batch[pos*m.InDim:(pos+1)*m.InDim])
				got := out[pos*m.OutDim+r]
				if math.Abs(float64(got-want)) > 1e-4 {
					t.Fatalf("Q4 expert=%d row=%d pos=%d got=%g want=%g", expert, r, pos, got, want)
				}
			}
		}
	}
}

func assertQ8BatchDotsMatchOracle(t *testing.T, m *gguf.ExpertMatrices, batch []float32, nPos int, scale float32) {
	t.Helper()
	row := make([]float32, m.InDim)
	for expert := 0; expert < m.Experts; expert++ {
		for r := 0; r < m.OutDim; r++ {
			if err := m.DequantExpertRowTo(row, expert, r); err != nil {
				t.Fatal(err)
			}
			for i := range row {
				row[i] *= scale
			}
			out := make([]float32, nPos*m.OutDim)
			if err := ggufQ8_0ExpertRowDotBatchTo(m, expert, r, batch, nPos, out[r:], m.OutDim, scale); err != nil {
				t.Fatal(err)
			}
			for pos := 0; pos < nPos; pos++ {
				want := simd.Sdot(row, batch[pos*m.InDim:(pos+1)*m.InDim])
				got := out[pos*m.OutDim+r]
				if math.Abs(float64(got-want)) > 1e-5 {
					t.Fatalf("Q8 expert=%d row=%d pos=%d got=%g want=%g", expert, r, pos, got, want)
				}
			}
		}
	}
}
