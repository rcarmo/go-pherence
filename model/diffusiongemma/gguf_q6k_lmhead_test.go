package diffusiongemma

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGGUFQ6KLMHeadRowDotMatchesQ8KRoundedDequantOracle(t *testing.T) {
	m := &gguf.QuantMatrix{Name: "synthetic.q6k", QType: gguf.QuantQ6_K, InDim: 256, OutDim: 3}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*m.OutDim)
	for r := 0; r < m.OutDim; r++ {
		blk := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for i := 0; i < 128; i++ {
			blk[i] = byte((i*13 + r*17) & 0xff)
		}
		for i := 0; i < 64; i++ {
			blk[128+i] = byte((i*7 + r*19) & 0xff)
		}
		for i := 0; i < 16; i++ {
			blk[192+i] = byte(int8((i%9)-4) + int8(r))
		}
		binary.LittleEndian.PutUint16(blk[208:210], half.F32ToF16(0.018+float32(r)*0.003))
	}
	x := make([]float32, m.InDim)
	for i := range x {
		x[i] = float32((i%23)-11) * 0.017
	}
	q8x := append([]float32(nil), x...)
	quantizeDequantQ8KForExpertDot(q8x, q8x)
	row := make([]float32, m.InDim)
	pre, err := prequantizeQ8KRowsForLMHead(x, 1, m.InDim)
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < m.OutDim; r++ {
		if err := m.DequantRowTo(row, r); err != nil {
			t.Fatal(err)
		}
		var want float32
		for i := range row {
			want += row[i] * q8x[i]
		}
		got, err := ggufQ6KMatrixRowDotQ8K(m, r, x)
		if err != nil {
			t.Fatal(err)
		}
		gotPre, err := ggufQ6KMatrixRowDotPrequant(m, r, pre.ds, pre.qs)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(got-want)) > 2e-4 {
			t.Fatalf("row=%d got=%g want=%g", r, got, want)
		}
		if math.Abs(float64(gotPre-got)) > 1e-6 {
			t.Fatalf("row=%d prequant=%g single=%g", r, gotPre, got)
		}
	}
}
