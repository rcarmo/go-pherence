package diffusiongemma

import (
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestDecodedExpertBatchMatchesPerPositionGEMV(t *testing.T) {
	const (
		batch        = 16
		hiddenSize   = 32
		intermediate = 8
	)
	x := makePattern(batch*hiddenSize, 0.011, -0.13)
	ew := decodedExpertWeights{
		gateW: makePattern(intermediate*hiddenSize, 0.017, 0.07),
		upW:   makePattern(intermediate*hiddenSize, -0.019, 0.03),
		downW: makePattern(hiddenSize*intermediate, 0.023, -0.04),
	}

	want := make([]float32, batch*hiddenSize)
	gateRow := make([]float32, intermediate)
	upRow := make([]float32, intermediate)
	actRow := make([]float32, intermediate)
	outRow := make([]float32, hiddenSize)
	for pos := 0; pos < batch; pos++ {
		row := x[pos*hiddenSize : (pos+1)*hiddenSize]
		if !simd.GemvRows(gateRow, row, ew.gateW, intermediate, hiddenSize) || !simd.GemvRows(upRow, row, ew.upW, intermediate, hiddenSize) {
			t.Fatalf("reference GEMV rejected")
		}
		if !simd.GELUTanhMulTo(actRow, gateRow, upRow) {
			t.Fatalf("reference activation rejected")
		}
		if !simd.GemvRows(outRow, actRow, ew.downW, hiddenSize, intermediate) {
			t.Fatalf("reference down GEMV rejected")
		}
		copy(want[pos*hiddenSize:(pos+1)*hiddenSize], outRow)
	}

	got := make([]float32, batch*hiddenSize)
	gate := make([]float32, batch*intermediate)
	up := make([]float32, batch*intermediate)
	act := make([]float32, batch*intermediate)
	if err := runDecodedExpertBatch(got, gate, up, act, x, ew, batch, hiddenSize, intermediate); err != nil {
		t.Fatalf("runDecodedExpertBatch: %v", err)
	}

	for i := range want {
		diff := math.Abs(float64(want[i] - got[i]))
		denom := math.Max(1, math.Abs(float64(want[i])))
		if diff > 1e-4 && diff/denom > 1e-6 {
			t.Fatalf("batched expert mismatch at %d: want %.8f got %.8f diff %.8g rel %.8g", i, want[i], got[i], diff, diff/denom)
		}
	}
}
