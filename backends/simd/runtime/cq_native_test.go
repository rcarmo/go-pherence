package simd

import (
	"encoding/binary"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

// Native dot-dispatch comparison for the released archive's CQ2/CQ4 widths.
// The reference independently performs scalar Sdot over the same unpacked
// groups and input transform; it does not alter global dispatch during tests.
func TestCQNativeDotMatchesScalarForReleasedWidths(t *testing.T) {
	for _, tc := range []struct{ bits, rows, cols int }{{2, 576, 768}, {4, 128, 768}, {2, 48, 768}, {4, 48, 768}} {
		rng := rand.New(rand.NewSource(int64(tc.bits*1000000 + tc.rows*1000 + tc.cols)))
		m, err := NewCQMatrix(tc.rows, tc.cols, tc.bits, makeRandomCQBlob(tc.rows, tc.cols, tc.bits, rng), testCQCodebook())
		if err != nil {
			t.Fatal(err)
		}
		input := make([]float32, tc.cols)
		for i := range input {
			input[i] = (rng.Float32()*2 - 1) * 0.25
		}
		transformed := make([]float32, m.paddedCols)
		m.transformInput(transformed, input)
		want := make([]float32, m.rows)
		var work [cqGroup128]float32
		for row := 0; row < m.rows; row++ {
			rowPacked := m.packed[row*m.packedPerRow : (row+1)*m.packedPerRow]
			rowNorms := m.norms[row*m.groups*2 : (row+1)*m.groups*2]
			for group := 0; group < m.groups; group++ {
				norm := half.F16ToF32(binary.LittleEndian.Uint16(rowNorms[group*2:]))
				if norm == 0 {
					continue
				}
				payload := rowPacked[group*m.packedPerGroup : (group+1)*m.packedPerGroup]
				switch m.bits {
				case 2:
					unpackCQCodebook(&work, payload, 2, m.codebook[:4])
				case 4:
					unpackCQCodebook(&work, payload, 4, m.codebook[12:28])
				}
				want[row] += sdotScalar(work[:], transformed[group*cqGroup128:(group+1)*cqGroup128]) * norm
			}
		}
		got := make([]float32, m.rows)
		if !m.Mul(got, input, 1) {
			t.Fatalf("Mul rejected bits=%d", tc.bits)
		}
		for i, v := range got {
			ref := want[i]
			diff := math.Abs(float64(v - ref))
			if diff > 5e-4+5e-4*math.Abs(float64(ref)) {
				t.Fatalf("arch=%s nativeDot=%v CQ%d rows=%d cols=%d row=%d got=%g scalar=%g diff=%g", runtime.GOARCH, HasDotAsm, tc.bits, tc.rows, tc.cols, i, v, ref, diff)
			}
		}
		t.Logf("arch=%s nativeDot=%v CQ%d rows=%d cols=%d scalar/native checked", runtime.GOARCH, HasDotAsm, tc.bits, tc.rows, tc.cols)
	}
}
