//go:build riscv64

package diffusiongemma

import (
	"math"
	"testing"
)

func TestBuildSelfConditioningProbRows(t *testing.T) {
	logits := [][]float32{{1, 2, -1}, {0.5, -0.25, 1.5}}
	probs := make([]float32, 6)
	if err := buildSelfConditioningProbRows(probs, logits, 2, 3, 1.25); err != nil {
		t.Fatal(err)
	}
	for pos, row := range logits {
		maxV := math.Inf(-1)
		for _, v := range row {
			z := float64(v) * 1.25
			if z > maxV {
				maxV = z
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v)*1.25 - maxV)
		}
		for i, v := range row {
			want := math.Exp(float64(v)*1.25-maxV) / sum
			got := probs[pos*3+i]
			if math.Abs(float64(got)-want) > 1e-6 {
				t.Fatalf("prob pos=%d i=%d got %.8f want %.8f all=%v", pos, i, got, want, probs)
			}
		}
	}
}
