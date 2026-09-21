package gliner2

import (
	"math"
	"testing"
)

func TestRotaryBoundaryInterleaved(t *testing.T) {
	got, err := RotaryBoundary([][]float32{{1, 2, 3, 4}, {1, 2, 3, 4}}, []int{0, 1}, 10000)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float32{1, 2, 3, 4} {
		if got[0][i] != w {
			t.Fatal("zero position changed vector")
		}
	}
	for pair, angle := range []float64{1, .01} {
		even, odd := float64(1+2*pair), float64(2+2*pair)
		want := []float64{even*math.Cos(angle) - odd*math.Sin(angle), even*math.Sin(angle) + odd*math.Cos(angle)}
		for i := range want {
			if math.Abs(float64(got[1][2*pair+i])-want[i]) > 1e-6 {
				t.Fatal(got)
			}
		}
	}
	if _, err := RotaryBoundary([][]float32{{1}}, []int{0}, 10000); err == nil {
		t.Fatal("odd dimension accepted")
	}
}
