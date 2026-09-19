package floatcmp

import (
	"math"
	"testing"
)

func TestCloseFiniteAndExceptional(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	for _, tc := range []struct {
		name string
		a, b []float32
		tol  float32
		want bool
	}{
		{"empty", nil, nil, 0, true},
		{"length", []float32{1}, nil, 1, false},
		{"boundary", []float32{1}, []float32{1.25}, 0.25, true},
		{"outside", []float32{1}, []float32{1.25}, 0.24, false},
		{"nan", []float32{nan}, []float32{1}, 1, false},
		{"two nan", []float32{nan}, []float32{nan}, 1, false},
		{"same inf", []float32{inf}, []float32{inf}, 0, true},
		{"opposite inf", []float32{inf}, []float32{-inf}, 1, false},
		{"infinite tolerance", nil, nil, inf, false},
		{"nan tolerance", []float32{1}, []float32{1}, nan, false},
		{"negative tolerance", nil, nil, -1, false},
		{"wide subtraction", []float32{math.MaxFloat32}, []float32{-math.MaxFloat32}, math.MaxFloat32, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := Close(tc.a, tc.b, tc.tol); got != tc.want {
				t.Fatalf("got=%v want=%v", got, tc.want)
			}
		})
	}
}
