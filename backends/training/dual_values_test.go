package training

import (
	"math"
	"testing"
)

func TestTrainingDualValueSchedules(t *testing.T) {
	// Integer linear gives an exact contract independent of transcendental error.
	x, xt, w, bias := []float32{1, 2}, []float32{3, 4}, []float32{5, 6}, float32(7)
	primal, tangent := float32(0), float32(0)
	for i := range x {
		primal += x[i] * w[i]
		tangent += xt[i] * w[i]
	}
	primal += bias
	if primal != 24 || tangent != 39 {
		t.Fatal("dual linear value/tangent bias", primal, tangent)
	}
	for _, x := range []float64{-10, -1, 0, .1, 3, 10} {
		xt := .3
		forward := func(t float64) float64 { v := x + t*xt; return v / (1 + math.Exp(-v)) }
		s := float32(1 / (1 + math.Exp(-x)))
		xf, tf := float32(x), float32(xt)
		value := xf * s
		tangent := (tf * s) * (1 + xf*(1-s))
		h := 1e-5
		if math.Abs(float64(value)-forward(0)) > 2e-6 || math.Abs(float64(tangent)-(forward(h)-forward(-h))/(2*h)) > 2e-6 {
			t.Fatal("dual silu forward", x, value, tangent)
		}
	}
	// AdaLN: A=2,At=3,shift=5,scale=7,shiftT=11,scaleT=13.
	av, at, shift, scale, st, scalet := float32(2), float32(3), float32(5), float32(7), float32(11), float32(13)
	if av*(1+scale)+shift != 21 || (at*(1+scale)+av*scalet)+st != 61 {
		t.Fatal("dual mod forward")
	}
}
