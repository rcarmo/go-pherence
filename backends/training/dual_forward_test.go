package training

import (
	"math"
	"testing"
)

func TestTrainingDualNormForwardSchedule(t *testing.T) {
	for kind := uint32(0); kind < 2; kind++ {
		for _, n := range []int{2, 5, 17} {
			x, xt, w, bias := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = float64(float32((i%5 - 2)) * .3)
				xt[i] = float64(float32((i%3 - 1)) * .2)
				w[i] = float64(float32((i%4 - 2)) * .15)
				bias[i] = float64(float32(i%3) * .1)
			}
			// Independent central difference along x+t*xt, not the backward formulas.
			forward := func(tangentStep float64) []float64 {
				values := make([]float64, n)
				mean := 0.
				for i := range x {
					values[i] = x[i] + tangentStep*xt[i]
					mean += values[i]
				}
				mean /= float64(n)
				variance := 0.
				for _, v := range values {
					variance += (v - mean) * (v - mean)
				}
				denom := float64(n)
				if kind == 1 {
					denom--
				}
				inv := 1 / math.Sqrt(variance/denom+1e-4)
				for i := range values {
					if kind == 0 {
						values[i] = (values[i]-mean)*inv*w[i] + bias[i]
					} else {
						values[i] = values[i] * w[i] * inv
					}
				}
				return values
			}
			xf, tf, wf := f32Values(x), f32Values(xt), f32Values(w)
			mean, dm := float32(0), float32(0)
			for i := range xf {
				mean += xf[i]
				dm += tf[i]
			}
			mean /= float32(n)
			dm /= float32(n)
			variance, dvar := float32(0), float32(0)
			for i := range xf {
				c, dc := xf[i]-mean, tf[i]-dm
				variance += c * c
				dvar += 2 * c * dc
			}
			denom := float32(n)
			if kind == 1 {
				denom--
			}
			variance /= denom
			dvar /= denom
			inv := float32(1 / math.Sqrt(float64(variance+float32(1e-4))))
			dinv := -.5 * inv * inv * inv * dvar
			expected, plus, minus := forward(0), forward(1e-5), forward(-1e-5)
			for i := range xf {
				value, tangent := xf[i], tf[i]
				if kind == 0 {
					value -= mean
					tangent -= dm
				}
				y := value * wf[i] * inv
				if kind == 0 {
					y = value*inv*wf[i] + float32(bias[i])
				}
				yt := wf[i] * (tangent*inv + value*dinv)
				if math.Abs(float64(y)-expected[i]) > 2e-6 || math.Abs(float64(yt)-(plus[i]-minus[i])/2e-5) > 2e-6 {
					t.Fatal("dual forward schedule", kind, n, i, y, yt)
				}
			}
		}
	}
	for _, e := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if DualNormForwardShape(1, 2, 0, 0, e) == nil {
			t.Fatal("epsilon admitted")
		}
	}
	if DualNormForwardShape(1, 2, 0, 3, 1e-4) == nil {
		t.Fatal("mode admitted")
	}
}
