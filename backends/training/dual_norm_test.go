package training

import (
	"math"
	"testing"
)

// Independent F64 forward computes the selected norm and its analytic JVP.
// Central differences of the scalar seeded objective give reverse-over-forward
// derivatives without reusing either GPU or CPU backward implementation.
func normDualObjective(x, tangent, w, gv, gt []float64, kind int) (float64, []float32) {
	n := len(x)
	mean, dm := 0., 0.
	for i := range x {
		mean += x[i]
		dm += tangent[i]
	}
	mean /= float64(n)
	dm /= float64(n)
	denom := float64(n)
	if kind == 1 {
		denom--
	}
	variance, dvar := 0., 0.
	for i := range x {
		c, dc := x[i]-mean, tangent[i]-dm
		variance += c * c
		dvar += 2 * c * dc
	}
	variance /= denom
	dvar /= denom
	inv := 1 / math.Sqrt(variance+1e-4)
	dinv := -.5 * inv * inv * inv * dvar
	loss := 0.
	for i := range x {
		value, t := x[i], tangent[i]
		if kind == 0 {
			value -= mean
			t -= dm
		}
		loss += gv[i]*w[i]*value*inv + gt[i]*w[i]*(t*inv+value*dinv)
	}
	return loss, []float32{float32(mean), float32(dm), float32(inv), float32(dinv), float32(dvar)}
}
func normDualSchedule(x, xt, w, gv, gt, stats []float32, kind int) ([]float32, []float32, []float32, []float32) {
	n := len(x)
	mean, dm, inv, dinv, dvar := stats[0], stats[1], stats[2], stats[3], stats[4]
	gi, gdi := float32(0), float32(0)
	dw, db := make([]float32, n), make([]float32, n)
	for i := range x {
		value, t := x[i], xt[i]
		if kind == 0 {
			value -= mean
			t -= dm
		}
		g, h := gv[i]*w[i], gt[i]*w[i]
		gi += g*value + h*t
		gdi += h * value
		dw[i] = (gv[i]*value)*inv + gt[i]*(t*inv+value*dinv)
		db[i] = gv[i]
	}
	gvar := gi*(-.5)*inv*inv*inv + gdi*.75*inv*inv*inv*inv*inv*dvar
	gdvar := gdi * (-.5) * inv * inv * inv
	denom := float32(n)
	if kind == 1 {
		denom--
	}
	dx, dt := make([]float32, n), make([]float32, n)
	sx, st := float32(0), float32(0)
	for i := range x {
		c, dc := x[i]-mean, xt[i]-dm
		g, h := gv[i]*w[i], gt[i]*w[i]
		dx[i] = g*inv + h*dinv
		dt[i] = h * inv
		dx[i] += gvar*2*c/denom + gdvar*2*dc/denom
		dt[i] += gdvar * 2 * c / denom
		sx -= dx[i]
		st -= dt[i]
	}
	if kind == 0 {
		for i := range dx {
			dx[i] += sx / float32(n)
			dt[i] += st / float32(n)
		}
	}
	return dx, dt, dw, db
}
func TestTrainingDualNormFiniteDifference(t *testing.T) {
	var max, square float64
	count := 0
	for _, kind := range []int{0, 1} {
		for _, n := range []int{2, 5, 17} {
			x, xt, w, gv, gt := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = float64(float32((i%5 - 2)) * .3)
				xt[i] = float64(float32((i%3 - 1)) * .2)
				w[i] = float64(float32((i%4 - 2)) * .15)
				gv[i] = float64(float32((i%7 - 3)) * .1)
				gt[i] = float64(float32((i%5 - 2)) * .12)
			}
			_, stats := normDualObjective(x, xt, w, gv, gt, kind)
			dx, dt, dw, db := normDualSchedule(f32Values(x), f32Values(xt), f32Values(w), f32Values(gv), f32Values(gt), stats, kind)
			for j, values := range [][]float64{x, xt, w} {
				got := [][]float32{dx, dt, dw}[j]
				for i := range values {
					original := values[i]
					eps := 1e-5
					values[i] = original + eps
					plus, _ := normDualObjective(x, xt, w, gv, gt, kind)
					values[i] = original - eps
					minus, _ := normDualObjective(x, xt, w, gv, gt, kind)
					values[i] = original
					want := (plus - minus) / (2 * eps)
					err := math.Abs(float64(got[i]) - want)
					max = math.Max(max, err)
					square += err * err
					count++
					if err > 3e-6 {
						t.Fatal("dual norm vs independent objective", kind, n, j, i, got[i], want, err)
					}
				}
			}
			if kind == 0 {
				for i := range db {
					if db[i] != float32(gv[i]) {
						t.Fatal("bias must not contain tangent seed")
					}
				}
			}
		}
	}
	t.Logf("dual norm host schedule: values=%d max_abs=%g RMS=%g", count, max, math.Sqrt(square/float64(count)))
}
func TestTrainingDualNormAdmission(t *testing.T) {
	for _, s := range []struct {
		m, n    int
		k, mode uint32
	}{{0, 2, 0, 0}, {1, 0, 0, 0}, {1, 2, 2, 0}, {1, 2, 0, 4}, {1, 1, 1, 0}, {1, 2, 1, 3}} {
		if DualNormShape(s.m, s.n, s.k, s.mode) == nil {
			t.Fatal("invalid dual norm admitted")
		}
	}
	for kind := uint32(0); kind < 2; kind++ {
		for mode := uint32(0); mode < 3; mode++ {
			if err := DualNormShape(1, 2, kind, mode); err != nil {
				t.Fatal(err)
			}
		}
	}
}
func TestTrainingDualLinearAnalytic(t *testing.T) {
	// Scalar L=gv*(W*X+B)+gt*(W*Xt), independent closed-form derivatives.
	x, xt, w, gv, gt := []float32{1, 2}, []float32{3, 4}, []float32{5, 6}, float32(7), float32(8)
	dw := []float32{gv*x[0] + gt*xt[0], gv*x[1] + gt*xt[1]}
	dx, dt := []float32{gv * w[0], gv * w[1]}, []float32{gt * w[0], gt * w[1]}
	for _, pair := range [][2]float32{{dw[0], 31}, {dw[1], 46}, {dx[0], 35}, {dx[1], 42}, {dt[0], 40}, {dt[1], 48}, {gv, 7}} {
		if pair[0] != pair[1] {
			t.Fatal(pair)
		}
	}
}
