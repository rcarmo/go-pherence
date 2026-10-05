package training

import (
	"math"
	"testing"
)

func TestTrainingDualLinearFiniteDifference(t *testing.T) {
	m, k, n := 3, 2, 4
	x, xt, w, b, gv, gt := make([]float64, m*k), make([]float64, m*k), make([]float64, n*k), make([]float64, n), make([]float64, m*n), make([]float64, m*n)
	for i := range x {
		x[i] = float64(float32(i%5-2) * .3)
		xt[i] = float64(float32(i%3-1) * .2)
	}
	for i := range w {
		w[i] = float64(float32(i%7-3) * .1)
	}
	for i := range gv {
		gv[i] = float64(float32(i%4-2) * .2)
		gt[i] = float64(float32(i%5-2) * .15)
	}
	objective := func() float64 {
		loss := 0.
		for r := 0; r < m; r++ {
			for out := 0; out < n; out++ {
				v, t := b[out], 0.
				for col := 0; col < k; col++ {
					v += x[r*k+col] * w[out*k+col]
					t += xt[r*k+col] * w[out*k+col]
				}
				loss += gv[r*n+out]*v + gt[r*n+out]*t
			}
		}
		return loss
	}
	xf, tf, wf, gf, hf := f32Values(x), f32Values(xt), f32Values(w), f32Values(gv), f32Values(gt)
	dx, dt, dw, db := make([]float32, len(x)), make([]float32, len(xt)), make([]float32, len(w)), make([]float32, len(b))
	for i := range dx {
		row, col := i/k, i%k
		for q := 0; q < n; q++ {
			dx[i] += gf[row*n+q] * wf[q*k+col]
			dt[i] += hf[row*n+q] * wf[q*k+col]
		}
	}
	for i := range dw {
		row, col := i/k, i%k
		for q := 0; q < m; q++ {
			dw[i] += gf[q*n+row] * xf[q*k+col]
		}
		for q := 0; q < m; q++ {
			dw[i] += hf[q*n+row] * tf[q*k+col]
		}
	}
	for i := range db {
		for q := 0; q < m; q++ {
			db[i] += gf[q*n+i]
		}
	}
	var max, square float64
	count := 0
	for j, values := range [][]float64{x, xt, w, b} {
		got := [][]float32{dx, dt, dw, db}[j]
		for i := range values {
			v := values[i]
			h := 1e-5
			values[i] = v + h
			plus := objective()
			values[i] = v - h
			minus := objective()
			values[i] = v
			expected := (plus - minus) / (2 * h)
			err := math.Abs(float64(got[i]) - expected)
			max = math.Max(max, err)
			square += err * err
			count++
			if err > 2e-6 {
				t.Fatal("dual linear host schedule vs independent F64 forward objective", j, i, got[i], expected)
			}
		}
	}
	t.Logf("dual linear host schedule: values=%d max_abs=%g RMS=%g", count, max, math.Sqrt(square/float64(count)))
}
