package training

import (
	"math"
	"testing"
)

func TestTrainingDualModFiniteDifference(t *testing.T) {
	for kind := uint32(0); kind < 2; kind++ {
		a, at := []float64{.3, -.2}, []float64{.4, .1}
		b, bt := []float64{.6, -.3}, []float64{.2, -.1}
		if kind == 0 {
			b = append(b, .2, .5)
			bt = append(bt, .3, -.4)
		}
		gv, gt := []float64{.7, -.2}, []float64{.1, .3}
		objective := func() float64 {
			loss := 0.
			for i := range a {
				if kind == 0 {
					scale := 1 + b[2+i]
					loss += gv[i]*(a[i]*scale+b[i]) + gt[i]*(at[i]*scale+a[i]*bt[2+i]+bt[i])
				} else {
					loss += gv[i]*a[i]*b[i] + gt[i]*(at[i]*b[i]+a[i]*bt[i])
				}
			}
			return loss
		}
		got := [][]float64{make([]float64, 2), make([]float64, 2), make([]float64, len(b)), make([]float64, len(b))}
		for i := range a {
			scale, st := b[i], bt[i]
			if kind == 0 {
				scale = 1 + b[2+i]
				st = bt[2+i]
				got[2][i] = gv[i]
				got[3][i] = gt[i]
			}
			got[0][i] = gv[i]*scale + gt[i]*st
			got[1][i] = gt[i] * scale
			j := i
			if kind == 0 {
				j += 2
			}
			got[2][j] = gv[i]*a[i] + gt[i]*at[i]
			got[3][j] = gt[i] * a[i]
		}
		for j, values := range [][]float64{a, at, b, bt} {
			for i := range values {
				v := values[i]
				h := 1e-5
				values[i] = v + h
				plus := objective()
				values[i] = v - h
				minus := objective()
				values[i] = v
				if math.Abs(got[j][i]-(plus-minus)/(2*h)) > 1e-8 {
					t.Fatal(kind, j, i, got[j][i], (plus-minus)/(2*h))
				}
			}
		}
	}
	for _, args := range []struct {
		n    int
		k, m uint32
	}{{0, 0, 0}, {1, 2, 0}, {1, 0, 4}, {1<<26 + 1, 1, 0}} {
		if DualModShape(args.n, args.k, args.m) == nil {
			t.Fatal("invalid mod admitted")
		}
	}
}
