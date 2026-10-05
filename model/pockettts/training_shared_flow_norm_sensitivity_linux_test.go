//go:build linux

package pockettts

import (
	"math"
	"testing"
)

func TestPocketSharedFlowRoundedNorm(t *testing.T) {
	x := dualVector{value: []float32{-.37, .18, .91, -.21}, tangent: []float32{.2, -.1, .3, -.2}}
	w, bias := []float32{.9, 1.1, .7, 1.2}, []float32{.1, -.2, .3, -.1}
	for _, varianceOnly := range []bool{false, true} {
		got, _, _, _, _, _, _ := sharedFlowRoundedNorm(x, w, bias, 1e-5, varianceOnly)
		var mean, dm, v, dv float64
		for i, a := range x.value {
			mean += float64(a)
			dm += float64(x.tangent[i])
		}
		mean /= 4
		dm /= 4
		denom := float64(4)
		if varianceOnly {
			denom = 3
		}
		for i, a := range x.value {
			c, dc := float64(a)-mean, float64(x.tangent[i])-dm
			v += c * c
			dv += 2 * c * dc
		}
		v /= denom
		dv /= denom
		inv := 1 / math.Sqrt(v+1e-5)
		dinv := -.5 * inv * inv * inv * dv
		for i, a := range x.value {
			c, dc := float64(a)-mean, float64(x.tangent[i])-dm
			if varianceOnly {
				c, dc = float64(a), float64(x.tangent[i])
			}
			want := c * float64(w[i]) * inv
			if !varianceOnly {
				want += float64(bias[i])
			}
			tangent := float64(w[i]) * (dc*inv + c*dinv)
			if math.Abs(float64(got.value[i])-want) > 3e-6 || math.Abs(float64(got.tangent[i])-tangent) > 3e-6 {
				t.Fatal("independent dual norm", varianceOnly, i, got, want, tangent)
			}
		}
	}
	m := tinyTrainableFlowHead()
	c, times, input := []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}
	gv := []float32{.1, .2}
	for _, index := range []int{-1, 0, 1} {
		gt := []float32{.3, -.4}
		if index < 0 {
			gt = []float32{0, 0}
		}
		r, err := (&sharedFlowForwardSensitivityBackend{serial: true, shaderNorm: true}).EvaluateFlow(m, c, times, input, index, gv, gt)
		if err != nil {
			t.Fatal(err)
		}
		want, err := (&sharedFlowForwardSensitivityBackend{}).EvaluateFlow(m, c, times, input, index, gv, gt)
		if err != nil {
			t.Fatal(err)
		}
		if err := validateFlowBackendResult(m, r); err != nil {
			t.Fatal(err)
		}
		for _, pair := range [][2][]float32{{r.Output, want.Output}, {r.Tangent, want.Tangent}, {r.DCondition, want.DCondition}, {r.DInput, want.DInput}, {r.DTimes, want.DTimes}} {
			assertSliceClose(t, "flow rounded norm boundary", pair[0], pair[1], 5e-5)
		}
		for name, v := range r.Parameters {
			assertSliceClose(t, name, v, want.Parameters[name], 5e-5)
		}
	}
}
