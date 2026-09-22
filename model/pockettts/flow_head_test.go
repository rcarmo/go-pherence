package pockettts

import (
	"math"
	"testing"
)

func identityLinear(dim int) LinearF32 {
	w := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		w[i*dim+i] = 1
	}
	return LinearF32{Weight: w, Bias: make([]float32, dim), In: dim, Out: dim}
}

func TestFlowHeadSIMDSynthetic(t *testing.T) {
	const d = 4
	zeroMod3 := LinearF32{Weight: make([]float32, 3*d*d), Bias: make([]float32, 3*d), In: d, Out: 3 * d}
	zeroMod2 := LinearF32{Weight: make([]float32, 2*d*d), Bias: make([]float32, 2*d), In: d, Out: 2 * d}
	time := TimestepMLP{
		Frequencies: []float32{1, .1},
		FC1:         identityLinear(d), FC2: identityLinear(d),
		RMSWeight: []float32{1, 1, 1, 1}, RMSEpsilon: 1e-5,
	}
	model := FlowHeadCPU{
		Input: identityLinear(d), Condition: identityLinear(d), Time: []TimestepMLP{time, time},
		Blocks: []AdaLNResidual{{NormWeight: []float32{1, 1, 1, 1}, NormBias: make([]float32, d), FC1: identityLinear(d), FC2: identityLinear(d), Modulation: zeroMod3, Epsilon: 1e-6}},
		Final:  AdaLNFinal{Linear: identityLinear(d), Modulation: zeroMod2, Epsilon: 1e-6},
	}
	out := make([]float32, d)
	if err := model.Forward(out, []float32{.1, .2, .3, .4}, []float32{0, .5}, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	mean := float32(2.5)
	variance := float32(1.25)
	inv := float32(1 / math.Sqrt(float64(variance+1e-6)))
	want := []float32{(1 - mean) * inv, (2 - mean) * inv, (3 - mean) * inv, (4 - mean) * inv}
	for i := range out {
		if math.Abs(float64(out[i]-want[i])) > 2e-5 {
			t.Fatalf("out[%d]=%g want=%g all=%v", i, out[i], want[i], out)
		}
	}
}

func TestFlowHeadRejectsMalformed(t *testing.T) {
	if err := (*FlowHeadCPU)(nil).Forward(nil, nil, nil, nil); err == nil {
		t.Fatal("accepted nil head")
	}
	bad := TimestepMLP{}
	if err := bad.Forward(nil, 0); err == nil {
		t.Fatal("accepted malformed timestep MLP")
	}
}
