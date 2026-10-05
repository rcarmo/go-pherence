//go:build linux

package pockettts

import (
	"math"
	"testing"
)

func TestPocketSharedRoundedNormForward(t *testing.T) {
	x := []float32{-.37, .18, .91, -.21}
	w := []float32{.9, 1.1, .7, 1.2}
	bias := []float32{.1, -.2, .3, -.1}
	tape, out := sharedRoundedNormForward(1, 4, x, w, bias, true)
	var mean, variance float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= 4
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= 4
	inv := 1 / math.Sqrt(variance+1e-5)
	for i := range out {
		want := (float64(x[i])-mean)*inv*float64(w[i]) + float64(bias[i])
		if math.Abs(float64(out[i])-want) > 3e-6 {
			t.Fatal("independent norm", out[i], want)
		}
	}
	if len(tape.inv) != 1 || tape.inv[0] <= 0 {
		t.Fatal("norm stats")
	}
	m := tinyTrainableTransformer()
	rows := 4
	input, seed := make([]float32, rows*m.Width), make([]float32, rows*m.Width)
	for i := range input {
		input[i] = float32(i%7-3) * .1
		seed[i] = float32(i%3-1) * .03
	}
	b := &sharedLinearForwardSensitivityBackend{shaderGelu: true, shaderNorm: true}
	r, err := b.EvaluateTransformer(m, input, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateTransformerBackendResult(m, r, input, seed, rows); err != nil {
		t.Fatal(err)
	}
	want, g, dx, err := m.ForwardBackward(input, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "rounded norm forward", r.Output, want, 3e-5)
	assertSliceClose(t, "rounded norm dx", r.DSequence, dx, 3e-5)
	grads := map[string][]float32{}
	addFlowLMGradients(grads, &FlowLMTrainingGradients{Transformer: g})
	for name, got := range r.Parameters {
		assertSliceClose(t, name, got, grads[name], 3e-5)
	}
}
