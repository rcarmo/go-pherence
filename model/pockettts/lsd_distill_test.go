package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestLSDDistillMinimalFiniteDifference(t *testing.T) {
	model := tinyTrainableFlowHead()
	condition := []float32{.2, -.3, .5}
	noise := []float32{-.4, .7}
	target := []float32{.6, -.2}
	s, end := float32(.25), float32(.8)
	logvar := float32(.12)
	result, gradients, err := model.LSDDistillForwardBackward(condition, s, end, noise, target, logvar)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Residual) != 2 || gradients.Flow == nil {
		t.Fatalf("malformed result: %+v %+v", result, gradients)
	}
	evaluate := func() float64 {
		value, _, e := model.LSDDistillForwardBackward(condition, s, end, noise, target, logvar)
		if e != nil {
			t.Fatal(e)
		}
		return value.Loss
	}
	checkCentralDifference(t, "lsd.condition", condition, gradients.Condition, evaluate, 1.5e-2)
	checkCentralDifference(t, "lsd.noise", noise, gradients.Noise, evaluate, 1.5e-2)
	checkCentralDifference(t, "lsd.target", target, gradients.Target, evaluate, 1.5e-2)
	checkScalarCentralDifference(t, "lsd.s", &s, gradients.DS, evaluate, 2e-2)
	checkScalarCentralDifference(t, "lsd.t", &end, gradients.DT, evaluate, 2e-2)
	checkScalarCentralDifference(t, "lsd.logvar", &logvar, gradients.DLogVariance, evaluate, 2e-3)
	params, grads := flowHeadParameterPairs(model, gradients.Flow)
	for i := range params {
		checkCentralDifference(t, "lsd."+params[i].name, params[i].values, grads[i].values, evaluate, 2.5e-2)
	}
}

func TestLSDDistillMinimalPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatal("unexpected LSD oracle schema/revision")
	}
	model := flowHeadFromOracle(t, oracle.Parameters)
	want := oracle.LSDMinimal
	result, gradients, err := model.LSDDistillForwardBackward(want.Condition, want.S, want.T, want.Noise, want.Target, want.LogVariance)
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "LSD PyTorch loss", result.Loss, float64(want.Loss), 2e-6)
	assertClose64(t, "LSD PyTorch raw square", result.RawSquare, float64(want.RawSquare), 2e-6)
	assertSliceClose(t, "LSD PyTorch velocity", result.Velocity, want.Velocity, 5e-6)
	assertSliceClose(t, "LSD PyTorch dvdt", result.TimeDerivative, want.TimeDerivative, 5e-6)
	assertSliceClose(t, "LSD PyTorch endpoint", result.Endpoint, want.Endpoint, 5e-6)
	assertSliceClose(t, "LSD PyTorch residual", result.Residual, want.Residual, 5e-6)
	assertSliceClose(t, "LSD PyTorch condition", gradients.Condition, want.DCondition, 2e-5)
	assertSliceClose(t, "LSD PyTorch noise", gradients.Noise, want.DNoise, 2e-5)
	assertSliceClose(t, "LSD PyTorch target", gradients.Target, want.DTarget, 2e-5)
	assertClose64(t, "LSD PyTorch ds", float64(gradients.DS), float64(want.DS), 2e-5)
	assertClose64(t, "LSD PyTorch dt", float64(gradients.DT), float64(want.DT), 2e-5)
	assertClose64(t, "LSD PyTorch logvar", float64(gradients.DLogVariance), float64(want.DLogVariance), 2e-5)
	gradientMap := flowHeadGradientMap(gradients.Flow)
	assertFlowHeadGradientCoverage(t, oracle, gradientMap)
	for name, got := range gradientMap {
		assertSliceClose(t, "LSD PyTorch "+name, got, want.Gradients[name], 3e-5)
	}
}

func TestLSDDistillMinimalStopsEndpointParameterGradient(t *testing.T) {
	model := tinyTrainableFlowHead()
	condition := []float32{.2, -.3, .5}
	noise := []float32{-.4, .7}
	target := []float32{.6, -.2}
	result, gradients, err := model.LSDDistillForwardBackward(condition, .25, .8, noise, target, .12)
	if err != nil {
		t.Fatal(err)
	}
	// Reconstruct the endpoint VJP and ensure it would have non-zero direct
	// parameter gradients. LSDDistillForwardBackward must discard them.
	dEndpoint := make([]float32, len(result.Residual))
	scale := float32(math.Exp(.12) / float64(len(dEndpoint)))
	for i := range dEndpoint {
		dEndpoint[i] = -2 * result.Residual[i] * scale
	}
	xs := []float32{.25*target[0] + .75*noise[0], .25*target[1] + .75*noise[1]}
	xt := []float32{xs[0] + .55*result.Velocity[0], xs[1] + .55*result.Velocity[1]}
	_, endpointGradients, _, _, _, err := model.ForwardBackward(condition, []float32{.8, .8}, xt, dEndpoint)
	if err != nil {
		t.Fatal(err)
	}
	nonzero := false
	for _, values := range flowHeadGradientMap(endpointGradients) {
		for _, value := range values {
			if value != 0 {
				nonzero = true
				break
			}
		}
	}
	if !nonzero || gradients.Flow == nil {
		t.Fatal("endpoint direct gradient fixture is degenerate")
	}
}

func TestLSDDistillRejectsMalformed(t *testing.T) {
	model := tinyTrainableFlowHead()
	if _, _, err := model.LSDDistillForwardBackward([]float32{0, 0, 0}, .8, .2, []float32{0, 0}, []float32{0, 0}, 0); err == nil {
		t.Fatal("accepted s > t")
	}
	if _, _, err := model.LSDDistillForwardBackward([]float32{0, 0, 0}, .2, .8, []float32{0}, []float32{0, 0}, 0); err == nil {
		t.Fatal("accepted latent shape mismatch")
	}
}

func checkScalarCentralDifference(t *testing.T, name string, value *float32, gradient float32, evaluate func() float64, tolerance float64) {
	t.Helper()
	const step = float32(2e-3)
	original := *value
	*value = original + step
	plus := evaluate()
	*value = original - step
	minus := evaluate()
	*value = original
	numeric := (plus - minus) / (2 * float64(step))
	if math.Abs(float64(gradient)-numeric) > tolerance {
		t.Fatalf("%s gradient=%g numeric=%g diff=%g", name, gradient, numeric, math.Abs(float64(gradient)-numeric))
	}
}
