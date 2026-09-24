package pockettts

import "testing"

func TestFlowTapeScratchKeepsLiveSlicesDisjoint(t *testing.T) {
	model := tinyTrainableFlowHead()
	scratch, err := newFlowTapeScratch(model)
	if err != nil {
		t.Fatal(err)
	}
	first := scratch.take(3)
	copy(first, []float32{1, 2, 3})
	second := scratch.take(3)
	second[0] = 9
	if first[0] != 1 || scratch.fallbacks != 0 {
		t.Fatal("live scratch slices overlapped or spilled")
	}
	scratch.reset()
	third := scratch.take(3)
	if &third[0] != &first[0] || third[0] != 0 {
		t.Fatal("scratch reset did not clear and reuse storage")
	}
}

func TestFlowStandaloneResultsStayOwned(t *testing.T) {
	model := tinyTrainableFlowHead()
	condition, times := []float32{.2, -.3, .5}, []float32{.25, .8}
	input, seed, tangentSeed := []float32{-.4, .7}, []float32{.6, -.9}, []float32{-.35, .45}
	output, _, _, _, _, err := model.ForwardBackward(condition, times, input, seed)
	if err != nil {
		t.Fatal(err)
	}
	beforeOutput := append([]float32(nil), output...)
	mixed, tangent, _, _, _, _, err := model.ForwardTimeJVPBackward(condition, times, input, 1, seed, tangentSeed)
	if err != nil {
		t.Fatal(err)
	}
	beforeMixed, beforeTangent := append([]float32(nil), mixed...), append([]float32(nil), tangent...)
	result, _, err := model.LSDDistillForwardBackward(condition, .25, .8, input, []float32{.6, -.2}, .12)
	if err != nil {
		t.Fatal(err)
	}
	beforeVelocity, beforeEndpoint := append([]float32(nil), result.Velocity...), append([]float32(nil), result.Endpoint...)
	input[0] += .1
	condition[0] += .1
	if _, _, _, _, _, err = model.ForwardBackward(condition, times, input, seed); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, _, _, err = model.ForwardTimeJVPBackward(condition, times, input, 1, seed, tangentSeed); err != nil {
		t.Fatal(err)
	}
	if _, _, err = model.LSDDistillForwardBackward(condition, .25, .8, input, []float32{.6, -.2}, .12); err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "owned ordinary output", output, beforeOutput, 0)
	assertSliceClose(t, "owned mixed output", mixed, beforeMixed, 0)
	assertSliceClose(t, "owned mixed tangent", tangent, beforeTangent, 0)
	assertSliceClose(t, "owned LSD velocity", result.Velocity, beforeVelocity, 0)
	assertSliceClose(t, "owned LSD endpoint", result.Endpoint, beforeEndpoint, 0)
}
