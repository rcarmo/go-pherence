package pockettts

import "testing"

// Compare identical released-topology weights and sampled inputs through the
// owned-tape reference and the request-owned bounded tape scratch path.
func TestReleasedTrainingFlowTapeScratchParity(t *testing.T) {
	lm, flow, weighting, batch, samples, plan := releasedTrainingRow(t, 32)
	ownedWorkspace, err := NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan)
	if err != nil {
		t.Fatal(err)
	}
	metricsOwned, gradientsOwned, err := pocketTrainingStepIntoTapes(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig(), ownedWorkspace, false)
	if err != nil {
		t.Fatal(err)
	}
	// Owned-path gradients include request-owned buffers; copy them before a
	// second call on a different workspace for a strict all-gradient comparison.
	want := make(map[string][]float32)
	ownedMap, err := fullGradientMap(gradientsOwned)
	if err != nil {
		t.Fatal(err)
	}
	for name, values := range ownedMap {
		want[name] = append([]float32(nil), values...)
	}
	wantInput := append([]float32(nil), gradientsOwned.Inputs.NormalizedLatents...)
	wantVoice := append([]float32(nil), gradientsOwned.Inputs.VoiceLatents...)
	wantDiagonal := append([]float32(nil), gradientsOwned.DiagonalLogVariance...)
	wantDistill := append([]float32(nil), gradientsOwned.DistillLogVariance...)
	scratchWorkspace, err := NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan)
	if err != nil {
		t.Fatal(err)
	}
	metricsScratch, gradientsScratch, err := PocketTrainingStepInto(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig(), scratchWorkspace)
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "released tape diagonal", metricsScratch.FlowDiagonal, metricsOwned.FlowDiagonal, 0)
	assertClose64(t, "released tape distill", metricsScratch.FlowDistill, metricsOwned.FlowDistill, 0)
	assertClose64(t, "released tape flow loss", metricsScratch.FlowLoss, metricsOwned.FlowLoss, 0)
	assertClose64(t, "released tape EOS", metricsScratch.EOS, metricsOwned.EOS, 0)
	assertClose64(t, "released tape loss", metricsScratch.Loss, metricsOwned.Loss, 0)
	got, err := fullGradientMap(gradientsScratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("gradient names=%d want=%d", len(got), len(want))
	}
	for name, values := range want {
		actual, ok := got[name]
		if !ok {
			t.Fatalf("missing gradient %q", name)
		}
		assertSliceClose(t, name, actual, values, 0)
	}
	assertSliceClose(t, "released tape normalized latents", gradientsScratch.Inputs.NormalizedLatents, wantInput, 0)
	assertSliceClose(t, "released tape voice latents", gradientsScratch.Inputs.VoiceLatents, wantVoice, 0)
	assertSliceClose(t, "released tape diagonal logvar", gradientsScratch.DiagonalLogVariance, wantDiagonal, 0)
	assertSliceClose(t, "released tape distill logvar", gradientsScratch.DistillLogVariance, wantDistill, 0)
	if scratchWorkspace.primaryFlowTape.fallbacks != 0 || scratchWorkspace.endpointFlowTape.fallbacks != 0 || scratchWorkspace.primaryFlowBackward.fallbacks != 0 || scratchWorkspace.endpointFlowBackward.fallbacks != 0 {
		t.Fatalf("released flow scratch spilled: primary=%d endpoint=%d backward=%d endpoint_backward=%d", scratchWorkspace.primaryFlowTape.fallbacks, scratchWorkspace.endpointFlowTape.fallbacks, scratchWorkspace.primaryFlowBackward.fallbacks, scratchWorkspace.endpointFlowBackward.fallbacks)
	}
}
