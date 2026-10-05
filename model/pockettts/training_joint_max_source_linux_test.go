//go:build linux

package pockettts

import (
	"os"
	"strings"
	"testing"
)

// Read-only geometry admission. It never executes the model or allocates a
// native owner. Checkpoint and prepared source assets remain unchanged.
func TestPocketReleasedJointMaximumSourceAudit(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINT_MAX_SOURCE_AUDIT") != "1" {
		t.Skip("explicit joint maximum source audit")
	}
	lm, flow, w, cfg := releasedTrainingModels(t)
	batch := FlowLMTrainingBatch{Frames: 375, VoiceFrames: 62, TextTokens: make([]uint32, 512), NormalizedLatents: make([]float32, 375*lm.LatentDim), VoiceLatents: make([]float32, 62*lm.LatentDim)}
	for i := range batch.TextTokens {
		batch.TextTokens[i] = uint32(1 + i%(lm.Vocabulary-1))
	}
	mask := make([]bool, batch.Frames)
	for i := range mask {
		mask[i] = i < 374
	}
	owner := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
	lm.EOSBackend = owner
	shape := TrainingShape{MicroBatchRows: 1, TargetFrames: 375, VoiceFrames: 62, TextTokens: 512, GradientAccumulation: 1, FlowBatchMultiplier: 1}
	plan, planErr := PlanTrainingShape(cfg, shape, ProductionTrainingShapeLimits(6<<30))
	t.Logf("joint maximum rows=%d parameterBytes=%d residentEstimate=%d CPUplanError=%v", 1+batch.Frames+batch.VoiceFrames+len(batch.TextTokens), plan.ParameterBytes, plan.ResidentUpperBytes, planErr)
	_, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 2<<30)
	if err == nil {
		t.Fatal("joint forward source unexpectedly fit; renew native joint-storage audit before execution")
	}
	if !strings.Contains(err.Error(), "sequence tape byte ceiling") {
		t.Fatal("unexpected joint forward rejection", err)
	}
	t.Logf("joint forward rejected before materialisation/native allocation: %v", err)
	in := StepReductionInputs{Frames: batch.Frames, Hidden: lm.Hidden, Latent: lm.LatentDim, Mask: mask, DiagonalCondition: make([]float32, batch.Frames*lm.Hidden), DistillCondition: make([]float32, batch.Frames*lm.Hidden), DiagonalTarget: make([]float32, batch.Frames*lm.LatentDim), DistillTarget: make([]float32, batch.Frames*lm.LatentDim), DiagonalLogvar: make([]float32, batch.Frames), DistillLogvar: make([]float32, batch.Frames), RowMetrics: make([]float32, batch.Frames*4), EOS: .1}
	samples := WeightSampleInputs{DiagonalTime: make([]float32, batch.Frames), DistillS: make([]float32, batch.Frames), DistillT: make([]float32, batch.Frames), DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	_, err = lm.buildStepBackboneTrainingTape(w, batch, make([]float32, batch.Frames*lm.Hidden), in, DefaultTrainingStepConfig(), samples, 2<<30, true)
	if err == nil {
		t.Fatal("joint reverse source unexpectedly fit; renew grouped resource audit")
	}
	if !strings.Contains(err.Error(), "sequence tape byte ceiling") {
		t.Fatal("unexpected joint reverse rejection", err)
	}
	t.Logf("joint reverse rejected before materialisation/native allocation: %v", err)
	_ = flow // model topology was loaded/pinned by the fixture, never executed
}
