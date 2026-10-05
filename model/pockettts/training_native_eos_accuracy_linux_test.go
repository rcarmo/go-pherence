//go:build linux

package pockettts

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestPocketNativeEOSAccuracy(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_EOS_ACCURACY_NATIVE") != "1" {
		t.Skip("explicit EOS rounding qualification")
	}
	lm, batch := nativeConditionerInputs(t)
	seq, _, e := trunkCaptureSequence(lm, batch)
	if e != nil {
		t.Fatal(e)
	}
	rows := len(seq) / lm.Hidden
	_, out := lm.Transformer.forwardTraining(rows, seq)
	prefix := rows - batch.Frames
	conditions := out[prefix*lm.Hidden:]
	mask := make([]bool, batch.Frames)
	for i := 0; i < batch.VoiceFrames; i++ {
		mask[i] = true
	}
	p, e := lm.buildEOSProjectionTrainingTape(conditions, mask, .1, 2<<30, true)
	if e != nil {
		t.Fatal(e)
	}
	seedID, logitID := -1, -1
	for _, c := range p.Tape.Commands {
		if c.Op == "eos" && c.Mode == 1 {
			seedID = c.Output
			logitID = c.Inputs[0]
		}
	}
	if seedID < 0 {
		t.Fatal("EOS seed")
	}
	p.Tape.Outputs = append(p.Tape.Outputs, logitID, seedID)
	if e = train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 2<<30); e != nil {
		t.Fatal(e)
	}
	logits := make([]float32, batch.Frames)
	for row := range logits {
		logits[row] = linearForwardTraining(lm.EOS, conditions[row*lm.Hidden:(row+1)*lm.Hidden])[0]
	}
	_, seed, e := EOSLossAndGradient(logits, mask)
	if e != nil {
		t.Fatal(e)
	}
	for i := range seed {
		seed[i] *= .1
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("runtime")
	}
	if e = vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 8 << 20, MaxAllocations: 128}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	owner, e := vulkan.NewVkTrainingTape(ctx, op, p.Tape, 2<<30)
	if e != nil {
		t.Fatal(e)
	}
	got, e := owner.Run(ctx)
	if e != nil {
		t.Fatal("native failure no retry", e)
	}
	type rowError struct {
		Row                                        int
		NativeLogit, CPULogit, NativeSeed, CPUSeed float32
	}
	var report []rowError
	for i := range seed {
		report = append(report, rowError{i, got[len(got)-2][i], logits[i], got[len(got)-1][i], seed[i]})
		t.Logf("row%d logit %.10g/%.10g seed %.10g/%.10g", i, got[len(got)-2][i], logits[i], got[len(got)-1][i], seed[i])
	}
	if e = owner.Close(); e != nil {
		t.Fatal(e)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_EOS_ACCURACY_OUTPUT"), "eos-accuracy.json"), report); e != nil {
		t.Fatal(e)
	}
}
