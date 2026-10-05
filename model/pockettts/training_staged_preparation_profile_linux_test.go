//go:build linux

package pockettts

import (
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
)

// Released staged graph preparation only. No device work, update, or output
// oracle. Profiles identify construction costs before native qualification.
func TestPocketReleasedStagedPreparationProfile(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_STAGED_PREPARATION_PROFILE") != "1" {
		t.Skip("explicit staged preparation allocation profile")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	owner := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
	flow.TrainingBackend = owner
	flow.LossBackend = owner
	flow.BranchBackend = owner
	flow.StepReductionBackend = owner
	lm.TransformerBackend = owner
	lm.ProjectionBackend = owner
	lm.SequenceBackend = owner
	lm.EOSBackend = owner
	w.TrainingBackend = owner
	batch := FlowLMTrainingBatch{Frames: 375, VoiceFrames: 62, TextTokens: make([]uint32, 512), NormalizedLatents: make([]float32, 375*lm.LatentDim), VoiceLatents: make([]float32, 62*lm.LatentDim)}
	workspace, err := newTrainingWorkspace(lm, flow, batch, true)
	if err != nil {
		t.Fatal(err)
	}
	var pool *trainingZeroPool
	if os.Getenv("GO_PHERENCE_POCKET_STAGED_PREPARATION_POOL") == "1" {
		pool = workspace.zeroGradientPool
	}
	const rows = 950
	sequence, seed := make([]float32, rows*lm.Hidden), make([]float32, rows*lm.Hidden)
	for i := range lm.Transformer.Layers {
		if _, err := buildTransformerLayerStage(lm.Transformer, i, sequence, seed, rows, true, 2<<30, pool); err != nil {
			t.Fatal(err)
		}
	}
	final, err := buildTransformerFinalNormStage(lm.Transformer, sequence, seed, rows, true, 2<<30, pool)
	if err != nil {
		t.Fatal(err)
	}
	// Keep only the actual last native-ready stage, matching sequential execution.
	runtime.GC()
	path := os.Getenv("GO_PHERENCE_POCKET_STAGED_PREPARATION_LIVE_PROFILE")
	if path == "" {
		t.Fatal("live profile output required")
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	err = pprof.Lookup("heap").WriteTo(f, 0)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	t.Logf("staged prep rows=%d layers=%d liveHeap=%d heapSys=%d finalSource=%d", rows, len(lm.Transformer.Layers), mem.HeapAlloc, mem.HeapSys, deviceAuditTapeBytes(t, final.Tape))
	runtime.KeepAlive(workspace)
	runtime.KeepAlive(final)
	runtime.KeepAlive(sequence)
	runtime.KeepAlive(seed)
	runtime.KeepAlive(lm)
}
