//go:build linux

package pockettts

import (
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
)

// Allocation attribution only: no native work, update, or fixture output oracle.
func TestPocketReleasedForwardPlaceholderProfile(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_FORWARD_PLACEHOLDER_PROFILE") != "1" {
		t.Skip("explicit released forward-placeholder allocation profile")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	var corpus struct{ Rows []PreparedTrainingRow }
	if err := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal(err)
	}
	b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
	flow.TrainingBackend = b
	flow.LossBackend = b
	flow.BranchBackend = b
	flow.StepReductionBackend = b
	lm.TransformerBackend = b
	lm.ProjectionBackend = b
	lm.SequenceBackend = b
	lm.EOSBackend = b
	w.TrainingBackend = b
	if os.Getenv("GO_PHERENCE_POCKET_FORWARD_PLACEHOLDER_LONG") == "1" {
		corpus.Rows[0] = integratedLongSyntheticRow(corpus.Rows[0])
	}
	batch := corpus.Rows[0].Batch
	ws, err := newTrainingWorkspace(lm, flow, batch, true)
	if err != nil {
		t.Fatal(err)
	}
	mask := make([]bool, batch.Frames)
	for i := range mask {
		mask[i] = i < corpus.Rows[0].ValidFrames
	}
	b.forwardZeroWorkspace = ws
	defer func() { b.forwardZeroWorkspace = nil }()
	a, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	second, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	roots := sharedHostF32Payload(t, ws, a, second, lm, flow, w, corpus)
	runtime.GC() // live attribution only; never production admission/recovery
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	path := os.Getenv("GO_PHERENCE_POCKET_FORWARD_PLACEHOLDER_LIVE_PROFILE")
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
	t.Logf("two live forward producers + admitted workspace frames=%d F32=%d heapAlloc=%d heapInuse=%d sourceBytes=%d/%d poolExtents=%d", batch.Frames, roots, mem.HeapAlloc, mem.HeapInuse, deviceAuditTapeBytes(t, a.Tape), deviceAuditTapeBytes(t, second.Tape), len(*ws.zeroGradientPool))
	runtime.KeepAlive(ws)
	runtime.KeepAlive(a)
	runtime.KeepAlive(second)
	runtime.KeepAlive(lm)
	runtime.KeepAlive(flow)
	runtime.KeepAlive(w)
	runtime.KeepAlive(corpus)
}
