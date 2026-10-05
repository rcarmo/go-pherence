//go:build linux

package pockettts

import (
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
)

func TestPocketReleasedStagedHostLedger(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_STAGED_HOST_LEDGER") != "1" {
		t.Skip("explicit staged host ledger")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	var corpus struct{ Rows []PreparedTrainingRow }
	if err := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal(err)
	}
	row := integratedLongSyntheticRow(corpus.Rows[0])
	corpus.Rows[0] = row
	tr, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	host, err := tr.State()
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := sharedInitialSnapshot(host)
	if err != nil {
		t.Fatal(err)
	}
	b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1", host: host}
	b.rememberSharedInitialMoments(snapshot)
	flow.TrainingBackend = b
	flow.LossBackend = b
	flow.BranchBackend = b
	flow.StepReductionBackend = b
	lm.TransformerBackend = b
	lm.ProjectionBackend = b
	lm.SequenceBackend = b
	lm.EOSBackend = b
	w.TrainingBackend = b
	ws, err := newTrainingWorkspace(lm, flow, row.Batch, true)
	if err != nil {
		t.Fatal(err)
	}
	base := sharedHostF32Payload(tr, b, ws, corpus)
	rows := 1 + row.Batch.Frames + row.Batch.VoiceFrames + len(row.Batch.TextTokens)
	boundaries := make([][]float32, len(lm.Transformer.Layers)+1)
	for i := range boundaries {
		boundaries[i] = make([]float32, rows*lm.Hidden)
	}
	seed := make([]float32, rows*lm.Hidden)
	hidden := make([]float32, row.Batch.Frames*lm.Hidden)
	ticket := &stagedBackboneBoundary{model: lm, batch: row.Batch, hidden: append([]float32(nil), hidden...), mask: make([]bool, row.Batch.Frames), weight: .1}
	ticket.batch.TextTokens = append([]uint32(nil), row.Batch.TextTokens...)
	ticket.batch.VoiceLatents = append([]float32(nil), row.Batch.VoiceLatents...)
	ticket.batch.NormalizedLatents = append([]float32(nil), row.Batch.NormalizedLatents...)
	b.stagedBackbone = ticket
	b.transformerStages = &transformerStageBoundary{owner: b, model: lm.Transformer, rows: rows, boundaries: boundaries}
	reverse, err := buildTransformerLayerStage(lm.Transformer, 0, boundaries[0], seed, rows, false, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	// Retain row reductions and direct gradients which coexist with stage reverse.
	reduction := StepReductionInputs{Frames: row.Batch.Frames, Hidden: lm.Hidden, Latent: lm.LatentDim, Mask: make([]bool, row.Batch.Frames), DiagonalCondition: make([]float32, row.Batch.Frames*lm.Hidden), DistillCondition: make([]float32, row.Batch.Frames*lm.Hidden), DiagonalTarget: make([]float32, row.Batch.Frames*lm.LatentDim), DistillTarget: make([]float32, row.Batch.Frames*lm.LatentDim), DiagonalLogvar: make([]float32, row.Batch.Frames), DistillLogvar: make([]float32, row.Batch.Frames), RowMetrics: make([]float32, row.Batch.Frames*4)}
	all := sharedHostF32Payload(tr, b, ws, corpus, seed, hidden, reverse, reduction)
	var parameterBytes int64
	for _, x := range tr.params {
		parameterBytes += int64(len(x)) * 4
	}
	runtime.GC() // attribution only, never production memory policy
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	path := os.Getenv("GO_PHERENCE_POCKET_STAGED_HOST_LIVE_PROFILE")
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
	// Conservative shared resident retains branch allowance; live moments are
	// additional independent host arrays after first update/restore.
	forecast := all + 2*parameterBytes + 1839522904 + 310695284 + 219029196
	t.Logf("staged host rows%d frames%d voice%d tokens%d baseF32=%d allF32=%d forecast=%d remaining6GiB=%d HeapAlloc=%d HeapSys=%d", rows, row.Batch.Frames, row.Batch.VoiceFrames, len(row.Batch.TextTokens), base, all, forecast, int64(6<<30)-forecast, mem.HeapAlloc, mem.HeapSys)
	if forecast >= 6<<30 {
		t.Fatal("staged host payload exceeds cap")
	}
	runtime.KeepAlive(tr)
	runtime.KeepAlive(b)
	runtime.KeepAlive(ws)
	runtime.KeepAlive(corpus)
	runtime.KeepAlive(seed)
	runtime.KeepAlive(hidden)
	runtime.KeepAlive(reverse)
	runtime.KeepAlive(reduction)
}
