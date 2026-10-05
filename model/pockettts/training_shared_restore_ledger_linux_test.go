//go:build linux

package pockettts

import (
	"os"
	"runtime"
	"testing"
)

// Attribution only: read the existing coherent CPU checkpoint, perform public
// restore in a fresh trainer, then take the owned shared boundary and workspace.
// No update, native buffer allocation, export or file publication occurs.
func TestPocketSharedReleasedRestorePayloadLedger(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_RESTORE_LEDGER") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_RESTORE_LEDGER=1 with pinned released assets/CPU bundle")
	}
	checkpoint := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_CPU_MODEL", "d0d366f789a6ab0078bd0f1534d16a38dddb4e08a3193797be23c9f6b1794e5c", 1431217112)
	lm, flow, w, _ := releasedTrainingModels(t)
	tr, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	var memory runtime.MemStats
	sample := func(phase string, payload int64) {
		runtime.ReadMemStats(&memory)
		t.Logf("phase=%s retainedF32=%d HeapAlloc=%d HeapInuse=%d HeapSys=%d TotalAlloc=%d NumGC=%d", phase, payload, memory.HeapAlloc, memory.HeapInuse, memory.HeapSys, memory.TotalAlloc, memory.NumGC)
	}
	sample("fresh-trainer", sharedHostF32Payload(tr))
	state, err := LoadFullTrainingStateBinary(checkpoint, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	if state.Step != 1 || state.AdamW != tr.Config || state.EMADecay != tr.EMADecay {
		t.Fatal("restore fixture config")
	}
	sample("decoded-plus-fresh", sharedHostF32Payload(tr, state))
	owned := os.Getenv("GO_PHERENCE_POCKET_SHARED_RESTORE_LEDGER_MODE") == "owned"
	b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
	if owned {
		if err := b.restoreSharedOwnedState(tr, &state); err != nil {
			t.Fatal(err)
		}
	} else if err := tr.LoadState(state); err != nil {
		t.Fatal(err)
	}
	check := state
	if owned {
		check = b.host
	}
	if !sameResidentLiveBoundary(check, tr) {
		t.Fatal("restore values")
	}
	sample("restored-plus-decoded", sharedHostF32Payload(tr, state, b))
	check = FullTrainingState{}
	// This explicit drop models the caller return; no production GC policy is
	// changed. Memory statistics before forced GC retain real dead-heap costs.
	state = FullTrainingState{}
	var host FullTrainingState
	if owned {
		host = b.host
	} else {
		host, err = tr.State()
		if err != nil {
			t.Fatal(err)
		}
		b.host, b.trainer = host, tr
	}
	snapshot, err := sharedInitialSnapshot(host)
	if err != nil {
		t.Fatal(err)
	}
	b.rememberSharedInitialMoments(snapshot)
	if b.sharedInitialMoments != nil || len(host.Adam) != len(host.Params) {
		t.Fatal("restored moments duplicated")
	}
	if err := validateSharedSnapshotStorage(host, tr); err != nil {
		t.Fatal(err)
	}
	sample("restored-plus-owned-boundary", sharedHostF32Payload(tr, b, snapshot))
	// Workspace geometry only: no codec or model inference. Corpus latents are
	// irrelevant to its fixed shape/capacity and all future computations are idle.
	batch := FlowLMTrainingBatch{Frames: 14, VoiceFrames: 13, TextTokens: []uint32{1}, NormalizedLatents: make([]float32, 14*lm.LatentDim), VoiceLatents: make([]float32, 13*lm.LatentDim)}
	workspace, err := NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		t.Fatal(err)
	}
	payload := sharedHostF32Payload(tr, b, workspace, batch)
	sample("restored-boundary-workspace-before-attribution-GC", payload)
	const nativeResident = 1839522904
	const laterTape = 317970992
	const mapping = 219029196
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	t.Logf("restore retained-payload-plus-native-forecast=%d HeapAlloc-plus-native-forecast-beforeGC=%d cap6GiB=%d; native/metadata/padding/mapping retention and GC policy are unqualified", payload+nativeResident+laterTape+mapping, int64(before.HeapAlloc)+nativeResident+laterTape+mapping, int64(6<<30))
	runtime.GC() // attribution boundary only, never restore/admission workaround
	sample("after-attribution-GC", sharedHostF32Payload(tr, b, workspace, batch))
	runtime.KeepAlive(tr)
	runtime.KeepAlive(b)
	runtime.KeepAlive(workspace)
	runtime.KeepAlive(snapshot)
	runtime.KeepAlive(batch)
}
