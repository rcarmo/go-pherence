//go:build linux

package pockettts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Attribution only: restore existing owned checkpoint then select EMA and
// frozen Mimi export tensors. No writer/publication, update, inference or native
// allocation executes. Native resident payload remains a separate forecast.
func TestPocketSharedReleasedExportCoexistenceLedger(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_EXPORT_LEDGER") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_EXPORT_LEDGER=1 with pinned assets/checkpoint and evidence directory")
	}
	checkpoint := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_CPU_MODEL", "d0d366f789a6ab0078bd0f1534d16a38dddb4e08a3193797be23c9f6b1794e5c", 1431217112)
	lm, flow, w, cfg := releasedTrainingModels(t)
	tr, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	state, err := LoadFullTrainingStateBinary(checkpoint, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	backend := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
	if err := backend.restoreSharedOwnedState(tr, &state); err != nil {
		t.Fatal(err)
	}
	if len(state.Params) != 0 || !sameResidentLiveBoundary(backend.host, tr) {
		t.Fatal("owned restore association")
	}
	snapshot, err := sharedInitialSnapshot(backend.host)
	if err != nil {
		t.Fatal(err)
	}
	native, err := train.SharedTapeResidentBytes(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if native.Bytes != 1822745688 {
		t.Fatal("resident inventory", native.Bytes)
	}
	batch := FlowLMTrainingBatch{Frames: 14, VoiceFrames: 13, TextTokens: []uint32{1}, NormalizedLatents: make([]float32, 14*lm.LatentDim), VoiceLatents: make([]float32, 13*lm.LatentDim)}
	workspace, err := NewTrainingWorkspace(lm, flow, batch)
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.Open(os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	type phase struct {
		Name                                                   string
		RetainedF32, HeapAlloc, HeapInuse, HeapSys, TotalAlloc uint64
		NumGC                                                  uint32
	}
	report := struct {
		Scope                                                      string
		NativeResidentBytes, MappingBytes, HostCap, ExportAddedF32 int64
		FlowTensors, MimiTensors                                   int
		Phases                                                     []phase
		PayloadForecast, HeapAllocForecastBeforeGC                 int64
	}{Scope: "CPU-only owned restore/workspace/export tensor selection; native payload forecast excludes padding/driver/kernel/GC/metadata, no export writer/publication.", NativeResidentBytes: native.Bytes, MappingBytes: 219029196, HostCap: 6 << 30}
	sample := func(name string, payload int64) {
		var mem runtime.MemStats
		runtime.ReadMemStats(&mem)
		report.Phases = append(report.Phases, phase{Name: name, RetainedF32: uint64(payload), HeapAlloc: mem.HeapAlloc, HeapInuse: mem.HeapInuse, HeapSys: mem.HeapSys, TotalAlloc: mem.TotalAlloc, NumGC: mem.NumGC})
		t.Logf("phase=%s retainedF32=%d HeapAlloc=%d HeapInuse=%d HeapSys=%d TotalAlloc=%d NumGC=%d", name, payload, mem.HeapAlloc, mem.HeapInuse, mem.HeapSys, mem.TotalAlloc, mem.NumGC)
	}
	base := sharedHostF32Payload(tr, backend, workspace, batch, snapshot)
	sample("completed-owned-boundary-workspace", base)
	exports, err := pocketFlowExportTensors(cfg, tr, true)
	if err != nil {
		t.Fatal(err)
	}
	report.FlowTensors = len(exports)
	flowPayload := sharedHostF32Payload(tr, backend, workspace, batch, snapshot, exports)
	if flowPayload != base {
		t.Fatal("EMA flow tensor selection duplicated payload", flowPayload, base)
	}
	sample("EMA-flow-selection", flowPayload)
	mimi, err := pocketMimiExportTensors(cfg, source)
	if err != nil {
		t.Fatal(err)
	}
	report.MimiTensors = len(mimi)
	if len(exports)+len(mimi) != 214 {
		t.Fatal("export tensor inventory")
	}
	payload := sharedHostF32Payload(tr, backend, workspace, batch, snapshot, exports, mimi)
	report.ExportAddedF32 = payload - base
	if report.ExportAddedF32 <= 0 {
		t.Fatal("missing frozen decode accounting")
	}
	sample("frozen-Mimi-materialized-before-GC", payload)
	report.PayloadForecast = payload + native.Bytes + report.MappingBytes
	report.HeapAllocForecastBeforeGC = int64(report.Phases[len(report.Phases)-1].HeapAlloc) + native.Bytes + report.MappingBytes
	// Attribution-only GC after the measured exporter coexistence. Do not add
	// production GC or lower admission using this post-GC sample.
	runtime.GC()
	sample("after-attribution-GC", sharedHostF32Payload(tr, backend, workspace, batch, snapshot, exports, mimi))
	t.Logf("export coexistence payloadForecast=%d payloadHeadroom=%d HeapAllocForecastBeforeGC=%d; native padding/opaque plans, header/writer, deadGC/RSS unqualified", report.PayloadForecast, report.HostCap-report.PayloadForecast, report.HeapAllocForecastBeforeGC)
	root := os.Getenv("GO_PHERENCE_POCKET_SHARED_EXPORT_OUTPUT")
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(root, "export-coexistence.json"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, we := f.Write(append(data, '\n'))
	ce := f.Close()
	if we != nil || ce != nil {
		t.Fatal(we, ce)
	}
	runtime.KeepAlive(tr)
	runtime.KeepAlive(backend)
	runtime.KeepAlive(workspace)
	runtime.KeepAlive(batch)
	runtime.KeepAlive(snapshot)
	runtime.KeepAlive(exports)
	runtime.KeepAlive(mimi)
	runtime.KeepAlive(source)
}
