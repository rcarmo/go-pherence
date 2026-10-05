//go:build linux

package pockettts

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"runtime"
	"runtime/pprof"
	"sort"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Test-only address intervals measure unique retained F32 backing storage,
// including capacity and borrowed views. Addresses are never dereferenced and
// objects remain live until the ledger is complete. This excludes metadata,
// runtime heap reservation, native memory, immutable mmap and GC dead objects.
func sharedHostF32Payload(roots ...any) int64 {
	type span struct{ start, end uintptr }
	var spans []span
	seen := map[uintptr]bool{}
	var visit func(reflect.Value)
	visit = func(v reflect.Value) {
		if !v.IsValid() {
			return
		}
		switch v.Kind() {
		case reflect.Interface:
			if !v.IsNil() {
				visit(v.Elem())
			}
		case reflect.Pointer:
			if v.IsNil() || seen[v.Pointer()] {
				return
			}
			seen[v.Pointer()] = true
			visit(v.Elem())
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				visit(v.Field(i))
			}
		case reflect.Map:
			iter := v.MapRange()
			for iter.Next() {
				visit(iter.Value())
			}
		case reflect.Slice:
			if v.IsNil() || v.Cap() == 0 {
				return
			}
			if v.Type().Elem().Kind() == reflect.Float32 {
				start := v.Pointer()
				spans = append(spans, span{start, start + uintptr(v.Cap())*4})
				return
			}
			for i := 0; i < v.Len(); i++ {
				visit(v.Index(i))
			}
		case reflect.Array:
			for i := 0; i < v.Len(); i++ {
				visit(v.Index(i))
			}
		}
	}
	for _, root := range roots {
		visit(reflect.ValueOf(root))
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
	var total int64
	var end uintptr
	for _, s := range spans {
		if s.end <= end {
			continue
		}
		start := s.start
		if start < end {
			start = end
		}
		total += int64(s.end - start)
		end = s.end
	}
	runtime.KeepAlive(roots)
	return total
}

func TestPocketSharedHostF32PayloadUnion(t *testing.T) {
	a := make([]float32, 20)
	b := make([]float32, 5)
	x := struct {
		Private []float32
		Aliases map[string][]float32
	}{a[3:7], map[string][]float32{"a": a, "view": a[1:2], "b": b}}
	if got := sharedHostF32Payload(&x, &x, a); got != 100 {
		t.Fatal("retained capacity union", got)
	}
}

func TestPocketSharedReleasedHostPayloadLedger(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_HOST_LEDGER") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_HOST_LEDGER=1 with pinned local model/corpus")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	path := os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS")
	info, err := os.Stat(path)
	if err != nil || info.Size() > 1<<20 {
		t.Fatal("corpus bound", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	if hex.EncodeToString(digest[:]) != "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9" {
		t.Fatal("corpus provenance")
	}
	var corpus struct {
		Rows []PreparedTrainingRow `json:"rows"`
	}
	if err := json.Unmarshal(raw, &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal("corpus", err)
	}
	batch := corpus.Rows[0].Batch
	if batch.Frames != 14 || batch.VoiceFrames != 13 || corpus.Rows[0].ValidFrames != 13 || len(batch.TextTokens) != 1 {
		t.Fatal("ledger row")
	}
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_HOST_LONG") == "1" {
		corpus.Rows[0] = integratedLongSyntheticRow(corpus.Rows[0])
		batch = corpus.Rows[0].Batch
	}
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
	backend := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1", host: host}
	backend.rememberSharedInitialMoments(snapshot)
	pooled := os.Getenv("GO_PHERENCE_POCKET_SHARED_POOLED_WORKSPACE") == "1"
	if pooled {
		flow.TrainingBackend = backend
		flow.LossBackend = backend
		flow.BranchBackend = backend
		flow.StepReductionBackend = backend
		lm.TransformerBackend = backend
		lm.ProjectionBackend = backend
		lm.SequenceBackend = backend
		lm.EOSBackend = backend
		w.TrainingBackend = backend
	}
	workspace, err := newTrainingWorkspace(lm, flow, batch, pooled)
	if err != nil {
		t.Fatal(err)
	}
	base := sharedHostF32Payload(tr, backend, workspace, corpus)
	params, err := fullParameterMap(lm, flow, w)
	if err != nil {
		t.Fatal(err)
	}
	var parameterBytes int64
	for _, x := range params {
		parameterBytes += int64(len(x)) * 4
	}
	// Build just the maximum later source to attribute concrete host Values.
	// It is admitted as a complete source and never run/pruned to bypass it.
	h, n := lm.Hidden, lm.LatentDim
	mask := make([]bool, batch.Frames)
	for i := 0; i < corpus.Rows[0].ValidFrames; i++ {
		mask[i] = true
	}
	in := StepReductionInputs{Frames: batch.Frames, Hidden: h, Latent: n, Mask: mask, DiagonalCondition: make([]float32, batch.Frames*h), DistillCondition: make([]float32, batch.Frames*h), DiagonalTarget: make([]float32, batch.Frames*n), DistillTarget: make([]float32, batch.Frames*n), DiagonalLogvar: make([]float32, batch.Frames), DistillLogvar: make([]float32, batch.Frames), RowMetrics: make([]float32, batch.Frames*4), EOS: .1}
	s := WeightSampleInputs{DiagonalTime: make([]float32, batch.Frames), DistillS: make([]float32, batch.Frames), DistillT: make([]float32, batch.Frames), DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	later, err := lm.buildStepBackboneTrainingTape(w, batch, make([]float32, batch.Frames*h), in, DefaultTrainingStepConfig(), s, 2<<30, true)
	if err != nil {
		t.Fatal(err)
	}
	later, err = buildStepResultsTape(later, batch, in, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	if err := train.ValidateResidentTapeBindings(later.Tape, later.ResidentBindings, 2<<30); err != nil {
		t.Fatal(err)
	}
	all := sharedHostF32Payload(tr, backend, workspace, corpus, in, s, later)
	var logical, values int64
	for _, tensor := range later.Tape.Tensors {
		count, err := tensor.Elements()
		if err != nil {
			t.Fatal(err)
		}
		logical += int64(count) * 4
		values += int64(cap(tensor.Values)) * 4
	}
	runtime.GC() // attribution only, never production admission/recovery
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	if path := os.Getenv("GO_PHERENCE_POCKET_SHARED_LIVE_PROFILE"); path != "" {
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if err = pprof.Lookup("heap").WriteTo(file, 0); err != nil {
			t.Fatal(err)
		}
		if err = file.Close(); err != nil {
			t.Fatal(err)
		}
	}
	const resident = 1839522904
	laterOwned := int64(317970992)
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_HOST_LONG") == "1" {
		laterOwned = resourceGroupedBackbone(t, later.Tape, later.ResidentBindings, stepResultsOutputs(later, batch), 2<<30)
	}
	const modelMapping = 219029196
	// Live moments are created at first successful load (2P), or already exist
	// on restored steps. Heap/runtime/native overhead is still unmeasured.
	forecast := all + 2*parameterBytes + resident + laterOwned + modelMapping
	t.Logf("unique retained F32: initial host/model/EMA/zeroM/V/workspace/corpus=%d later-sources-added=%d all=%d P=%d sourceLogical=%d sourceValuesWithAliases=%d", base, all-base, all, parameterBytes, logical, values)
	t.Logf("later completed/restored payload forecast=%d remaining6GiB=%d excludes heap/native allocator/runtime/plan overhead+deadGC; afterGC HeapAlloc=%d HeapInuse=%d HeapSys=%d", forecast, int64(6<<30)-forecast, mem.HeapAlloc, mem.HeapInuse, mem.HeapSys)
	runtime.KeepAlive(tr)
	runtime.KeepAlive(backend)
	runtime.KeepAlive(workspace)
	runtime.KeepAlive(later)
	runtime.KeepAlive(in)
	runtime.KeepAlive(s)
	runtime.KeepAlive(corpus)
}
