package pockettts

import (
	"encoding/json"
	"os"
	"sort"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Host-only complete-source/physical-slot audit. Model assets explicitly pinned;
// never initialises a runtime or allocates native buffers. No graph execution.
func TestPocketSharedReleasedWorkingSet(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_RESOURCE") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_RESOURCE=1 with pinned released model and prepared corpus")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_RESOURCE_FORWARD_ONLY") == "1" {
		lm.EOSBackend = &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
	}
	raw, e := os.ReadFile(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"))
	if e != nil {
		t.Fatal(e)
	}
	if len(raw) > 1<<20 {
		t.Fatal("corpus ceiling")
	}
	var corpus struct {
		Rows []PreparedTrainingRow `json:"rows"`
	}
	if e = json.Unmarshal(raw, &corpus); e != nil || len(corpus.Rows) != 1 {
		t.Fatal("corpus", e)
	}
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_RESOURCE_LONG") == "1" {
		corpus.Rows[0] = integratedLongSyntheticRow(corpus.Rows[0])
	}
	batch := corpus.Rows[0].Batch
	mask := make([]bool, batch.Frames)
	for i := 0; i < corpus.Rows[0].ValidFrames; i++ {
		mask[i] = true
	}
	cap := int64(2 << 30)
	p, e := lm.buildEOSForwardTrainingTape(batch, mask, .1, cap)
	if e != nil {
		t.Fatal(e)
	}
	p, e = buildForwardResultsTape(p, lm, batch, cap)
	if e != nil {
		t.Fatal(e)
	}
	roots := forwardResultsOutputs(p)
	p, e = appendForwardEOSAdmission(p, lm, batch, cap)
	if e != nil {
		t.Fatal(e)
	}
	roots = append(roots, p.Tape.Outputs[len(p.Tape.Outputs)-1])
	parentBytes := deviceAuditTapeBytes(t, p.Tape)
	resource := func(name string, tape train.Tape, b map[string]train.ResidentTapeBinding, roots []int) int64 {
		source := deviceAuditTapeBytes(t, tape)
		if e := train.ValidateResidentTapeBindings(tape, b, cap); e != nil {
			t.Fatal(name, e)
		}
		// Match runtime compact rewrite before source admission/storage planning.
		tape, b, compact, e := compactResidentEmbeddingTape(tape, b, cap)
		if e != nil {
			t.Fatal(name, e)
		}
		if compact != nil {
			if e = train.ValidateResidentCompactTapeBindings(tape, b, compact, cap); e != nil {
				t.Fatal(e)
			}
		} else if e = train.ValidateResidentTapeBindings(tape, b, cap); e != nil {
			t.Fatal(e)
		}
		source = deviceAuditTapeBytes(t, tape)
		closure := append([]int(nil), roots...)
		for n, v := range b {
			closure = append(closure, v.Gradient)
			if c, ok := compact[n]; ok {
				closure = append(closure, c.Index)
			}
		}
		pruned, mapping, e := train.PruneTape(tape, closure, cap)
		if e != nil {
			t.Fatal(name, e)
		}
		protect := map[int]bool{}
		for _, v := range b {
			if mapping[v.Parameter] >= 0 {
				protect[mapping[v.Parameter]] = true
			}
		}
		for i, s := range tape.Tensors {
			if s.External && mapping[i] >= 0 {
				protect[mapping[i]] = true
			}
		}
		slots, e := train.PlanZeroTapeStorage(pruned, cap, protect)
		if e != nil {
			t.Fatal(name, e)
		}
		borrowedSlots := map[int]bool{}
		for id := range protect {
			borrowedSlots[slots.Slots[id]] = true
		}
		var owned int64
		for id, n := range slots.Elements {
			if !borrowedSlots[id] {
				owned += 4 * int64(n)
			}
		}
		t.Logf("%s tensors=%d commands=%d fullsource=%d ownedPlanned=%d (metadata/allocator/GC excluded)", name, len(tape.Tensors), len(tape.Commands), source, owned)
		return owned
	}
	// Producer has no accumulation: do not retain dummy G for physical accounting.
	producerOwned := resourceForwardShared(t, p.Tape, p.ResidentBindings, roots, cap)
	h, n := lm.Hidden, lm.LatentDim
	condition := make([]float32, h)
	target := batch.NormalizedLatents[:n]
	noise := make([]float32, n)
	df, e := flow.buildDiagonalForwardTape(DiagonalForwardInputs{Condition: condition, Target: target, Noise: noise, Time: .3}, cap, true)
	if e != nil {
		t.Fatal(e)
	}
	db, e := flow.buildDiagonalDeviceTrainingTape(h, n, .3, 0, .75/float32(corpus.Rows[0].ValidFrames), cap)
	if e != nil {
		t.Fatal(e)
	}
	ff, e := flow.buildDistillForwardTape(DistillForwardInputs{Condition: condition, Target: target, Noise: noise, S: .3, T: .8}, cap, true)
	if e != nil {
		t.Fatal(e)
	}
	fb, e := flow.buildDistillDeviceTrainingTape(h, n, .3, .8, 0, target, noise, cap)
	if e != nil {
		t.Fatal(e)
	}
	maxOwned := producerOwned
	for _, row := range []struct {
		name     string
		forward  train.Tape
		bindings map[string]train.ResidentTapeBinding
		count    int
		backward train.Tape
		bb       map[string]train.ResidentTapeBinding
	}{{"diagonal", df.Tape, df.ResidentBindings, 3, db.tape, db.bindings}, {"distill", ff.Tape, ff.ResidentBindings, 6, fb.tape, fb.bindings}} {
		live := parentBytes + deviceAuditTapeBytes(t, row.forward) + deviceAuditTapeBytes(t, row.backward) + int64(4*(batch.Frames*h+h+1))
		if live > cap {
			t.Fatal("first complete-live source rejection", row.name, live, cap)
		}
		forwardOwned := resourceForwardShared(t, row.forward, row.bindings, row.forward.Outputs[:row.count], cap)
		backwardOwned := resource(row.name+" backward", row.backward, row.bb, row.backward.Outputs)
		totalOwned := producerOwned + int64(h*4) + forwardOwned + backwardOwned
		maxOwned = max(maxOwned, totalOwned)
		t.Logf("%s complete-live-source=%d ownedSlotsCoexistence=%d", row.name, live, totalOwned)
	}
	in := StepReductionInputs{Frames: batch.Frames, Hidden: h, Latent: n, Mask: mask, DiagonalCondition: make([]float32, batch.Frames*h), DistillCondition: make([]float32, batch.Frames*h), DiagonalTarget: make([]float32, batch.Frames*n), DistillTarget: make([]float32, batch.Frames*n), DiagonalLogvar: make([]float32, batch.Frames), DistillLogvar: make([]float32, batch.Frames), RowMetrics: make([]float32, batch.Frames*4), EOS: .1}
	s := WeightSampleInputs{DiagonalTime: make([]float32, batch.Frames), DistillS: make([]float32, batch.Frames), DistillT: make([]float32, batch.Frames), DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	later, e := lm.buildStepBackboneTrainingTape(w, batch, make([]float32, batch.Frames*h), in, DefaultTrainingStepConfig(), s, cap, true)
	if e != nil {
		t.Fatal(e)
	}
	later, e = buildStepResultsTape(later, batch, in, cap)
	if e != nil {
		t.Fatal(e)
	}
	monolithic := resource("later", later.Tape, later.ResidentBindings, stepResultsOutputs(later, batch))
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_RESOURCE_GROUPED") == "1" {
		maxOwned = max(maxOwned, resourceGroupedBackbone(t, later.Tape, later.ResidentBindings, stepResultsOutputs(later, batch), cap))
	} else {
		maxOwned = max(maxOwned, monolithic)
	}
	params, e := fullParameterMap(lm, flow, w)
	if e != nil {
		t.Fatal(e)
	}
	var elements int64
	var maxN, maxStats int
	for _, v := range params {
		elements += int64(len(v))
		maxN = max(maxN, len(v))
		ns, _ := train.NormStatsShape(len(v))
		maxStats = max(maxStats, ns)
	}
	resident := 20*elements + 12*int64(maxN) + 4*int64(maxStats) + 48
	if resident > cap {
		t.Fatal("resident rejection", resident)
	}
	t.Logf("released shared resident=%d maxOwnedTapes=%d predictedPhysicalPayload=%d cap=%d; runtime/allocator/plan overhead needs native measurement", resident, maxOwned, resident+maxOwned, cap)
}
func resourceForwardShared(t *testing.T, tape train.Tape, b map[string]train.ResidentTapeBinding, roots []int, cap int64) int64 {
	t.Helper()
	pruned, mapping, e := train.PruneTape(tape, roots, cap)
	if e != nil {
		t.Fatal(e)
	}
	protect := map[int]bool{}
	for _, v := range b {
		if mapping[v.Parameter] >= 0 {
			protect[mapping[v.Parameter]] = true
		}
	}
	for id, s := range tape.Tensors {
		if s.External && mapping[id] >= 0 {
			protect[mapping[id]] = true
		}
	}
	slots, e := train.PlanZeroTapeStorage(pruned, cap, protect)
	if e != nil {
		t.Fatal(e)
	}
	borrowed := map[int]bool{}
	for id := range protect {
		borrowed[slots.Slots[id]] = true
	}
	var owned int64
	for id, n := range slots.Elements {
		if !borrowed[id] {
			owned += 4 * int64(n)
		}
	}
	t.Logf("forward fullsource=%d ownedPlanned=%d tensors%d commands%d", deviceAuditTapeBytes(t, tape), owned, len(tape.Tensors), len(tape.Commands))
	return owned
}

func resourceGroupedBackbone(t *testing.T, source train.Tape, bindings map[string]train.ResidentTapeBinding, outputs []int, cap int64) int64 {
	t.Helper()
	tape, bindings, compact, err := compactResidentEmbeddingTape(source, bindings, cap)
	if err != nil {
		t.Fatal(err)
	}
	if compact == nil {
		err = train.ValidateResidentTapeBindings(tape, bindings, cap)
	} else {
		err = train.ValidateResidentCompactTapeBindings(tape, bindings, compact, cap)
	}
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(bindings))
	for name := range bindings {
		names = append(names, name)
	}
	sort.Strings(names)
	groupSize := stepBackboneGradientGroupSize(tape)
	var maxOwned int64
	for start := 0; start < len(names); start += groupSize {
		var roots []int
		if start == 0 {
			roots = append(roots, outputs...)
		}
		for _, name := range names[start:min(start+groupSize, len(names))] {
			roots = append(roots, bindings[name].Gradient)
			if c, ok := compact[name]; ok {
				roots = append(roots, c.Index)
			}
		}
		pruned, mapping, err := train.PruneTape(tape, roots, cap)
		if err != nil {
			t.Fatal(err)
		}
		protect := map[int]bool{}
		for _, b := range bindings {
			if mapping[b.Parameter] >= 0 {
				protect[mapping[b.Parameter]] = true
			}
		}
		for id, s := range tape.Tensors {
			if s.External && mapping[id] >= 0 {
				protect[mapping[id]] = true
			}
		}
		slots, err := train.PlanZeroTapeStorage(pruned, cap, protect)
		if err != nil {
			t.Fatal(err)
		}
		borrowed := map[int]bool{}
		for id := range protect {
			borrowed[slots.Slots[id]] = true
		}
		var owned int64
		for id, n := range slots.Elements {
			if !borrowed[id] {
				owned += int64(n) * 4
			}
		}
		maxOwned = max(maxOwned, owned)
		t.Logf("group=%d names=%d owned=%d commands=%d full-source=%d", start/groupSize, min(groupSize, len(names)-start), owned, len(pruned.Commands), deviceAuditTapeBytes(t, tape))
	}
	t.Logf("grouped later maxOwned=%d groups=%d", maxOwned, (len(names)+groupSize-1)/groupSize)
	return maxOwned
}
