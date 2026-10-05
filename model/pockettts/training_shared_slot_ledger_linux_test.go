//go:build linux

package pockettts

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"sort"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// A read-only attribution pass for the released later/backbone storage plan.
// It neither executes graph arithmetic nor allocates a native owner. Pinned
// zero payload and retained gradient outputs are accounted separately before
// considering any new binding/lifetime contract.
func TestPocketSharedReleasedLaterSlotLedger(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_SLOT_LEDGER") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_SLOT_LEDGER=1 with pinned model/corpus")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	raw, err := os.ReadFile(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"))
	if err != nil || len(raw) > 1<<20 {
		t.Fatal("corpus ceiling/read", err)
	}
	hash := sha256.Sum256(raw)
	if hex.EncodeToString(hash[:]) != "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9" {
		t.Fatal("corpus pin")
	}
	var corpus struct {
		Rows []PreparedTrainingRow `json:"rows"`
	}
	if err := json.Unmarshal(raw, &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal("corpus", err)
	}
	batch := corpus.Rows[0].Batch
	if batch.Frames != 14 || batch.VoiceFrames != 13 || len(batch.TextTokens) != 1 || corpus.Rows[0].ValidFrames != 13 {
		t.Fatal("released fixture shape")
	}
	mask := make([]bool, batch.Frames)
	for i := 0; i < corpus.Rows[0].ValidFrames; i++ {
		mask[i] = true
	}
	h, n := lm.Hidden, lm.LatentDim
	in := StepReductionInputs{Frames: batch.Frames, Hidden: h, Latent: n, Mask: mask, DiagonalCondition: make([]float32, batch.Frames*h), DistillCondition: make([]float32, batch.Frames*h), DiagonalTarget: make([]float32, batch.Frames*n), DistillTarget: make([]float32, batch.Frames*n), DiagonalLogvar: make([]float32, batch.Frames), DistillLogvar: make([]float32, batch.Frames), RowMetrics: make([]float32, batch.Frames*4), EOS: .1}
	samples := WeightSampleInputs{DiagonalTime: make([]float32, batch.Frames), DistillS: make([]float32, batch.Frames), DistillT: make([]float32, batch.Frames), DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	const capBytes = int64(2 << 30)
	later, err := lm.buildStepBackboneTrainingTape(w, batch, make([]float32, batch.Frames*h), in, DefaultTrainingStepConfig(), samples, capBytes, true)
	if err != nil {
		t.Fatal(err)
	}
	later, err = buildStepResultsTape(later, batch, in, capBytes)
	if err != nil {
		t.Fatal(err)
	}
	if err := train.ValidateResidentTapeBindings(later.Tape, later.ResidentBindings, capBytes); err != nil {
		t.Fatal(err)
	}
	tape, bindings, compact, err := compactResidentEmbeddingTape(later.Tape, later.ResidentBindings, capBytes)
	if err != nil {
		t.Fatal(err)
	}
	roots := append([]int(nil), stepResultsOutputs(later, batch)...)
	for name, b := range bindings {
		roots = append(roots, b.Gradient)
		if c, ok := compact[name]; ok {
			roots = append(roots, c.Index)
		}
	}
	pruned, mapping, err := train.PruneTape(tape, roots, capBytes)
	if err != nil {
		t.Fatal(err)
	}
	protected := map[int]bool{}
	for _, b := range bindings {
		if mapping[b.Parameter] >= 0 {
			protected[mapping[b.Parameter]] = true
		}
	}
	for id, s := range tape.Tensors {
		if s.External && mapping[id] >= 0 {
			protected[mapping[id]] = true
		}
	}
	plan, err := train.PlanZeroTapeStorage(pruned, capBytes, protected)
	if err != nil {
		t.Fatal(err)
	}
	type slotRow struct {
		Slot, Elements     int
		Category, Producer string
		TensorIDs          []int
	}
	rows := make([]slotRow, len(plan.Elements))
	for id, n := range plan.Elements {
		rows[id] = slotRow{Slot: id, Elements: n, Category: "transient", Producer: "unwritten"}
	}
	for id := range protected {
		rows[plan.Slots[id]].Category = "borrowed"
	}
	for id, s := range pruned.Tensors {
		r := &rows[plan.Slots[id]]
		r.TensorIDs = append(r.TensorIDs, id)
		if len(s.Values) > 0 && r.Category != "borrowed" {
			zero := true
			for _, v := range s.Values {
				if v != 0 {
					zero = false
					break
				}
			}
			if zero {
				r.Category = "constant_zero"
			} else {
				r.Category = "constant_nonzero"
			}
		}
	}
	gradientNames := map[int]string{}
	for name, b := range bindings {
		if mapping[b.Gradient] >= 0 {
			gradientNames[mapping[b.Gradient]] = name
		}
	}
	for _, c := range pruned.Commands {
		r := &rows[plan.Slots[c.Output]]
		if c.Op != "reshape" {
			r.Producer = c.Op
		}
		if name, ok := gradientNames[c.Output]; ok && r.Category == "transient" {
			r.Category = "gradient_output"
			r.Producer = name
		}
	}
	for _, id := range pruned.Outputs {
		r := &rows[plan.Slots[id]]
		if name, ok := gradientNames[id]; ok && r.Category == "transient" {
			r.Category = "gradient_output"
			r.Producer = name
		}
		if r.Category == "transient" {
			r.Category = "boundary_output"
		}
	}
	var owned int64
	categories := map[string]int64{}
	for _, r := range rows {
		bytes := 4 * int64(r.Elements)
		categories[r.Category] += bytes
		if r.Category != "borrowed" {
			owned += bytes
		}
	}
	if owned != 317970992 {
		t.Fatalf("slot baseline changed: %d", owned)
	}
	params, err := fullParameterMap(lm, flow, w)
	if err != nil {
		t.Fatal(err)
	}
	var elements int64
	maxN, maxStats := 0, 0
	for _, v := range params {
		elements += int64(len(v))
		maxN = max(maxN, len(v))
		stats, _ := train.NormStatsShape(len(v))
		maxStats = max(maxStats, stats)
	}
	resident := 20*elements + 12*int64(maxN) + 4*int64(maxStats) + 48
	t.Logf("baseline resident=%d owned=%d total=%d cap=%d excess=%d", resident, owned, resident+owned, capBytes, resident+owned-capBytes)
	tapeResident := resident - 4*int64(maxN)
	if output := os.Getenv("GO_PHERENCE_POCKET_SHARED_SLOT_INVENTORY"); output != "" {
		var residentAllocations, tapeAllocations []int64
		names := make([]string, 0, len(params))
		for name := range params {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			for i := 0; i < 5; i++ {
				residentAllocations = append(residentAllocations, 4*int64(len(params[name])))
			}
		}
		residentAllocations = append(residentAllocations, 4*int64(maxN), 4*int64(maxN), 4*int64(maxStats), 16, 16, 16)
		for _, r := range rows {
			if r.Category != "borrowed" {
				tapeAllocations = append(tapeAllocations, 4*int64(r.Elements))
			}
		}
		inventory := struct {
			Scope                  string
			Resident, Tape         []int64
			Tensors, Stages, Plans int
			Operations             map[string]int
		}{Scope: "Payload extents and non-reshape dispatch counts only; no native allocation/padding bound.", Resident: residentAllocations, Tape: tapeAllocations, Tensors: len(pruned.Tensors), Operations: map[string]int{}}
		for _, c := range pruned.Commands {
			if c.Op != "reshape" {
				inventory.Stages++
				inventory.Operations[c.Op]++
			}
		}
		inventory.Plans = (inventory.Stages + 63) / 64
		data, err := json.MarshalIndent(inventory, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		file, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		_, we := file.Write(append(data, '\n'))
		ce := file.Close()
		if we != nil || ce != nil {
			t.Fatal(we, ce)
		}
	}
	t.Logf("tape-only resident=%d owned=%d total=%d payload_headroom=%d; native overhead unqualified", tapeResident, owned, tapeResident+owned, capBytes-tapeResident-owned)
	owner := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
	for name := range params {
		if !owner.ownsResidentGradient(name) {
			t.Fatal("released tape-only unowned gradient", name)
		}
	}
	keys := make([]string, 0, len(categories))
	for key := range categories {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		t.Logf("category=%s bytes=%d", key, categories[key])
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Elements != rows[j].Elements {
			return rows[i].Elements > rows[j].Elements
		}
		return rows[i].Slot < rows[j].Slot
	})
	count := 0
	for _, r := range rows {
		if r.Category == "borrowed" {
			continue
		}
		t.Logf("owned slot=%d bytes=%d category=%s producer=%s logicalAliases=%d", r.Slot, 4*int64(r.Elements), r.Category, r.Producer, len(r.TensorIDs))
		count++
		if count == 24 {
			break
		}
	}
	// A hypothetical reuse forecast is explicitly excluded from admission:
	// overlapping read-only views require a separate native capability contract.
	t.Logf("diagnostic zero-reuse forecast=%d (unimplemented; cannot admit/execute)", resident+owned-categories["constant_zero"])
}
