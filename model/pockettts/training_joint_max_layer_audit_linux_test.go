//go:build linux

package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"os"
	"sort"
	"testing"
)

// Design/storage evidence only. Stages preserve original parameter names and
// full-model preflight; no native work or runner orchestration occurs here.
func TestPocketReleasedJointMaximumLayerAudit(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINT_MAX_LAYER_AUDIT") != "1" {
		t.Skip("explicit joint maximum layer storage audit")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	params, err := fullParameterMap(lm, flow, w)
	if err != nil {
		t.Fatal(err)
	}
	host := FullTrainingState{}
	for name, values := range params {
		host.Params = append(host.Params, NamedTrainingTensor{Name: name, Values: values})
	}
	owner := &ResidentTrainingBackend{host: host, MaxResidentBytes: 2 << 30, MaxTapeBytes: 2 << 30}
	const rows = 950
	const capBytes = int64(2 << 30)
	sequence, seed := make([]float32, rows*lm.Hidden), make([]float32, rows*lm.Hidden)
	// Zero-filled activation/seed constants can share immutable slots and are
	// not an upper bound for the actual executed training graph.
	for i := range sequence {
		sequence[i] = .01 * float32(1+i%11)
		seed[i] = .001 * float32(1+i%7)
	}
	var maxOwned int64
	for i, layer := range lm.Transformer.Layers {
		stage := *lm.Transformer
		stage.Layers = []TransformerLayerCPU{layer}
		stage.FinalWeight = nil
		stage.FinalBias = nil
		// Final affine norm will be a separately admitted/recomputed stage.
		var reverse TransformerTrainingTape
		var err error
		if os.Getenv("GO_PHERENCE_POCKET_JOINT_TRANSIENT_ZEROS") == "1" {
			reverse, err = buildTransformerLayerStage(lm.Transformer, i, sequence, seed, rows, false, capBytes)
		} else {
			reverse, err = stage.buildTransformerTrainingTape(sequence, seed, rows, capBytes, true)
		}
		if err != nil {
			t.Fatal("layer complete source", i, err)
		}
		if err = owner.admitStagedStorage(reverse.Tape, reverse.ResidentBindings, reverse.Tape.Outputs[1:2], true); err != nil {
			t.Fatal("production stage joint admission", i, err)
		}
		// Match the admitted grouped runtime root policy, charging the complete
		// one-layer source each time. Parameter owners stay distinct by inventory.
		// Forward output was produced in the preceding stage. Reverse needs
		// only the input adjoint, not another retained forward boundary.
		boundaryOwned := resourceForwardShared(t, reverse.Tape, reverse.ResidentBindings, reverse.Tape.Outputs[1:2], capBytes)
		gradientOwned := resourceGroupedBackbone(t, reverse.Tape, reverse.ResidentBindings, nil, capBytes)
		owned := max(boundaryOwned, gradientOwned)
		t.Logf("layer=%d separate reverse boundary owned=%d gradientGroups owned=%d", i, boundaryOwned, gradientOwned)
		maxOwned = max(maxOwned, owned)
		if i == 0 {
			attributeJointLayerSlots(t, reverse)
		}
		forward, err := buildTransformerLayerStage(lm.Transformer, i, sequence, seed, rows, true, capBytes)
		if err != nil {
			t.Fatal("layer forward source", i, err)
		}
		if err = owner.admitStagedStorage(forward.Tape, forward.ResidentBindings, forward.Tape.Outputs[:1], false); err != nil {
			t.Fatal("production forward joint admission", i, err)
		}
		forwardOwned := resourceForwardShared(t, forward.Tape, forward.ResidentBindings, forward.Tape.Outputs[:1], capBytes)
		maxOwned = max(maxOwned, forwardOwned)
		t.Logf("layer=%d original names forward source=%d forward owned=%d reverse source=%d grouped owned=%d", i, deviceAuditTapeBytes(t, forward.Tape), forwardOwned, deviceAuditTapeBytes(t, reverse.Tape), owned)
		if err = forward.Tape.Validate(capBytes); err != nil {
			t.Fatal(err)
		}
	}
	// The final affine norm has a separately admitted source, not omitted work.
	built, err := buildTransformerFinalNormStage(lm.Transformer, sequence, seed, rows, false, capBytes)
	if err != nil {
		t.Fatal("final norm source", err)
	}
	if err = owner.admitStagedStorage(built.Tape, built.ResidentBindings, built.Tape.Outputs[1:2], true); err != nil {
		t.Fatal("production final norm admission", err)
	}
	normOwned := resourceGroupedBackbone(t, built.Tape, built.ResidentBindings, built.Tape.Outputs[1:2], capBytes)
	maxOwned = max(maxOwned, normOwned)
	t.Logf("separate final norm source=%d grouped maxOwned=%d", deviceAuditTapeBytes(t, built.Tape), normOwned)
	const tapeOnlyResident = int64(1822745808)
	t.Logf("joint layer design resident=%d maxOwned=%d sum=%d margin2GiB=%d host boundary rows950 width%d layers%d bytes=%d", tapeOnlyResident, maxOwned, tapeOnlyResident+maxOwned, capBytes-tapeOnlyResident-maxOwned, lm.Hidden, len(lm.Transformer.Layers), (len(lm.Transformer.Layers)+1)*rows*lm.Hidden*4)
	if tapeOnlyResident+maxOwned >= capBytes {
		t.Fatal("layer grouping still exceeds joint native payload")
	}
}

func attributeJointLayerSlots(t *testing.T, stage TransformerTrainingTape) {
	t.Helper()
	names := append([]string(nil), stage.ParameterNames...)
	sort.Strings(names)
	for _, name := range names {
		roots := append([]int(nil), stage.Tape.Outputs[:2]...)
		roots = append(roots, stage.ResidentBindings[name].Gradient)
		tape, mapping, err := train.PruneTape(stage.Tape, roots, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		protected := map[int]bool{}
		for _, b := range stage.ResidentBindings {
			if mapping[b.Parameter] >= 0 {
				protected[mapping[b.Parameter]] = true
			}
		}
		plan, err := train.PlanZeroTapeStorage(tape, 2<<30, protected)
		if err != nil {
			t.Fatal(err)
		}
		borrowed := map[int]bool{}
		for id := range protected {
			borrowed[plan.Slots[id]] = true
		}
		var owned int64
		for slot, n := range plan.Elements {
			if !borrowed[slot] {
				owned += int64(n) * 4
			}
		}
		if owned < 310<<20 {
			continue
		}
		producers := map[int]map[string]bool{}
		for id, s := range tape.Tensors {
			slot := plan.Slots[id]
			if producers[slot] == nil {
				producers[slot] = map[string]bool{}
			}
			if len(s.Values) > 0 {
				kind := "constant"
				if zeroTrainingSeed(s.Values) {
					kind = "zero-source"
				}
				producers[slot][kind] = true
			}
		}
		for _, c := range tape.Commands {
			if c.Op != "reshape" {
				producers[plan.Slots[c.Output]][c.Op] = true
			}
		}
		for slot, n := range plan.Elements {
			if borrowed[slot] || n*4 < 8<<20 {
				continue
			}
			kinds := make([]string, 0, len(producers[slot]))
			for kind := range producers[slot] {
				kinds = append(kinds, kind)
			}
			sort.Strings(kinds)
			t.Logf("joint owned root=%s slot=%d bytes=%d producers=%v", name, slot, n*4, kinds)
		}
	}
}
