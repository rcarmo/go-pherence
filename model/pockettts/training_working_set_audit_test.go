package pockettts

import (
	"fmt"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Pure source/planner audit. No trainer, model execution, native driver or assets.
// Synthetic host backing shares zeros by extent; logical and owned slot payloads
// are reported separately. This does not measure allocator peak or backend scratch.
func TestTrainingWorkingSetPhaseAudit(t *testing.T) {
	cfg, flow, backing := releasedDeviceAuditFixture(t)
	zero := func(n int) []float32 {
		if backing[n] == nil {
			backing[n] = make([]float32, n)
		}
		return backing[n]
	}
	h, n, ff := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim, cfg.FlowLM.Transformer.DModel*cfg.FlowLM.Transformer.HiddenScale
	linear := func(in, out int) LinearF32 { return LinearF32{In: in, Out: out, Weight: zero(in * out)} }
	tr := &TransformerCPU{Width: h, Heads: cfg.FlowLM.Transformer.NumHeads, HeadDim: h / cfg.FlowLM.Transformer.NumHeads, MaxPeriod: cfg.FlowLM.Transformer.MaxPeriod, Layers: make([]TransformerLayerCPU, cfg.FlowLM.Transformer.NumLayers), FinalWeight: zero(h), FinalBias: zero(h)}
	for i := range tr.Layers {
		tr.Layers[i] = TransformerLayerCPU{Norm1Weight: zero(h), Norm1Bias: zero(h), Norm2Weight: zero(h), Norm2Bias: zero(h), InProjection: linear(h, 3*h), OutProjection: linear(h, h), FC1: linear(h, ff), FC2: linear(ff, h)}
	}
	lm := &FlowLMTrainingCPU{Transformer: tr, Embedding: zero((cfg.FlowLM.LookupTable.NBins + 1) * h), BOS: zero(n), BOSBeforeVoice: zero(h), SpeakerProjection: linear(n, h), Input: linear(n, h), EOS: linear(h, 1), Hidden: h, LatentDim: n, Vocabulary: cfg.FlowLM.LookupTable.NBins + 1}
	lm.EOS.Bias = zero(1)
	inventory := releasedDeviceAuditInventory(t, cfg, flow)
	var elements, flowElements int64
	maxN, maxStats := 0, 0
	for name, size := range inventory {
		elements += int64(size)
		if len(name) >= len("flow_lm.flow_net.") && name[:len("flow_lm.flow_net.")] == "flow_lm.flow_net." {
			flowElements += int64(size)
		}
		maxN = max(maxN, size)
		stats, _ := train.NormStatsShape(size)
		maxStats = max(maxStats, stats)
	}
	resident := 20*elements + 12*int64(maxN) + 4*int64(maxStats) + 48
	// Include the released trainable pad-token embedding row, as in the
	// inventory mapping test and actual released-model loader.
	if elements != 89449730 || resident != 1839522904 {
		t.Fatal("inventory drift", elements, resident)
	}
	t.Logf("inventory E=%d flowE=%d sharedResident=%d denseHostGradient=%d secondFlowGradient=%d hostLiveP/M/V/EMA=%d hostBoundary=%d hostTransientSnapshot=%d threeHostSets=%d threeHostSetsPlusGradients=%d", elements, flowElements, resident, 4*elements, 4*flowElements, 16*elements, 16*elements, 16*elements, 48*elements, 52*elements+4*flowElements)
	// The complete source charge remains the acceptance contract. Borrowed physical
	// slots are subtracted exactly once per tape plan, never from admission.
	account := func(label string, tape train.Tape, b map[string]train.ResidentTapeBinding, outputs []int, external []int) (int64, int64) {
		t.Helper()
		if err := tape.Validate(2 << 30); err != nil {
			t.Fatal(label, err)
		}
		logical, _ := auditLogicalBytes(tape)
		protected := map[int]bool{}
		borrowed := map[int]bool{}
		for _, v := range b {
			protected[v.Parameter] = true
			borrowed[v.Parameter] = true
		}
		for _, id := range external {
			borrowed[id] = true
		}
		pruned, remap, err := train.PruneTape(tape, outputs, 2<<30)
		if err != nil {
			t.Fatal(label, err)
		}
		pm, bm := map[int]bool{}, map[int]bool{}
		for id := range protected {
			if mapped := remap[id]; mapped >= 0 {
				pm[mapped] = true
			}
		}
		for id := range borrowed {
			if mapped := remap[id]; mapped >= 0 {
				bm[mapped] = true
			}
		}
		for id, s := range pruned.Tensors {
			if s.External {
				bm[id] = true
			}
		}
		plan, err := train.PlanZeroTapeStorage(pruned, 2<<30, pm)
		if err != nil {
			t.Fatal(label, err)
		}
		owned := plan.Bytes
		slots := map[int]bool{}
		for id := range bm {
			slot := plan.Slots[id]
			if !slots[slot] {
				owned -= 4 * int64(plan.Elements[slot])
				slots[slot] = true
			}
		}
		if owned < 0 {
			t.Fatal("negative owned")
		}
		prunedBytes, _ := auditLogicalBytes(pruned)
		t.Logf("tape=%s fullLogical=%d prunedLogical=%d dedupedOwnedPlan=%d tensors=%d commands=%d", label, logical, prunedBytes, owned, len(tape.Tensors), len(tape.Commands))
		return logical, owned
	}
	for _, g := range []struct{ frames, tokens, voice int }{{1, 0, 0}, {1, 1, 0}, {7, 3, 2}, {32, 8, 4}, {128, 32, 8}, {256, 32, 8}} {
		label := fmt.Sprintf("f%d-t%d-v%d", g.frames, g.tokens, g.voice)
		rows := 1 + g.frames + g.tokens + g.voice
		batch := FlowLMTrainingBatch{Frames: g.frames, VoiceFrames: g.voice, TextTokens: make([]uint32, g.tokens), VoiceLatents: make([]float32, g.voice*n), NormalizedLatents: make([]float32, g.frames*n)}
		mask := make([]bool, g.frames)
		for i := range mask {
			mask[i] = true
		}
		wantErr := ""
		switch {
		case g.tokens == 0:
			wantErr = "invalid Pocket TTS FlowLM training boundary"
		}
		parent, err := lm.buildEOSForwardTrainingTape(batch, mask, 1, 2<<30)
		if wantErr != "" {
			if err == nil || err.Error() != wantErr {
				t.Fatal("parent boundary rejection", label, err)
			}
			t.Logf("geometry=%s parent admission rejected: %v", label, err)
			continue
		}
		if err != nil {
			t.Fatal("admitted geometry", label, err)
		}
		parent, err = buildForwardResultsTape(parent, lm, batch, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		// Production selects positional hidden/loss/witness roots before appending
		// the EOS flag; recomputing afterwards shifts the positions.
		parentOutputs := forwardResultsOutputs(parent)
		originalOutputs := append([]int(nil), parent.Tape.Outputs...)
		parent, err = appendForwardEOSAdmission(parent, lm, batch, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		if len(parent.Tape.Outputs) != len(originalOutputs)+1 {
			t.Fatal("EOS append output count")
		}
		for i, id := range originalOutputs {
			if parent.Tape.Outputs[i] != id {
				t.Fatal("EOS append changed source output IDs")
			}
		}
		parentOutputs = append(parentOutputs, parent.Tape.Outputs[len(parent.Tape.Outputs)-1])
		assertAuditRoots(t, parent.Tape, parentOutputs, []int{g.frames * h, 1, 1, 1})
		writers := map[int]train.Command{}
		for _, c := range parent.Tape.Commands {
			writers[c.Output] = c
		}
		for i, expected := range []struct {
			op   string
			mode uint32
		}{{"vector", 1}, {"eos", 2}, {"vector", 0}, {"eos-admission", 0}} {
			c := writers[parentOutputs[i]]
			if c.Op != expected.op || c.Mode != expected.mode {
				t.Fatal("ordered parent root producer", i, c)
			}
		}
		t.Logf("parent ordered hidden/loss/witness/flag roots=%v extents=%v", parentOutputs, []int{g.frames * h, 1, 1, 1})
		pl, po := account(label+"/parent", parent.Tape, parent.ResidentBindings, parentOutputs, nil)
		// Row condition is a read-only vector source; actual BeginChildTape preserves
		// the full n-row producer charge until the row's children close.
		condition := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{g.frames * h}, External: true}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{h}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Mode: 1, Offset: 0}}, Outputs: []int{2}}
		cl, co := account(label+"/condition", condition, nil, condition.Outputs, []int{0})
		for _, kind := range []string{"diagonal", "distill"} {
			var ft train.Tape
			var fb map[string]train.ResidentTapeBinding
			var bt train.Tape
			var bb map[string]train.ResidentTapeBinding
			var outputs, externals []int
			var guard train.Tape
			var guardExternal []int
			if kind == "diagonal" {
				f, err := flow.buildDiagonalForwardTape(DiagonalForwardInputs{Condition: make([]float32, h), Target: make([]float32, n), Noise: make([]float32, n), Time: .3}, 2<<30, true)
				if err != nil {
					t.Fatal(err)
				}
				ft, fb = f.Tape, f.ResidentBindings
				d, err := flow.buildDiagonalDeviceTrainingTape(h, n, .3, 0, .25, 2<<30)
				if err != nil {
					t.Fatal(err)
				}
				bt, bb = d.tape, d.bindings
				outputs = append([]int(nil), d.tape.Outputs[:3]...)
				for _, b := range bb {
					outputs = append(outputs, b.Gradient)
				}
				externals = []int{d.ids.Condition, d.ids.Input, d.ids.Desired, d.ids.Prediction}
				var ids diagonalDeviceIDs
				guard, ids, err = buildDiagonalDeviceAdmission(h, n, 0, 2<<30)
				if err != nil {
					t.Fatal(err)
				}
				guardExternal = []int{ids.Condition, ids.Input, ids.Desired, ids.Prediction}
			} else {
				f, err := flow.buildDistillForwardTape(DistillForwardInputs{Condition: make([]float32, h), Target: make([]float32, n), Noise: make([]float32, n), S: .3, T: .8}, 2<<30, true)
				if err != nil {
					t.Fatal(err)
				}
				ft, fb = f.Tape, f.ResidentBindings
				d, err := flow.buildDistillDeviceTrainingTape(h, n, .3, .8, 0, make([]float32, n), make([]float32, n), 2<<30)
				if err != nil {
					t.Fatal(err)
				}
				bt, bb = d.tape, d.bindings
				outputs = append([]int(nil), d.tape.Outputs[:5]...)
				for _, b := range bb {
					outputs = append(outputs, b.Gradient)
				}
				externals = []int{d.ids.Condition}
				externals = append(externals, d.ids.Forward[:]...)
				var ids distillDeviceIDs
				guard, ids, err = buildDistillDeviceAdmission(h, n, 0, 2<<30)
				if err != nil {
					t.Fatal(err)
				}
				guardExternal = append([]int{ids.Condition}, ids.Forward[:]...)
			}
			condParam := fb["flow_lm.flow_net.cond_embed.weight"].Parameter
			writers := map[int]train.Command{}
			for _, c := range ft.Commands {
				writers[c.Output] = c
			}
			var ids []int
			seen := map[int]bool{}
			for _, c := range ft.Commands {
				if c.Op == "linear-forward" && len(c.Inputs) == 4 && c.Inputs[2] == condParam {
					id := c.Inputs[0]
					for {
						w, ok := writers[id]
						if !ok || w.Op != "reshape" || len(w.Inputs) != 1 {
							break
						}
						id = w.Inputs[0]
					}
					if !seen[id] {
						ids = append(ids, id)
						seen[id] = true
					}
				}
			}
			count := 1
			if kind == "distill" {
				count = 2
			}
			if len(ids) != count {
				t.Fatal("forward condition sources", ids)
			}
			ft, err = train.WithExternalTapeInputs(ft, ids, 2<<30)
			if err != nil {
				t.Fatal(err)
			}
			if kind == "distill" {
				for i, c := range ft.Commands {
					for j, id := range c.Inputs {
						if id == ids[1] {
							ft.Commands[i].Inputs[j] = ids[0]
						}
					}
				}
				// Production preserves the unused logical extent but clears its
				// external declaration before binding the single condition edge.
				ft.Tensors[ids[1]].External = false
			}
			if err = train.ValidateResidentTapeBindings(ft, fb, 2<<30); err != nil {
				t.Fatal(err)
			}
			conditionID := ids[0]
			assertAuditExternalInventory(t, ft, []int{conditionID})
			forwardCount := 3
			if kind == "distill" {
				forwardCount = 6
			}
			fl, fo := account(label+"/"+kind+"-forward", ft, fb, ft.Outputs[:forwardCount], []int{conditionID})
			bl, bo := account(label+"/"+kind+"-backward", bt, bb, outputs, externals)
			logical := pl + cl + fl + bl
			// Admission executes before backward and closes its consumer first.
			// Its actual finite-witness graph owns zeros/reshapes/dots too.
			assertAuditRoots(t, guard, guard.Outputs, []int{1, 1})
			assertAuditExternalInventory(t, guard, guardExternal)
			guardLogical, guardOwned := account(label+"/"+kind+"-guard", guard, nil, guard.Outputs, guardExternal)
			guardPeak := pl + cl + fl + guardLogical
			logical = max(logical, guardPeak)
			// Shape-view deduplication reduces the declared source itself; keep
			// the original ceiling instead of an obsolete expected rejection.
			if g.frames == 128 && logical > 2<<30 {
				t.Fatal("128-frame complete source exceeds ceiling", kind, logical)
			}
			if g.frames <= 32 && logical > 2<<30 {
				t.Fatal("small geometry all-live ceiling drift", label, kind, logical)
			}
			owned := po + co + fo + max(bo, guardOwned)
			t.Logf("phase=%s/%s rows=%d residentLogical=%d allLiveTapeLogicalLowerBound=%d defaultTape256MiBReject=%t separateTape2GiBReject=%t residentPlusDedupedOwnedLowerBound=%d omitted=compact-index-rewrite,backend-overhead,host-metadata", label, kind, rows, resident, logical, logical > 256<<20, logical > 2<<30, resident+owned)
		}
	}
}

func auditLogicalBytes(tape train.Tape) (int64, error) {
	var total int64
	for _, s := range tape.Tensors {
		n, err := s.Elements()
		if err != nil {
			return 0, err
		}
		total += 4 * int64(n)
	}
	return total, nil
}

func assertAuditRoots(t *testing.T, tape train.Tape, roots, sizes []int) {
	t.Helper()
	if len(roots) != len(sizes) {
		t.Fatal("audit root count")
	}
	declared := map[int]bool{}
	for _, id := range tape.Outputs {
		declared[id] = true
	}
	seen := map[int]bool{}
	for i, id := range roots {
		if id < 0 || id >= len(tape.Tensors) || seen[id] || !declared[id] {
			t.Fatal("audit undeclared/duplicate root", id)
		}
		seen[id] = true
		n, err := tape.Tensors[id].Elements()
		if err != nil || n != sizes[i] {
			t.Fatal("ordered audit root extent", i, id, n, err)
		}
	}
}
func auditExternalInventory(tape train.Tape, ids []int) error {
	bindings := map[int]bool{}
	for _, id := range ids {
		if id < 0 || id >= len(tape.Tensors) || bindings[id] || !tape.Tensors[id].External {
			return fmt.Errorf("duplicate/undeclared audit external %d", id)
		}
		bindings[id] = true
	}
	return train.ValidateExternalTapeBindings(tape, bindings)
}
func assertAuditExternalInventory(t *testing.T, tape train.Tape, ids []int) {
	t.Helper()
	if err := auditExternalInventory(tape, ids); err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 {
		return
	}
	if auditExternalInventory(tape, ids[1:]) == nil {
		t.Fatal("missing external accepted")
	}
	duplicate := append(append([]int(nil), ids...), ids[0])
	if auditExternalInventory(tape, duplicate) == nil {
		t.Fatal("duplicate external accepted")
	}
	for id, s := range tape.Tensors {
		if !s.External {
			invalid := append(append([]int(nil), ids...), id)
			if auditExternalInventory(tape, invalid) == nil {
				t.Fatal("undeclared external accepted")
			}
			break
		}
	}
}
