package pockettts

import (
	"fmt"
	"sort"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Offline geometry sensitivity and DESIGN-only recomputation storage arithmetic.
// Plans are never executed, and complete original source admission is preserved.
func TestTrainingWorkingSetSensitivityAudit(t *testing.T) {
	cfg, _, backing := releasedDeviceAuditFixture(t)
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
	lm := &FlowLMTrainingCPU{Transformer: tr, Embedding: zero(cfg.FlowLM.LookupTable.NBins * h), BOS: zero(n), BOSBeforeVoice: zero(h), SpeakerProjection: linear(n, h), Input: linear(n, h), EOS: linear(h, 1), Hidden: h, LatentDim: n, Vocabulary: cfg.FlowLM.LookupTable.NBins}
	lm.EOS.Bias = zero(1)
	const resident int64 = 1839502424
	account := func(label string, tape train.Tape, b map[string]train.ResidentTapeBinding, outputs []int, c map[string]train.ResidentCompactEmbeddingBinding) (int64, int64) {
		t.Helper()
		if err := tape.Validate(2 << 30); err != nil {
			t.Fatal(label, err)
		}
		roots := append([]int(nil), outputs...)
		for _, v := range b {
			roots = append(roots, v.Gradient)
		}
		for _, v := range c {
			roots = append(roots, v.Index)
		}
		p, remap, err := train.PruneTape(tape, roots, 2<<30)
		if err != nil {
			t.Fatal(label, err)
		}
		borrowed := map[int]bool{}
		protected := map[int]bool{}
		for _, v := range b {
			if id := remap[v.Parameter]; id >= 0 {
				borrowed[id] = true
				protected[id] = true
			}
		}
		for id, s := range p.Tensors {
			if s.External {
				borrowed[id] = true
			}
		}
		plan, err := train.PlanZeroTapeStorage(p, 2<<30, protected)
		if err != nil {
			t.Fatal(err)
		}
		owned := plan.Bytes
		seen := map[int]bool{}
		for id := range borrowed {
			slot := plan.Slots[id]
			if !seen[slot] {
				owned -= 4 * int64(plan.Elements[slot])
				seen[slot] = true
			}
		}
		logical, err := auditLogicalBytes(tape)
		if err != nil || owned < 0 {
			t.Fatal(err, owned)
		}
		pruned, _ := auditLogicalBytes(p)
		t.Logf("laterPhase=%s fullLogical=%d prunedLogical=%d dedupedOwnedPlan=%d residentPlusOwned=%d default256MiBReject=%t source2GiBReject=%t tensors=%d commands=%d", label, logical, pruned, owned, resident+owned, logical > 256<<20, logical > 2<<30, len(tape.Tensors), len(tape.Commands))
		return logical, owned
	}
	weight := NewDefaultLSDWeightMLP()
	type sensitivityPlan struct{ logical, owned int64 }
	observed := map[string]sensitivityPlan{}
	for _, g := range []struct {
		frames, tokens, voice, valid int
		unique                       bool
	}{{7, 3, 2, 7, true}, {7, 3, 2, 3, true}, {32, 32, 0, 32, true}, {32, 32, 0, 32, false}, {32, 32, 0, 8, true}, {32, 8, 32, 32, false}, {128, 32, 8, 16, true}, {32, 1025, 0, 32, true}, {32, 1, 1025, 32, false}} {
		label := fmt.Sprintf("f%d-t%d-v%d", g.frames, g.tokens, g.voice) + fmt.Sprintf("-valid%d-unique%t", g.valid, g.unique)
		mask := make([]bool, g.frames)
		s := TrainingStepSamples{Mask: mask, Noise: make([]float32, g.frames*n), DiagonalTime: make([]float32, g.frames), DistillS: make([]float32, g.frames), DistillT: make([]float32, g.frames)}
		for i := range mask {
			mask[i] = i < g.valid
			if !mask[i] {
				continue
			}
			s.DiagonalTime[i] = .3
			s.DistillS[i] = .3
			s.DistillT[i] = .8
		}
		in := StepReductionInputs{Frames: g.frames, Hidden: h, Latent: n, Mask: mask, DiagonalCondition: make([]float32, g.frames*h), DiagonalTarget: make([]float32, g.frames*n), RowMetrics: make([]float32, g.frames*4), DiagonalLogvar: make([]float32, g.frames), DistillCondition: make([]float32, g.frames*h), DistillTarget: make([]float32, g.frames*n), DistillLogvar: make([]float32, g.frames)}
		r, err := BuildStepReductionTrainingTape(in, DefaultTrainingStepConfig(), 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		account(label+"/standalone-reduction", r, nil, r.Outputs, nil)
		samples := WeightSampleInputs{DiagonalTime: s.DiagonalTime, DistillS: s.DistillS, DistillT: s.DistillT, DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
		w, err := weight.buildStepWeightTrainingTape(in, DefaultTrainingStepConfig(), samples, 2<<30, true)
		if err != nil {
			t.Fatal(err)
		}
		account(label+"/standalone-weight", w.Tape, w.ResidentBindings, w.Tape.Outputs[:5], nil)
		batch := FlowLMTrainingBatch{Frames: g.frames, VoiceFrames: g.voice, TextTokens: make([]uint32, g.tokens), VoiceLatents: make([]float32, g.voice*n), NormalizedLatents: make([]float32, g.frames*n)}
		if g.unique {
			for i := range batch.TextTokens {
				batch.TextTokens[i] = uint32(i)
			}
		}
		b, err := lm.buildStepBackboneTrainingTape(weight, batch, make([]float32, g.frames*h), in, DefaultTrainingStepConfig(), samples, 2<<30, true)
		if err != nil {
			if g.tokens <= 1024 && g.voice <= 1024 {
				t.Fatal("admitted sensitivity geometry", label, err)
			}
			if err.Error() != "step backbone rows" {
				t.Fatal("sensitivity rejection category", label, err)
			}
			t.Logf("sensitivity expected topology rejection %s: %v", label, err)
			continue
		}
		b, err = buildStepResultsTape(b, batch, in, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		outputs := stepResultsOutputs(b, batch)
		dl, do := account(label+"/combined-backbone-dense", b.Tape, b.ResidentBindings, outputs, nil)
		ct, cb, compact, err := compactResidentEmbeddingTape(b.Tape, b.ResidentBindings, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		if len(compact) != 1 {
			t.Fatal("compact inventory", len(compact))
		}
		// compactResidentEmbeddingTape appends new tensors/commands/output roots;
		// original IDs are deliberately stable. Assert that metadata contract
		// before reusing the original reduced-result roots, not a signal equality.
		if len(ct.Outputs) < len(b.Tape.Outputs) || len(ct.Tensors) < len(b.Tape.Tensors) {
			t.Fatal("compact source metadata shortened")
		}
		for i, id := range b.Tape.Outputs {
			if ct.Outputs[i] != id {
				t.Fatal("compact original output ID changed", i)
			}
		}
		for _, id := range outputs {
			a, e1 := b.Tape.Tensors[id].Elements()
			c, e2 := ct.Tensors[id].Elements()
			if e1 != nil || e2 != nil || a != c {
				t.Fatal("compact result-root extent changed", id)
			}
		}
		cl, co := account(label+"/combined-backbone-compact", ct, cb, outputs, compact)
		if cl <= dl || co >= do {
			t.Fatal("compact full-source/owned expectation", cl, dl, co, do)
		}
		if cl > 2<<30 {
			t.Fatal("later full-source ceiling drift", label, cl)
		}
		observed[label] = sensitivityPlan{cl, co}
		if g.tokens > 1024 || g.voice > 1024 {
			t.Fatal("topology rejection missing")
		}
		if g.frames == 32 && g.tokens == 32 && g.valid == 32 {
			// DESIGN-only recomputation batches: retain numerical output roots in every
			// batch, compute selected parameter roots, consume them before next batch.
			// Full original source admission remains cl, never reduced to these plans.
			monolithicRoots := append([]int(nil), outputs...)
			for _, v := range cb {
				monolithicRoots = append(monolithicRoots, v.Gradient)
			}
			for _, v := range compact {
				monolithicRoots = append(monolithicRoots, v.Index)
			}
			prunedMonolithic, _, err := train.PruneTape(ct, monolithicRoots, 2<<30)
			if err != nil {
				t.Fatal(err)
			}
			names := make([]string, 0, len(cb))
			for name := range cb {
				names = append(names, name)
			}
			sort.Strings(names)
			var peak int64
			commands := 0
			groups := 0
			for first := 0; first < len(names); first += 8 {
				roots := append([]int(nil), outputs...)
				for _, name := range names[first:min(first+8, len(names))] {
					roots = append(roots, cb[name].Gradient)
					if v, ok := compact[name]; ok {
						roots = append(roots, v.Index)
					}
				}
				// All original parameter sources remain resident borrows even though
				// only this group's gradients are requested. A subset binding map
				// alone would incorrectly charge unselected parameters as owned.
				p, remap, err := train.PruneTape(ct, roots, 2<<30)
				if err != nil {
					t.Fatal(err)
				}
				protected := map[int]bool{}
				for _, v := range cb {
					if id := remap[v.Parameter]; id >= 0 {
						protected[id] = true
					}
				}
				plan, err := train.PlanZeroTapeStorage(p, 2<<30, protected)
				if err != nil {
					t.Fatal(err)
				}
				owned := plan.Bytes
				slots := map[int]bool{}
				for id := range protected {
					slots[plan.Slots[id]] = true
				}
				for slot := range slots {
					owned -= 4 * int64(plan.Elements[slot])
				}
				peak = max(peak, owned)
				commands += len(p.Commands)
				groups++
			}
			if peak >= co || resident+peak >= 2<<30 || commands <= len(prunedMonolithic.Commands) {
				t.Fatal("DESIGN peak/work tradeoff not established", peak, co, commands, len(prunedMonolithic.Commands))
			}
			t.Logf("DESIGN recompute-batches=%d maxOwned=%d residentPlusMax=%d retainedFullSourceCharge=%d cumulativePrunedCommands=%d monolithicPrunedCommands=%d originalFullSourceCommands=%d no-execution-no-latency-no-overhead-claim", groups, peak, resident+peak, cl, commands, len(prunedMonolithic.Commands), len(ct.Commands))
		}

		t.Logf("laterPhase=%s compact fullsourceAddition=%d ownedPlanReduction=%d no-crossphase-sum/no-native-overhead", label, cl-dl, do-co)
	}
	// Mask/unique token data cannot remove original admitted tensor extents.
	for _, pair := range [][2]string{{"f7-t3-v2-valid7-uniquetrue", "f7-t3-v2-valid3-uniquetrue"}, {"f32-t32-v0-valid32-uniquetrue", "f32-t32-v0-valid32-uniquefalse"}, {"f32-t32-v0-valid32-uniquetrue", "f32-t32-v0-valid8-uniquetrue"}} {
		a, ok := observed[pair[0]]
		b, ok2 := observed[pair[1]]
		if !ok || !ok2 || a.logical != b.logical {
			t.Fatal("sensitivity admitted inventory missing/different", pair, a, b)
		}
	}
}
