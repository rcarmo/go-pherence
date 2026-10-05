package pockettts

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

type pocketAdapterErrors struct {
	count                 int
	sum, squares, maximum float64
}

func (e *pocketAdapterErrors) compare(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal(label, "length", len(got), len(want))
	}
	for i, x := range got {
		d := math.Abs(float64(x) - float64(want[i]))
		if !isFinite(x) || !isFinite(want[i]) || d > tol {
			t.Fatal(label, i, x, want[i], d)
		}
		e.count++
		e.sum += d
		e.squares += d * d
		e.maximum = math.Max(e.maximum, d)
	}
}
func (e pocketAdapterErrors) log(t *testing.T, label string) {
	t.Helper()
	t.Logf("%s comparisons=%d meanAbs=%.9g RMS=%.9g maxAbs=%.9g", label, e.count, e.sum/float64(e.count), math.Sqrt(e.squares/float64(e.count)), e.maximum)
}
func TestPocketAdapterFlowOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var o flowHeadBackwardOracle
	if json.Unmarshal(data, &o) != nil || o.Schema != 1 || o.UpstreamRevision != UpstreamCommit {
		t.Fatal("oracle provenance")
	}
	var stats pocketAdapterErrors
	m := flowHeadFromOracle(t, o.Parameters)
	for index, mix := range o.Mixed {
		p, err := m.BuildFlowDualTrainingTape(o.Condition, o.Times, o.Input, mix.TimeIndex, mix.DOutput, mix.DTangent, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		expected := flowAdapterExpected(mix.Gradients)
		if len(expected) != 34 {
			t.Fatal("pinned flow expected inventory drift")
		}
		if e := adapterOracleInventory(p.Tape, p.ResidentBindings, expected); e != nil {
			t.Fatal(e)
		}
		for _, reuse := range []bool{false, true} {
			out, ledger, err := runPocketAdapter(context.Background(), p.Tape, nil, nil, pocketAdapterOptions{reuse: reuse, shareZero: true})
			if err != nil {
				t.Fatal(index, err)
			}
			r, err := p.Decode(out)
			if err != nil {
				t.Fatal(err)
			}
			if e := adapterComputedInventory(r.Parameters, expected); e != nil {
				t.Fatal(e)
			}
			stats.compare(t, "flow output", r.Output, o.Output, 3e-6)
			stats.compare(t, "flow tangent", r.Tangent, o.JVPTimes[mix.TimeIndex], 5e-6)
			stats.compare(t, "flow input", r.DInput, mix.DInput, 8e-6)
			stats.compare(t, "flow condition", r.DCondition, mix.DCondition, 8e-6)
			stats.compare(t, "flowtime", r.DTimes, mix.DTimes, 8e-6)
			for name, v := range r.Parameters {
				stats.compare(t, name, v, mix.Gradients[name[len("flow_lm.flow_net."):]], 1e-5)
			}
			t.Logf("flowcase%d reuse%t hostledger=%+v", index, reuse, ledger)
		}
	}
	stats.log(t, "flowmixed pinned PyTorch")
}
func TestPocketAdapterTransformerOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/transformer_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle transformerBackwardOracle
	if json.Unmarshal(data, &oracle) != nil || oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatal("oracle provenance")
	}
	var stats pocketAdapterErrors
	for i, o := range []transformerBackwardOracle{oracle} {
		m := transformerFromOracle(t, o)
		p, err := m.BuildTransformerTrainingTape(o.Sequence, o.DOutput, o.Rows, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		expected := transformerAdapterExpected(o.Gradients)
		if len(expected) != 12 {
			t.Fatal("pinned transformer expected inventory drift")
		}
		if e := adapterOracleInventory(p.Tape, p.ResidentBindings, expected); e != nil {
			t.Fatal(e)
		}
		borrowed := map[int][]float32{}
		protect := map[int]bool{}
		for name, b := range p.ResidentBindings {
			key := name[len("flow_lm."):]
			if len(key) >= len("out_norm.") && key[:len("out_norm.")] == "out_norm." {
				key = "final." + key[len("out_norm."):]
			}
			borrowed[b.Parameter] = o.Parameters[key]
			protect[b.Parameter] = true
		}
		out, ledger, err := runPocketAdapter(context.Background(), p.Tape, borrowed, protect, pocketAdapterOptions{reuse: true, shareZero: true})
		if err != nil {
			t.Fatal(i, err)
		}
		computed := map[string][]float32{}
		positions := map[int]int{}
		for j, id := range p.Tape.Outputs {
			positions[id] = j
		}
		for name, b := range p.ResidentBindings {
			computed[name] = out[positions[b.Gradient]]
		}
		if e := adapterComputedInventory(computed, expected); e != nil {
			t.Fatal(e)
		}
		stats.compare(t, "transformer output", out[0], o.Output, 4e-6)
		stats.compare(t, "transformer input", out[1], o.DSequence, 8e-6)
		for j, id := range p.Tape.Outputs {
			for name, b := range p.ResidentBindings {
				if b.Gradient == id {
					key := name[len("flow_lm."):]
					if len(key) >= len("out_norm.") && key[:len("out_norm.")] == "out_norm." {
						key = "final." + key[len("out_norm."):]
					}
					stats.compare(t, name, out[j], o.Gradients[key], 1e-5)
				}
			}
		}
		t.Logf("transformercase%d hostledger=%+v", i, ledger)
	}
	stats.log(t, "transformer pinned PyTorch")
}
func TestPocketAdapterAdmissionCancellation(t *testing.T) {
	base := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}, {Op: "vector", Inputs: []int{2, 1}, Output: 3, Scale: 1}}, Outputs: []int{3}}
	for _, kind := range []string{"op", "mode", "kind", "missing-external", "bad-extent", "outputs", "ctx"} {
		t.Run(kind, func(t *testing.T) {
			tape := pocketAdapterFreeze(base)
			bound := map[int][]float32{}
			ctx := context.Background()
			switch kind {
			case "op":
				tape.Commands[1].Op = "unknown"
			case "mode":
				tape.Commands[1].Mode = 99
			case "kind":
				tape.Commands[1].Kind = 1
			case "missing-external":
				tape.Tensors[0].Values = nil
				tape.Tensors[0].External = true
			case "bad-extent":
				bound[0] = []float32{1}
			case "outputs":
				tape.Outputs = make([]int, 4097)
			case "ctx":
				c, cancel := context.WithCancel(ctx)
				cancel()
				ctx = c
			}
			out, ledger, err := runPocketAdapter(ctx, tape, bound, nil, pocketAdapterOptions{reuse: true})
			if err == nil || out != nil || ledger.Dispatched != 0 || ledger.SlotPayload != 0 {
				t.Fatal("preallocation/preflight rejection", kind, err, ledger)
			}
		})
	}
	for _, stage := range []string{"before", "after"} {
		t.Run(stage, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			opt := pocketAdapterOptions{reuse: true}
			if stage == "before" {
				opt.before = func(i int) {
					if i == 0 {
						cancel()
					}
				}
			} else {
				opt.after = func(i int) {
					if i == 0 {
						cancel()
					}
				}
			}
			out, ledger, err := runPocketAdapter(ctx, base, nil, nil, opt)
			want := 0
			if stage == "after" {
				want = 1
			}
			if err != context.Canceled || out != nil || ledger.Dispatched != want {
				t.Fatal("cancel no later dispatch", stage, err, ledger)
			}
		})
	}
}
func TestPocketAdapterBorrowedReshapeOwnedOutputs(t *testing.T) {
	borrowed := []float32{2, 4}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, External: true}, {Shape: []int{1, 2}}, {Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "reshape", Inputs: []int{0}, Output: 1}, {Op: "reshape", Inputs: []int{1}, Output: 3}, {Op: "vector", Inputs: []int{3, 2}, Output: 4, Scale: 1}}, Outputs: []int{4, 3}}
	var stats pocketAdapterErrors
	out, ledger, err := runPocketAdapter(context.Background(), tape, map[int][]float32{0: borrowed}, nil, pocketAdapterOptions{reuse: true, shareZero: true, before: func(i int) {
		if i == 0 {
			borrowed[0] = 77
			tape.Tensors[2].Values[0] = 99
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	stats.compare(t, "frozen borrowed reshape", out[0], []float32{2, 4}, 2e-6)
	stats.compare(t, "reshape returned owned", out[1], []float32{2, 4}, 2e-6)
	if ledger.BorrowCopies != 8 || ledger.ReboundSlotPayload != 8 || ledger.SlotPayload <= ledger.ReboundSlotPayload {
		t.Fatal("rebound allocated slot cost omitted", ledger)
	}
	out[0][0] = 88
	if out[1][0] != 2 || borrowed[0] != 77 {
		t.Fatal("outputs not independently owned")
	}
	t.Logf("borrowedreshape hostledger=%+v", ledger)
}
func pocketAdapterManifest(t *testing.T, label string, tape train.Tape, bindings map[string]train.ResidentTapeBinding, roots []int, compact ...map[string]train.ResidentCompactEmbeddingBinding) {
	t.Helper()
	var err error
	if len(compact) > 0 {
		err = train.ValidateResidentCompactTapeBindings(tape, bindings, compact[0], 1<<20)
	} else {
		err = train.ValidateResidentTapeBindings(tape, bindings, 1<<20)
	}
	if err != nil {
		t.Fatal(err)
	}
	inventory := map[string]int{}
	writers := map[int]train.Command{}
	declared := map[int]bool{}
	for _, id := range tape.Outputs {
		declared[id] = true
	}
	for _, c := range tape.Commands {
		if !pocketAdapterTuple(c) {
			t.Fatal("emitted unsupported tuple", c)
		}
		inventory[fmt.Sprintf("%s/%d/%d", c.Op, c.Mode, c.Kind)]++
		writers[c.Output] = c
	}
	names := make([]string, 0, len(bindings))
	for n := range bindings {
		names = append(names, n)
	}
	sort.Strings(names)
	tuples := make([]string, 0, len(inventory))
	for n := range inventory {
		tuples = append(tuples, n)
	}
	sort.Strings(tuples)
	t.Logf("manifest=%s tuples=%v counts=%v", label, tuples, inventory)
	seen := map[int]bool{}
	for i, id := range roots {
		if !declared[id] || seen[id] {
			t.Fatal("result undeclared/duplicate")
		}
		seen[id] = true
		n, _ := tape.Tensors[id].Elements()
		c := writers[id]
		t.Logf("manifest=%s resultIndex%d ID%d elements%d producer%s/%d/%d", label, i, id, n, c.Op, c.Mode, c.Kind)
	}
	for _, n := range names {
		b := bindings[n]
		extent, _ := tape.Tensors[b.Gradient].Elements()
		t.Logf("manifest=%s computedGradient=%s parameter%d gradient%d elements%d", label, n, b.Parameter, b.Gradient, extent)
	}
}
func TestPocketAdapterLaterGraphContracts(t *testing.T) {
	for _, activeTail := range []bool{false, true} {
		for _, voice := range []int{0, 1} {
			lm, w, batch, hidden, in, s := stepBackboneFixture(t, voice)
			if activeTail {
				in.Mask[1] = true
				in.DiagonalLogvar[1] = .2
				in.DistillLogvar[1] = -.1
				in.RowMetrics[4] = .1
				in.RowMetrics[5] = .2
				in.RowMetrics[6] = .3
				in.RowMetrics[7] = .4
				for i := lm.LatentDim; i < len(in.DiagonalTarget); i++ {
					in.DiagonalTarget[i] = .25
				}
				s = stepWeightSamples(in)
			}
			p, err := lm.buildStepBackboneTrainingTape(w, batch, hidden, in, DefaultTrainingStepConfig(), s, 1<<20, true)
			if err != nil {
				t.Fatal(err)
			}
			p, err = buildStepResultsTape(p, batch, in, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			roots := stepResultsOutputs(p, batch)
			label := fmt.Sprintf("activeTail%t/voice%d", activeTail, voice)
			if e := adapterStepRootRoles(p, batch, in, s, roots); e != nil {
				t.Fatal(e)
			}
			swapped := append([]int(nil), roots...)
			swapped[0], swapped[1] = swapped[1], swapped[0]
			if adapterStepRootRoles(p, batch, in, s, swapped) == nil {
				t.Fatal("swapped semantic roles admitted")
			}
			pocketAdapterManifest(t, label+"/dense", p.Tape, p.ResidentBindings, roots)
			out, ledger, err := runPocketAdapter(context.Background(), p.Tape, nil, nil, pocketAdapterOptions{reuse: true, shareZero: true})
			if err != nil {
				t.Fatal(err)
			}
			selected := make([][]float32, len(roots))
			for i, id := range roots {
				for j, output := range p.Tape.Outputs {
					if output == id {
						selected[i] = out[j]
						break
					}
				}
			}
			r, err := decodeStepResultsBoundary(selected, batch, hidden, in, s)
			if err != nil {
				t.Fatal(label, err)
			}
			tail := selected[len(selected)-1]
			if len(tail) != lm.LatentDim {
				t.Fatal("real directtail extent")
			}
			if activeTail {
				nonzero := false
				for _, v := range tail {
					nonzero = nonzero || v != 0
				}
				if !nonzero {
					t.Fatal("active tail fixture fails nonzero contract")
				}
			} else {
				for _, v := range tail {
					if v != 0 {
						t.Fatal("inactive directtail")
					}
				}
			}
			_ = r
			// Typed policy negative checks: no scalar-zero sentinel substitution.
			copyResults := func() [][]float32 {
				x := make([][]float32, len(selected))
				for i, v := range selected {
					x[i] = append([]float32(nil), v...)
				}
				return x
			}
			bad := copyResults()
			bad[len(bad)-1][0] += 1
			if _, e := decodeStepResultsBoundary(bad, batch, hidden, in, s); e == nil {
				t.Fatal("directtail mismatch admitted")
			}
			bad = copyResults()
			bad[2][0] = float32(math.NaN())
			if _, e := decodeStepResultsBoundary(bad, batch, hidden, in, s); e == nil {
				t.Fatal("nonfinite metrics admitted")
			}
			bad = copyResults()
			bad[1] = bad[0]
			if _, e := decodeStepResultsBoundary(bad, batch, hidden, in, s); e == nil {
				t.Fatal("aliased results admitted")
			}
			ct, cb, compact, err := compactResidentEmbeddingTape(p.Tape, p.ResidentBindings, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			if err = train.ValidateResidentCompactTapeBindings(ct, cb, compact, 1<<20); err != nil {
				t.Fatal(err)
			}
			// Manifest validates compact descriptors separately; ordinary binding admission
			// must reject a compact gradient. Original result IDs are append-only.
			for i, id := range p.Tape.Outputs {
				if ct.Outputs[i] != id {
					t.Fatal("compact source roots changed")
				}
			}
			if e := adapterStepRootRoles(StepBackboneTrainingTape{Tape: ct, ResidentBindings: cb}, batch, in, s, roots); e != nil {
				t.Fatal(e)
			}
			pocketAdapterManifest(t, label+"/compact", ct, cb, roots, compact)
			compactOut, compactLedger, err := runPocketAdapter(context.Background(), ct, nil, nil, pocketAdapterOptions{reuse: true, shareZero: true})
			if err != nil {
				t.Fatal(err)
			}
			// Schedule-only dense/compact diagnostic; independent component gates
			// above remain the numerical oracles for actual computed operations.
			var schedule pocketAdapterErrors
			for i, id := range roots {
				for j, output := range ct.Outputs {
					if output == id {
						schedule.compare(t, "compact result", compactOut[j], selected[i], 2e-6)
					}
				}
			}
			schedule.log(t, "later compact/dense schedule diagnostic")
			t.Logf("manifest=%s compactDescriptors=%v originalResultRoots=%v hostledger=%+v compactHostledger=%+v", label, compact, roots, ledger, compactLedger)
			for n, b := range cb {
				if _, ok := compact[n]; ok {
					extent, _ := ct.Tensors[b.Gradient].Elements()
					t.Logf("manifest=%s compactComputedGradient=%s ID%d elements%d", label, n, b.Gradient, extent)
				}
			}
		}
	}
}

func TestPocketAdapterFlowLMOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/flowlm_training_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var o flowLMTrainingOracle
	if json.Unmarshal(data, &o) != nil || o.Schema != 1 || o.UpstreamRevision != UpstreamCommit {
		t.Fatal("FlowLM oracle provenance")
	}
	m := flowLMTrainingFromOracle(t, o)
	batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
	rows := 1 + o.VoiceFrames + len(o.TextTokens) + o.Frames
	seed := make([]float32, rows*m.Hidden)
	prefix := rows - o.Frames
	for row := 0; row < o.Frames; row++ {
		for col := 0; col < m.Hidden; col++ {
			seed[(prefix+row)*m.Hidden+col] = o.DZ[row*m.Hidden+col] + o.DEOS[row]*m.EOS.Weight[col]
		}
	}
	p, err := m.BuildBackboneTrainingTape(batch, seed, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	expected, e := flowLMAdapterExpected(o.Gradients)
	if e != nil || len(o.Gradients) != 19 || len(expected) != 17 {
		t.Fatal("pinned FlowLM named EOS subset", e)
	}
	if e = adapterOracleInventory(p.Tape, p.ResidentBindings, expected); e != nil {
		t.Fatal(e)
	}
	out, ledger, err := runPocketAdapter(context.Background(), p.Tape, nil, nil, pocketAdapterOptions{reuse: true, shareZero: true})
	if err != nil {
		t.Fatal(err)
	}
	var stats pocketAdapterErrors
	stats.compare(t, "FlowLM sequence", out[0], o.Sequence, 2e-6)
	stats.compare(t, "FlowLM Z", out[1][prefix*m.Hidden:], o.Z, 5e-6)
	stats.compare(t, "FlowLM normalized input", out[3], o.DNormalizedLatents, 1.5e-5)
	stats.compare(t, "FlowLM voice input", out[2], o.DVoiceLatents, 1.5e-5)
	if len(p.ResidentBindings) != len(o.Gradients)-2 {
		t.Fatal("EOS oracle contribution map")
	}
	values := map[int][]float32{}
	for i, id := range p.Tape.Outputs {
		values[id] = out[i]
	}
	computed := map[string][]float32{}
	for name, b := range p.ResidentBindings {
		computed[name] = values[b.Gradient]
	}
	if e := adapterComputedInventory(computed, expected); e != nil {
		t.Fatal(e)
	}
	for name, b := range p.ResidentBindings {
		stats.compare(t, name, values[b.Gradient], o.Gradients[name[len("flow_lm."):]], 2e-5)
	}
	stats.log(t, "FlowLM pinned PyTorch (EOS parameter roots excluded, EOS input seed supplied)")
	t.Logf("FlowLM hostledger=%+v", ledger)
}
func TestPocketAdapterOverwriteAndTuples(t *testing.T) {
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}, {Shape: []int{2}}, {Shape: []int{2}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}, {Op: "vector", Inputs: []int{2, 1}, Output: 3, Scale: 1}, {Op: "vector", Inputs: []int{3, 1}, Output: 4, Scale: 0}, {Op: "vector", Inputs: []int{4, 1}, Output: 5, Scale: 0}}, Outputs: []int{5}}
	out, ledger, err := runPocketAdapter(context.Background(), tape, nil, nil, pocketAdapterOptions{reuse: true, shareZero: true})
	if err != nil {
		t.Fatal(err)
	}
	var stats pocketAdapterErrors
	stats.compare(t, "last-use slot overwrite", out[0], []float32{7, 10}, 2e-6)
	if ledger.SlotPayload >= 6*2*4 {
		t.Fatal("fixture did not reuse", ledger)
	}
	for _, c := range []train.Command{{Op: "vector", Mode: 3}, {Op: "norm-backward", Mode: 3, Kind: 1}, {Op: "mod-forward", Mode: 2}, {Op: "rope", Mode: 2}, {Op: "embedding", Mode: 4}, {Op: "eos", Kind: 1}, {Op: "unknown"}} {
		if pocketAdapterTuple(c) {
			t.Fatal("unsupported exact tuple admitted", c)
		}
	}
}
