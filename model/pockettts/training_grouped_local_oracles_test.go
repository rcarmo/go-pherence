package pockettts

import (
	"context"
	"encoding/json"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"os"
	"sort"
	"testing"
)

func pocketGroupPartitions(b map[string]train.ResidentTapeBinding, variant int) [][]string {
	names := make([]string, 0, len(b))
	for n := range b {
		names = append(names, n)
	}
	sort.Strings(names)
	if variant == 1 {
		for i, j := 0, len(names)-1; i < j; i, j = i+1, j-1 {
			names[i], names[j] = names[j], names[i]
		}
	}
	groups := [][]string{}
	for len(names) > 0 {
		n := min(8, len(names))
		groups = append(groups, append([]string(nil), names[:n]...))
		names = names[n:]
	}
	return groups
}
func pocketGroupComponent(t train.Tape, roots []int, roles []string) pocketGroupPolicy {
	p := pocketGroupPolicy{kind: "component", roles: roles}
	for _, id := range roots {
		n, _ := t.Tensors[id].Elements()
		p.extents = append(p.extents, n)
	}
	return p
}
func pocketGroupPrior(p *pocketGroupLocal) map[string][]float32 {
	prior := map[string][]float32{}
	for name, g := range p.state.g {
		for i := range g {
			g[i] = float32((i%5)-2) * .015
		}
		prior[name] = append([]float32(nil), g...)
	}
	return prior
}
func pocketGroupCompareG(t *testing.T, label string, p *pocketGroupLocal, expected, prior map[string][]float32, scale float32, tol float64, stats *pocketAdapterErrors) {
	t.Helper()
	if len(p.state.g) != len(expected) {
		t.Fatal("group exact computed inventory")
	}
	for n, w := range expected {
		want := make([]float32, len(w))
		for i, x := range w {
			want[i] = prior[n][i] + scale*x
		}
		stats.compare(t, label+"/"+n, p.state.g[n], want, tol)
	}
}
func TestPocketGroupedLocalFlowOracle(t *testing.T) {
	raw, e := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if e != nil {
		t.Fatal(e)
	}
	var o flowHeadBackwardOracle
	if e = json.Unmarshal(raw, &o); e != nil || o.Schema != 1 || o.UpstreamRevision != UpstreamCommit {
		t.Fatal("flow fixture pin", e)
	}
	m := flowHeadFromOracle(t, o.Parameters)
	var grads, values pocketAdapterErrors
	for ci, mix := range o.Mixed {
		p, e := m.BuildFlowDualTrainingTape(o.Condition, o.Times, o.Input, mix.TimeIndex, mix.DOutput, mix.DTangent, 1<<20)
		if e != nil {
			t.Fatal(e)
		}
		want := flowAdapterExpected(mix.Gradients)
		if e = adapterOracleInventory(p.Tape, p.ResidentBindings, want); e != nil {
			t.Fatal(e)
		}
		roots := append([]int(nil), p.Tape.Outputs[:p.boundarySize()]...)
		roles := []string{"value", "tangent", "condition-vjp", "input-vjp"}
		for i := 4; i < len(roots); i++ {
			roles = append(roles, fmt.Sprintf("time-adjoint%d", i))
		}
		policy := pocketGroupComponent(p.Tape, roots, roles)
		for _, order := range []int{0, 1} {
			for _, scale := range []float32{0, .25, 1} {
				local := newPocketGroup(p.Tape, p.ResidentBindings)
				prior := pocketGroupPrior(local)
				out, e := local.run(context.Background(), p.Tape, p.ResidentBindings, nil, roots, pocketGroupPartitions(p.ResidentBindings, order), scale, policy)
				if e != nil {
					t.Fatal(ci, order, scale, e)
				}
				values.compare(t, "flow value", out[0], o.Output, 3e-6)
				values.compare(t, "flow tangent", out[1], o.JVPTimes[mix.TimeIndex], 5e-6)
				boundary, e := p.decodeBoundary(out)
				if e != nil {
					t.Fatal(e)
				}
				values.compare(t, "flow input", boundary.DInput, mix.DInput, 8e-6)
				values.compare(t, "flow condition", boundary.DCondition, mix.DCondition, 8e-6)
				values.compare(t, "flow time", boundary.DTimes, mix.DTimes, 8e-6)
				pocketGroupCompareG(t, "flow", local, want, prior, scale, 1e-5, &grads)
				if order == 0 && scale == 1 {
					t.Logf("flowcase%d grouped hostledger=%+v", ci, local.ledger)
				}
			}
		}
	}
	grads.log(t, "grouped flow pinned PyTorch prior+scale G")
	values.log(t, "grouped flow pinned PyTorch values/VJPs")
}
func TestPocketGroupedLocalTransformerOracle(t *testing.T) {
	raw, e := os.ReadFile("testdata/transformer_backward_pytorch.json")
	if e != nil {
		t.Fatal(e)
	}
	var o transformerBackwardOracle
	if e = json.Unmarshal(raw, &o); e != nil || o.Schema != 1 || o.UpstreamRevision != UpstreamCommit {
		t.Fatal("transformer pin", e)
	}
	m := transformerFromOracle(t, o)
	p, e := m.BuildTransformerTrainingTape(o.Sequence, o.DOutput, o.Rows, 1<<20)
	if e != nil {
		t.Fatal(e)
	}
	want := transformerAdapterExpected(o.Gradients)
	if e = adapterOracleInventory(p.Tape, p.ResidentBindings, want); e != nil {
		t.Fatal(e)
	}
	roots := append([]int(nil), p.Tape.Outputs[:2]...)
	policy := pocketGroupComponent(p.Tape, roots, []string{"transformer-output", "sequence-vjp"})
	var grads, values pocketAdapterErrors
	for _, order := range []int{0, 1} {
		for _, scale := range []float32{0, .25, 1} {
			local := newPocketGroup(p.Tape, p.ResidentBindings)
			prior := pocketGroupPrior(local)
			out, e := local.run(context.Background(), p.Tape, p.ResidentBindings, nil, roots, pocketGroupPartitions(p.ResidentBindings, order), scale, policy)
			if e != nil {
				t.Fatal(e)
			}
			values.compare(t, "transformer output", out[0], o.Output, 4e-6)
			values.compare(t, "transformer input", out[1], o.DSequence, 8e-6)
			pocketGroupCompareG(t, "transformer", local, want, prior, scale, 1e-5, &grads)
			if order == 0 && scale == 1 {
				t.Logf("transformer grouped hostledger=%+v", local.ledger)
			}
		}
	}
	grads.log(t, "grouped transformer pinned PyTorch prior+scale G")
	values.log(t, "grouped transformer pinned PyTorch values/VJPs")
}
func TestPocketGroupedLocalFlowLMOracle(t *testing.T) {
	raw, e := os.ReadFile("testdata/flowlm_training_pytorch.json")
	if e != nil {
		t.Fatal(e)
	}
	var o flowLMTrainingOracle
	if e = json.Unmarshal(raw, &o); e != nil || o.Schema != 1 || o.UpstreamRevision != UpstreamCommit {
		t.Fatal("FlowLM pin", e)
	}
	m := flowLMTrainingFromOracle(t, o)
	batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
	rows := 1 + o.VoiceFrames + len(o.TextTokens) + o.Frames
	seed := make([]float32, rows*m.Hidden)
	offset := rows - o.Frames
	for row := 0; row < o.Frames; row++ {
		for ch := 0; ch < m.Hidden; ch++ {
			seed[(offset+row)*m.Hidden+ch] = o.DZ[row*m.Hidden+ch] + o.DEOS[row]*m.EOS.Weight[ch]
		}
	}
	p, e := m.BuildBackboneTrainingTape(batch, seed, 1<<20)
	if e != nil {
		t.Fatal(e)
	}
	want, e := flowLMAdapterExpected(o.Gradients)
	if e != nil {
		t.Fatal(e)
	}
	if e = adapterOracleInventory(p.Tape, p.ResidentBindings, want); e != nil {
		t.Fatal(e)
	}
	roots := append([]int(nil), p.Tape.Outputs[:4]...)
	policy := pocketGroupComponent(p.Tape, roots, []string{"sequence", "z", "voice-vjp", "normalized-vjp"})
	var grads, values pocketAdapterErrors
	for _, order := range []int{0, 1} {
		for _, scale := range []float32{0, .25, 1} {
			local := newPocketGroup(p.Tape, p.ResidentBindings)
			prior := pocketGroupPrior(local)
			out, e := local.run(context.Background(), p.Tape, p.ResidentBindings, nil, roots, pocketGroupPartitions(p.ResidentBindings, order), scale, policy)
			if e != nil {
				t.Fatal(e)
			}
			values.compare(t, "FlowLM sequence", out[0], o.Sequence, 2e-6)
			values.compare(t, "FlowLM z", out[1][offset*m.Hidden:], o.Z, 5e-6)
			values.compare(t, "FlowLM normalized", out[3], o.DNormalizedLatents, 1.5e-5)
			values.compare(t, "FlowLM voice", out[2], o.DVoiceLatents, 1.5e-5)
			pocketGroupCompareG(t, "FlowLM", local, want, prior, scale, 2e-5, &grads)
			if order == 0 && scale == 1 {
				t.Logf("FlowLM grouped hostledger=%+v", local.ledger)
			}
		}
	}
	grads.log(t, "grouped FlowLM pinned PyTorch prior+scale G (named EOS parameters excluded; EOS input seed supplied)")
	values.log(t, "grouped FlowLM pinned PyTorch values/VJPs")
}
func pocketGroupLaterFixture(t *testing.T, voice int, active bool) (StepBackboneTrainingTape, []int, pocketGroupPolicy) {
	t.Helper()
	m, w, batch, hidden, in, s := stepBackboneFixture(t, voice)
	if active {
		in.Mask = []bool{true, true}
		in.DiagonalCondition = append([]float32(nil), in.DiagonalCondition...)
		copy(in.DiagonalCondition[2:], []float32{.25, -.15})
		in.DiagonalTarget = append([]float32(nil), in.DiagonalTarget...)
		copy(in.DiagonalTarget[2:], []float32{.2, -.1})
		in.DiagonalLogvar = []float32{.4, .3}
		in.RowMetrics = append([]float32(nil), in.RowMetrics...)
		copy(in.RowMetrics[4:], []float32{.2, .1, .3, .2})
		s.DiagonalSeed = append([]float32(nil), in.DiagonalLogvar...)
	}
	p, e := m.BuildStepBackboneTrainingTape(w, batch, hidden, in, TrainingStepConfig{PEqual: .25, EOSLossWeight: .1}, s, 1<<20)
	if e != nil {
		t.Fatal(e)
	}
	p, e = buildStepResultsTape(p, batch, in, 1<<20)
	if e != nil {
		t.Fatal(e)
	}
	roots := stepResultsOutputs(p, batch)
	roles := []string{"diagonal-logvar", "distill-logvar", "metrics"}
	if voice > 0 {
		roles = append(roles, "voice-vjp")
	}
	roles = append(roles, "normalized-vjp", "direct-tail")
	policy := pocketGroupComponent(p.Tape, roots, roles)
	policy.kind = "step"
	policy.step = &pocketGroupStepPolicy{batch: batch, in: in, hidden: hidden, sample: s, cfg: TrainingStepConfig{PEqual: .25, EOSLossWeight: .1}}
	return p, roots, policy
}
func TestPocketGroupedLocalLaterDiagnostic(t *testing.T) {
	var stats pocketAdapterErrors
	for _, voice := range []int{0, 1} {
		for _, active := range []bool{false, true} {
			p, roots, policy := pocketGroupLaterFixture(t, voice, active)
			mono, _, e := runPocketAdapter(context.Background(), p.Tape, nil, nil, pocketAdapterOptions{reuse: true, shareZero: true})
			if e != nil {
				t.Fatal(e)
			}
			pos := map[int]int{}
			for i, id := range p.Tape.Outputs {
				pos[id] = i
			}
			for _, compact := range []bool{false, true} {
				tape, b := p.Tape, p.ResidentBindings
				var cb map[string]train.ResidentCompactEmbeddingBinding
				if compact {
					tape, b, cb, e = compactResidentEmbeddingTape(tape, b, 1<<20)
					if e != nil {
						t.Fatal(e)
					}
				}
				for _, order := range []int{0, 1} {
					local := newPocketGroup(tape, b)
					out, e := local.run(context.Background(), tape, b, cb, roots, pocketGroupPartitions(b, order), .25, policy)
					if e != nil {
						t.Fatal(voice, active, compact, order, e)
					}
					for i, id := range roots {
						stats.compare(t, "later selected diagnostic", out[i], mono[pos[id]], 2e-6)
					}
					for name, binding := range p.ResidentBindings {
						want := make([]float32, len(mono[pos[binding.Gradient]]))
						for i, x := range mono[pos[binding.Gradient]] {
							want[i] = .25 * x
						}
						stats.compare(t, name+" later G diagnostic", local.state.g[name], want, 2e-6)
					}
					if order == 0 {
						t.Logf("later voice%d active%t compact%t groups%d hostledger=%+v", voice, active, compact, len(pocketGroupPartitions(b, order)), local.ledger)
					}
				}
			}
		}
	}
	stats.log(t, "grouped later supplied-row/absent-earlier-flowG schedule diagnostic")
}
func TestPocketGroupedLocalTypedTailLateFailure(t *testing.T) {
	p, roots, policy := pocketGroupLaterFixture(t, 1, true)
	local := newPocketGroup(p.Tape, p.ResidentBindings)
	local.corrupt = func(gi int, v [][]float32) {
		if gi != 1 {
			return
		}
		v[len(roots)-1][0] += 1
	}
	out, e := local.run(context.Background(), p.Tape, p.ResidentBindings, nil, roots, pocketGroupPartitions(p.ResidentBindings, 0), 1, policy)
	if e == nil || out != nil || local.state.terminal == nil || len(local.state.retained) != 1 {
		t.Fatal("typed tail late fault", e)
	}
	if len(local.state.ready) != 8 {
		t.Fatal("earlier committed group", local.state.ready)
	}
	t.Logf("typed-tail rejected after first group: %v", e)
}
