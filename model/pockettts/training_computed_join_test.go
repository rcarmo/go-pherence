package pockettts

// TESTONLY sequential, ungrouped boundary join. Execution receives no oracle
// metrics, logvars, derivatives, gradients or optimiser outputs. All candidates
// are private; a stage error returns no result. No resident/group coordinator.
import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

type pocketJoinExecution struct {
	parameters              map[string][]float32 // owned P and fixed buffers only
	batch                   FlowLMTrainingBatch
	mask                    []bool
	noise, diagonal, s, end []float32
	cfg                     TrainingStepConfig
}

type pocketJoinPhase struct {
	Name                                        string
	Tensors, Commands                           int
	LogicalSource, SourceConstants, HeldPayload int64
	InventoryPayload                            int64
	Adapter                                     pocketAdapterLedger
	Operations                                  map[string]int
}
type pocketJoinTrace struct {
	Phases        []pocketJoinPhase
	Contributions []string
	beforeStage   func(string)
	afterStage    func(string, [][]float32) // corruption hooks for rejection tests only
}
type pocketJoinResult struct {
	Later                         stepBackboneTrainingResult
	G                             map[string][]float32
	DiagonalLogvar, DistillLogvar []float32
}

// Exact fixture inventory, independent of the expected gradient arrays.
func pocketJoinInventory() (flow, later, fixed map[string]int) {
	flow, later = map[string]int{}, map[string]int{}
	add := func(dst map[string]int, prefix string, weight, bias int) {
		dst[prefix+".weight"], dst[prefix+".bias"] = weight, bias
	}
	p := "flow_lm.flow_net."
	add(flow, p+"cond_embed", 16, 4)
	add(flow, p+"input_proj", 8, 4)
	add(flow, p+"final_layer.adaLN_modulation.1", 32, 8)
	add(flow, p+"final_layer.linear", 8, 2)
	for i := 0; i < 2; i++ {
		r := fmt.Sprintf("%sres_blocks.%d.", p, i)
		add(flow, r+"adaLN_modulation.1", 48, 12)
		add(flow, r+"in_ln", 4, 4)
		for _, j := range []int{0, 2} {
			add(flow, fmt.Sprintf("%smlp.%d", r, j), 16, 4)
			add(flow, fmt.Sprintf("%stime_embed.%d.mlp.%d", p, i, j), 16, 4)
		}
		flow[fmt.Sprintf("%stime_embed.%d.mlp.3.alpha", p, i)] = 4
	}
	for n, z := range map[string]int{"bos_before_voice": 4, "bos_emb": 2, "conditioner.embed.weight": 24, "input_linear.weight": 8, "speaker_proj_weight": 8, "out_eos.bias": 1, "out_eos.weight": 4, "out_norm.bias": 4, "out_norm.weight": 4} {
		later["flow_lm."+n] = z
	}
	r := "flow_lm.transformer.layers.0."
	for _, i := range []int{1, 2} {
		later[fmt.Sprintf("%slayer_scale_%d.scale", r, i)] = 4
		add(later, fmt.Sprintf("%snorm%d", r, i), 4, 4)
	}
	for n, z := range map[string]int{"linear1.weight": 24, "linear2.weight": 24, "self_attn.in_proj.weight": 48, "self_attn.out_proj.weight": 16} {
		later[r+n] = z
	}
	for i, z := range []int{6, 9, 3} {
		bias := 3
		if i == 2 {
			bias = 1
		}
		add(later, fmt.Sprintf("flow.w_s_t.%d", i*2), z, bias)
	}
	fixed = map[string]int{"flow_lm.emb_mean": 2, "flow_lm.emb_std": 2, "flow_lm.flow_net.time_embed.0.freqs": 2, "flow_lm.flow_net.time_embed.1.freqs": 2}
	return
}
func pocketJoinUnion(a, b map[string]int) (map[string]int, error) {
	out := map[string]int{}
	for _, m := range []map[string]int{a, b} {
		for n, z := range m {
			if _, ok := out[n]; ok {
				return nil, fmt.Errorf("join duplicate name %s", n)
			}
			out[n] = z
		}
	}
	return out, nil
}
func pocketJoinArrayInventory(m map[string][]float32, inv map[string]int) error {
	if len(m) != len(inv) {
		return fmt.Errorf("join name count %d want %d", len(m), len(inv))
	}
	for n, z := range inv {
		v, ok := m[n]
		if !ok || len(v) != z || !finiteF32(v) {
			return fmt.Errorf("join missing/extent/nonfinite %s", n)
		}
	}
	return nil
}
func loadPocketJoinFixture(t *testing.T) (pocketJoinExecution, trainingStepOracle) {
	t.Helper()
	raw, e := os.ReadFile("testdata/training_step_pytorch.json")
	if e != nil {
		t.Fatal(e)
	}
	var o trainingStepOracle
	if e = json.Unmarshal(raw, &o); e != nil {
		t.Fatal(e)
	}
	var meta struct {
		Generator string `json:"generator"`
		Config    struct {
			P          float32 `json:"p_equal"`
			EOS        float32 `json:"eos_loss_weight"`
			Multiplier int     `json:"flow_batch_multiplier"`
		} `json:"config"`
	}
	if e = json.Unmarshal(raw, &meta); e != nil {
		t.Fatal(e)
	}
	if o.Schema != 1 || o.UpstreamRevision != UpstreamCommit || o.Seed != 1234 || meta.Generator != "pocket-tts .venv/bin/python (torch 2.13.0+cu130)" || o.Frames != 3 || o.VoiceFrames != 2 || o.Hidden != 4 || o.LatentDim != 2 || o.Vocabulary != 6 || meta.Config.P != .75 || meta.Config.EOS != .1 || meta.Config.Multiplier != 1 || !reflect.DeepEqual(o.Mask, []bool{true, true, false}) || !reflect.DeepEqual(o.TextTokens, []uint32{1, 3}) {
		t.Fatal("join explicit fixture provenance/config/geometry")
	}
	f, l, b := pocketJoinInventory()
	all, e := pocketJoinUnion(f, l)
	if e != nil {
		t.Fatal(e)
	}
	if len(f) != 34 || len(l) != 25 || len(all) != 59 {
		t.Fatal("join manifest names")
	}
	elems := 0
	for _, z := range all {
		elems += z
	}
	if elems != 606 {
		t.Fatal("join manifest elements", elems)
	}
	if e = pocketJoinArrayInventory(o.Gradients, all); e != nil {
		t.Fatal(e)
	}
	state, e := pocketJoinUnion(all, b)
	if e != nil {
		t.Fatal(e)
	}
	if e = pocketJoinArrayInventory(o.Parameters, state); e != nil {
		t.Fatal(e)
	}
	for i, v := range [][]float32{o.NormalizedLatents, o.VoiceLatents, o.Noise, o.DiagonalTime, o.DistillS, o.DistillT} {
		if len(v) != []int{6, 4, 6, 3, 3, 3}[i] || !finiteF32(v) {
			t.Fatal("join execution input extent/finite", i)
		}
	}
	if !reflect.DeepEqual(o.DiagonalTime, []float32{.2, .7, 0}) || !reflect.DeepEqual(o.DistillS, []float32{.15, .3, 0}) || !reflect.DeepEqual(o.DistillT, []float32{.8, .65, 0}) || o.Noise[4] != 0 || o.Noise[5] != 0 {
		t.Fatal("join pinned sample schedule/padding")
	}
	for i, v := range [][]float32{o.DiagonalLogVariance, o.DistillLogVariance, o.DDiagonalLogVariance, o.DDistillLogVariance, o.DNormalizedLatents, o.DVoiceLatents} {
		if len(v) != []int{2, 2, 2, 2, 6, 4}[i] || !finiteF32(v) {
			t.Fatal("join comparison inventory", i)
		}
	}
	if len(o.Metrics) != 5 {
		t.Fatal("join metric names")
	}
	for _, n := range []string{"flow_diag", "flow_distill", "flow_loss", "eos_loss", "loss"} {
		v, ok := o.Metrics[n]
		if !ok || !finite64(v) {
			t.Fatal("join metric", n)
		}
	}
	// Construct only execution-owned source arrays. No expected field survives.
	clone := func(v []float32) []float32 { return append([]float32(nil), v...) }
	x := pocketJoinExecution{parameters: map[string][]float32{}, cfg: TrainingStepConfig{PEqual: meta.Config.P, EOSLossWeight: meta.Config.EOS}}
	for n, v := range o.Parameters {
		x.parameters[n] = clone(v)
	}
	x.batch = FlowLMTrainingBatch{Frames: 3, VoiceFrames: 2, NormalizedLatents: clone(o.NormalizedLatents), VoiceLatents: clone(o.VoiceLatents), TextTokens: append([]uint32(nil), o.TextTokens...)}
	x.mask = append([]bool(nil), o.Mask...)
	x.noise = clone(o.Noise)
	x.diagonal = clone(o.DiagonalTime)
	x.s = clone(o.DistillS)
	x.end = clone(o.DistillT)
	return x, o
}

func pocketJoinRun(ctx context.Context, name string, tape train.Tape, bindings map[string]train.ResidentTapeBinding, inv map[string]int, held int64, trace *pocketJoinTrace) ([][]float32, error) {
	phase := pocketJoinPhase{Name: name, Tensors: len(tape.Tensors), Commands: len(tape.Commands), HeldPayload: held, Operations: map[string]int{}}
	for _, s := range tape.Tensors {
		n, e := s.Elements()
		if e != nil {
			return nil, e
		}
		phase.LogicalSource += 4 * int64(n)
		phase.SourceConstants += 4 * int64(len(s.Values))
	}
	for _, c := range tape.Commands {
		phase.Operations[fmt.Sprintf("%s/%d/%d", c.Op, c.Mode, c.Kind)]++
	}
	if len(bindings) != len(inv) {
		return nil, fmt.Errorf("%s binding inventory", name)
	}
	extents := map[string][]float32{}
	for n, z := range inv {
		extents[n] = make([]float32, z)
		phase.InventoryPayload += 4 * int64(z)
	}
	if e := adapterOracleInventory(tape, bindings, extents); e != nil {
		return nil, fmt.Errorf("%s: %w", name, e)
	}
	if e := train.ValidateResidentTapeBindings(tape, bindings, 1<<20); e != nil {
		return nil, fmt.Errorf("%s: %w", name, e)
	}
	if trace.beforeStage != nil {
		trace.beforeStage(name)
	}
	out, ledger, e := runPocketAdapter(ctx, tape, nil, nil, pocketAdapterOptions{reuse: true, shareZero: true})
	phase.Adapter = ledger
	trace.Phases = append(trace.Phases, phase)
	if e != nil {
		return nil, fmt.Errorf("%s: %w", name, e)
	}
	if trace.afterStage != nil {
		trace.afterStage(name, out)
	}
	return out, nil
}
func pocketJoinGradients(tape train.Tape, b map[string]train.ResidentTapeBinding, out [][]float32, inv map[string]int) (map[string][]float32, error) {
	if len(out) != len(tape.Outputs) {
		return nil, fmt.Errorf("join output count")
	}
	pos := map[int]int{}
	for i, id := range tape.Outputs {
		if _, ok := pos[id]; ok {
			return nil, fmt.Errorf("join duplicate output")
		}
		pos[id] = i
	}
	g := map[string][]float32{}
	for n, v := range b {
		i, ok := pos[v.Gradient]
		if !ok {
			return nil, fmt.Errorf("join undeclared gradient")
		}
		g[n] = out[i]
	}
	if e := pocketJoinArrayInventory(g, inv); e != nil {
		return nil, e
	}
	return g, nil
}
func pocketJoinAccumulate(total map[string][]float32, got map[string][]float32, inv map[string]int, scale float32) error {
	if e := pocketJoinArrayInventory(got, inv); e != nil {
		return e
	}
	if !isFinite(scale) || scale < 0 || scale > 1 {
		return fmt.Errorf("join contribution scale")
	}
	// Validate the whole candidate sum before writing even private totals.
	for n, v := range got {
		if len(total[n]) != len(v) {
			return fmt.Errorf("join total extent")
		}
		for i, z := range v {
			if !isFinite(total[n][i] + scale*z) {
				return fmt.Errorf("join sum nonfinite")
			}
		}
	}
	for n, v := range got {
		for i, z := range v {
			total[n][i] += scale * z
		}
	}
	return nil
}
func pocketJoinSelected(tape train.Tape, out [][]float32, roots []int) ([][]float32, error) {
	if len(out) != len(tape.Outputs) {
		return nil, fmt.Errorf("join selected count")
	}
	pos := map[int]int{}
	for i, id := range tape.Outputs {
		if _, ok := pos[id]; ok {
			return nil, fmt.Errorf("join duplicate declared root")
		}
		pos[id] = i
	}
	seen := map[int]bool{}
	selected := make([][]float32, len(roots))
	for i, id := range roots {
		at, ok := pos[id]
		if !ok || seen[id] {
			return nil, fmt.Errorf("join missing/duplicate selected root")
		}
		seen[id] = true
		selected[i] = out[at]
	}
	return selected, nil
}

func computePocketJoin(t *testing.T, ctx context.Context, x pocketJoinExecution, trace *pocketJoinTrace) (*pocketJoinResult, error) {
	t.Helper()
	// This construction helper receives only source state and geometry: the
	// type accepted by computePocketJoin cannot carry oracle result arrays.
	lm, flow, weight := trainingStepModelsFromOracle(t, trainingStepOracle{Frames: 3, VoiceFrames: 2, Hidden: 4, LatentDim: 2, Vocabulary: 6, Parameters: x.parameters})
	f, l, _ := pocketJoinInventory()
	cfg := x.cfg
	batch := x.batch
	// Source-array accounting excludes Go metadata/allocator/GC overhead. The
	// models duplicate the immutable execution P; charge both until return.
	held := int64(0)
	for _, v := range x.parameters {
		held += 8 * int64(len(v))
	}
	held += 4*int64(len(batch.NormalizedLatents)+len(batch.VoiceLatents)+len(x.noise)+len(x.diagonal)+len(x.s)+len(x.end)+len(batch.TextTokens)) + int64(len(x.mask))
	candidate := map[string][]float32{}
	for n, z := range f {
		candidate[n] = make([]float32, z)
		held += 4 * int64(z)
	}
	packed := make([]float32, 12)
	for row := 0; row < 3; row++ {
		packed[2*row], packed[2*row+1] = x.diagonal[row], x.diagonal[row]
		packed[6+2*row], packed[7+2*row] = x.s[row], x.end[row]
	}
	wp, e := weight.BuildWeightTrainingTape(packed, make([]float32, 6), 6, 1<<20)
	if e != nil {
		return nil, e
	}
	wi := map[string]int{}
	for n, z := range l {
		if strings.HasPrefix(n, "flow.w_s_t.") {
			wi[n] = z
		}
	}
	wo, e := pocketJoinRun(ctx, "weight-forward-zero-seed", wp.Tape, wp.ResidentBindings, wi, held+72, trace)
	if e != nil {
		return nil, e
	}
	wr, e := wp.Decode(wo)
	if e != nil {
		return nil, e
	}
	logvars := wr.Output
	held += 24
	wp = WeightTrainingTape{}
	wo = nil
	wr = WeightTrainingResult{}
	packed = nil
	rows := 8
	seed := make([]float32, rows*lm.Hidden)
	bp, e := lm.BuildBackboneTrainingTape(batch, seed, 1<<20)
	if e != nil {
		return nil, e
	}
	bi := map[string]int{}
	for n, z := range l {
		if strings.HasPrefix(n, "flow_lm.") && !strings.HasPrefix(n, "flow_lm.out_eos.") {
			bi[n] = z
		}
	}
	bo, e := pocketJoinRun(ctx, "backbone-forward-zero-seed", bp.Tape, bp.ResidentBindings, bi, held+128, trace)
	if e != nil {
		return nil, e
	}
	br := BackboneTrainingResult{Sequence: bo[0], Output: bo[1], DVoice: bo[2], DNormalized: bo[3]}
	if e = validateBackboneResult(br, lm, batch, seed); e != nil {
		return nil, e
	}
	hidden := append([]float32(nil), br.Output[5*lm.Hidden:]...)
	held += 48
	bp = BackboneTrainingTape{}
	bo = nil
	br = BackboneTrainingResult{}
	seed = nil
	ep, e := lm.BuildEOSProjectionTrainingTape(hidden, x.mask, cfg.EOSLossWeight, 1<<20)
	if e != nil {
		return nil, e
	}
	ei := map[string]int{"flow_lm.out_eos.weight": 4, "flow_lm.out_eos.bias": 1}
	eo, e := pocketJoinRun(ctx, "eos-forward-only-candidate-excluded", ep.Tape, ep.ResidentBindings, ei, held, trace)
	if e != nil {
		return nil, e
	}
	er := EOSProjectionTrainingResult{Loss: float64(eo[0][0]), DInput: eo[1]}
	if e = validateEOSProjectionBoundary(er, 3, 4, hidden); e != nil {
		return nil, e
	}
	in := StepReductionInputs{Frames: 3, Hidden: 4, Latent: 2, Mask: x.mask, DiagonalCondition: make([]float32, 12), DistillCondition: make([]float32, 12), DiagonalTarget: make([]float32, 6), DistillTarget: make([]float32, 6), DiagonalLogvar: make([]float32, 3), DistillLogvar: make([]float32, 3), RowMetrics: make([]float32, 12), EOS: er.Loss}
	held += 216
	ep = ProjectionTrainingTape{}
	eo = nil
	er = EOSProjectionTrainingResult{}
	for row := 0; row < 2; row++ {
		condition := hidden[row*4 : (row+1)*4]
		target := batch.NormalizedLatents[row*2 : (row+1)*2]
		noise := x.noise[row*2 : (row+1)*2]
		di := DiagonalForwardInputs{Condition: condition, Target: target, Noise: noise, Time: x.diagonal[row]}
		dp, e := flow.BuildDiagonalForwardTape(di, 1<<20)
		if e != nil {
			return nil, e
		}
		dfo, e := pocketJoinRun(ctx, fmt.Sprintf("row%d-diagonal-forward", row), dp.Tape, dp.ResidentBindings, f, held, trace)
		if e != nil {
			return nil, e
		}
		df := DiagonalForwardResult{Input: dfo[0], Desired: dfo[1], Prediction: dfo[2]}
		if e = validateDiagonalForwardResult(df, di); e != nil {
			return nil, e
		}
		dp = DiagonalForwardTape{}
		dfo = nil
		db, e := flow.BuildDiagonalTrainingTape(condition, di.Time, df.Input, df.Prediction, df.Desired, logvars[row], .375, 1<<20)
		if e != nil {
			return nil, e
		}
		dbo, e := pocketJoinRun(ctx, fmt.Sprintf("row%d-diagonal-backward", row), db.Tape, db.ResidentBindings, f, held+24, trace)
		if e != nil {
			return nil, e
		}
		dr, e := decodeDiagonalBoundary(dbo[:3], condition, df.Input, df.Prediction, df.Desired)
		if e != nil {
			return nil, e
		}
		dg, e := pocketJoinGradients(db.Tape, db.ResidentBindings, dbo, f)
		if e != nil {
			return nil, e
		}
		if e = pocketJoinAccumulate(candidate, dg, f, 1); e != nil {
			return nil, e
		}
		trace.Contributions = append(trace.Contributions, fmt.Sprintf("row%d diagonal: 34/386, seed .375, accumulate 1", row))
		copy(in.DiagonalCondition[row*4:], dr.DCondition)
		copy(in.DiagonalTarget[row*2:], dr.DTarget)
		in.DiagonalLogvar[row] = dr.DLogVariance
		in.RowMetrics[row*4], in.RowMetrics[row*4+1] = float32(dr.Loss), float32(dr.RawSquare)
		db = DiagonalTrainingTape{}
		dbo = nil
		dg = nil
		df = DiagonalForwardResult{}
		fi := DistillForwardInputs{Condition: condition, Target: target, Noise: noise, S: x.s[row], T: x.end[row]}
		fp, e := flow.BuildDistillForwardTape(fi, 1<<20)
		if e != nil {
			return nil, e
		}
		fo, e := pocketJoinRun(ctx, fmt.Sprintf("row%d-distill-forward", row), fp.Tape, fp.ResidentBindings, f, held, trace)
		if e != nil {
			return nil, e
		}
		fr, e := decodeDistillForwardBoundary(fo[:6], fi)
		if e != nil {
			return nil, e
		}
		fp = DistillForwardTape{}
		fo = nil
		ti := DistillTrainingInputs{Condition: condition, PrimaryInput: fr.PrimaryInput, EndpointInput: fr.EndpointInput, Prediction: fr.Prediction, Desired: fr.Desired, Velocity: fr.Velocity, Tangent: fr.Tangent, Target: target, Noise: noise, S: fi.S, T: fi.T, LogVariance: logvars[3+row]}
		tp, e := flow.BuildDistillTrainingTape(ti, 1<<20)
		if e != nil {
			return nil, e
		}
		to, e := pocketJoinRun(ctx, fmt.Sprintf("row%d-distill-backward-primary-only", row), tp.Tape, tp.ResidentBindings, f, held+48, trace)
		if e != nil {
			return nil, e
		}
		tr, e := decodeDistillBoundary(to[:5], ti)
		if e != nil {
			return nil, e
		}
		tg, e := pocketJoinGradients(tp.Tape, tp.ResidentBindings, to, f)
		if e != nil {
			return nil, e
		}
		if e = pocketJoinAccumulate(candidate, tg, f, .125); e != nil {
			return nil, e
		}
		trace.Contributions = append(trace.Contributions, fmt.Sprintf("row%d distill: primary-only 34/386, raw, accumulate .125; endpoint G excluded", row))
		copy(in.DistillCondition[row*4:], tr.DCondition)
		copy(in.DistillTarget[row*2:], tr.DTarget)
		in.DistillLogvar[row] = tr.DLogVariance
		in.RowMetrics[row*4+2], in.RowMetrics[row*4+3] = float32(tr.Loss), float32(tr.RawSquare)
	}
	// Raw reduction seeds are computed here, not imported from expected arrays.
	samples := WeightSampleInputs{DiagonalTime: x.diagonal, DistillS: x.s, DistillT: x.end, DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	lp, e := lm.BuildStepBackboneTrainingTape(weight, batch, hidden, in, cfg, samples, 1<<20)
	if e != nil {
		return nil, e
	}
	lp, e = buildStepResultsTape(lp, batch, in, 1<<20)
	if e != nil {
		return nil, e
	}
	roots := stepResultsOutputs(lp, batch)
	if e = adapterStepRootRoles(lp, batch, in, samples, roots); e != nil {
		return nil, e
	}
	lo, e := pocketJoinRun(ctx, "later-reduction-weight-eos-backbone-finalinput", lp.Tape, lp.ResidentBindings, l, held, trace)
	if e != nil {
		return nil, e
	}
	selected, e := pocketJoinSelected(lp.Tape, lo, roots)
	if e != nil {
		return nil, e
	}
	later, e := decodeStepResultsBoundary(selected, batch, hidden, in, samples)
	if e != nil {
		return nil, e
	}
	lg, e := pocketJoinGradients(lp.Tape, lp.ResidentBindings, lo, l)
	if e != nil {
		return nil, e
	}
	// Disjoint family union: a later name must never overwrite early flow G.
	for n, v := range lg {
		if _, ok := candidate[n]; ok {
			return nil, fmt.Errorf("join duplicate contribution %s", n)
		}
		candidate[n] = append([]float32(nil), v...)
	}
	all, e := pocketJoinUnion(f, l)
	if e != nil {
		return nil, e
	}
	if e = pocketJoinArrayInventory(candidate, all); e != nil {
		return nil, e
	}
	trace.Contributions = append(trace.Contributions, "later: disjoint 25/220, unit accumulation; EOS .1 once; weight seeds .375/.125 once")
	return &pocketJoinResult{Later: later, G: candidate, DiagonalLogvar: append([]float32(nil), logvars[:3]...), DistillLogvar: append([]float32(nil), logvars[3:]...)}, nil
}
func TestPocketComputedFullStepJoin(t *testing.T) {
	x, o := loadPocketJoinFixture(t)
	frozen, _ := loadPocketJoinFixture(t)
	trace := &pocketJoinTrace{}
	r, e := computePocketJoin(t, context.Background(), x, trace)
	if !reflect.DeepEqual(x, frozen) {
		t.Fatal("join mutated frozen P/buffers/batch/samples/config")
	}
	for _, p := range trace.Phases {
		t.Logf("phase=%s tensors=%d commands=%d logicalSource=%d sourceConstants=%d heldSourceBoundaryCandidate=%d inventoryPayload=%d adapter=%+v operations=%v", p.Name, p.Tensors, p.Commands, p.LogicalSource, p.SourceConstants, p.HeldPayload, p.InventoryPayload, p.Adapter, p.Operations)
	}
	t.Logf("contribution order=%v", trace.Contributions)
	if e != nil {
		if r != nil {
			t.Fatal("partial join published")
		}
		t.Fatal("first join failure (no split/bypass/limit change):", e)
	}
	wantPhases := []string{"weight-forward-zero-seed", "backbone-forward-zero-seed", "eos-forward-only-candidate-excluded", "row0-diagonal-forward", "row0-diagonal-backward", "row0-distill-forward", "row0-distill-backward-primary-only", "row1-diagonal-forward", "row1-diagonal-backward", "row1-distill-forward", "row1-distill-backward-primary-only", "later-reduction-weight-eos-backbone-finalinput"}
	if len(trace.Phases) != len(wantPhases) || len(trace.Contributions) != 5 {
		t.Fatal("join phase/contribution count")
	}
	for i, name := range wantPhases {
		if trace.Phases[i].Name != name || trace.Phases[i].Adapter.Dispatched != trace.Phases[i].Commands {
			t.Fatal("join ordered computation/complete dispatch", i)
		}
	}
	var metrics, derivatives, fg, lmg, wg, diagnostic pocketAdapterErrors
	m := r.Later.Reduction.Metrics
	for i, pair := range [][2]float64{{m.FlowDiagonal, o.Metrics["flow_diag"]}, {m.FlowDistill, o.Metrics["flow_distill"]}, {m.FlowLoss, o.Metrics["flow_loss"]}, {m.EOS, o.Metrics["eos_loss"]}, {m.Loss, o.Metrics["loss"]}} {
		assertClose64(t, fmt.Sprintf("join metric%d", i), pair[0], pair[1], 3e-5)
		d := math.Abs(pair[0] - pair[1])
		metrics.count++
		metrics.sum += d
		metrics.squares += d * d
		metrics.maximum = math.Max(metrics.maximum, d)
	}
	derivatives.compare(t, "diagonal logvar VJP", r.Later.Reduction.DiagonalLogvar[:2], o.DDiagonalLogVariance, 5e-5)
	derivatives.compare(t, "distill logvar VJP", r.Later.Reduction.DistillLogvar[:2], o.DDistillLogVariance, 5e-5)
	if r.Later.Reduction.DiagonalLogvar[2] != 0 || r.Later.Reduction.DistillLogvar[2] != 0 {
		t.Fatal("padded logvar VJP")
	}
	derivatives.compare(t, "audio VJP", r.Later.Inputs.NormalizedLatents, o.DNormalizedLatents, 5e-5)
	derivatives.compare(t, "voice VJP", r.Later.Inputs.VoiceLatents, o.DVoiceLatents, 5e-5)
	names := make([]string, 0, len(r.G))
	for n := range r.G {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		stats := &lmg
		if strings.HasPrefix(n, "flow_lm.flow_net.") {
			stats = &fg
		} else if strings.HasPrefix(n, "flow.w_s_t.") {
			stats = &wg
		}
		stats.compare(t, n, r.G[n], o.Gradients[n], 8e-5)
	}
	// Diagnostic distribution only: no new forward-logvar acceptance tolerance.
	for branch, got := range [][]float32{r.DiagonalLogvar[:2], r.DistillLogvar[:2]} {
		want := [][]float32{o.DiagonalLogVariance, o.DistillLogVariance}[branch]
		for i, z := range got {
			if !isFinite(z) {
				t.Fatal("nonfinite computed forward logvar")
			}
			d := math.Abs(float64(z) - float64(want[i]))
			diagnostic.count++
			diagnostic.sum += d
			diagnostic.squares += d * d
			diagnostic.maximum = math.Max(diagnostic.maximum, d)
		}
	}
	metrics.log(t, "computed fullstep metrics")
	derivatives.log(t, "computed fullstep derivatives")
	fg.log(t, "computed flow 34/386")
	lmg.log(t, "computed FlowLM/EOS 19/195")
	wg.log(t, "computed weighting 6/25")
	diagnostic.log(t, "forward logvars DIAGNOSTIC ONLY")
}
