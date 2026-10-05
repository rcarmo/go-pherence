package pockettts

// TEST ONLY: Pocket-local ownership state machine. No ResidentState, optimiser,
// checkpoint, native adapter, exported helper or production call site.
import (
	"context"
	"fmt"
	"math"
	"sort"

	train "github.com/rcarmo/go-pherence/backends/training"
)

const pocketGroupLimit int64 = 1 << 20

type pocketGroupState struct {
	p, g           map[string][]float32
	ready          map[string]bool
	active, closed bool
	terminal       error
	retained       []*pocketGroupOwner
}

func (s *pocketGroupState) admit(ctx context.Context) error {
	if s == nil || ctx == nil {
		return fmt.Errorf("group nil state/context")
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	if s.active || s.closed || s.terminal != nil {
		return fmt.Errorf("group fenced state")
	}
	return nil
}

// Test-only fault-fence probes; no optimiser or checkpoint implementation.
func (s *pocketGroupState) probe(ctx context.Context, op string) error {
	if e := s.admit(ctx); e != nil {
		return e
	}
	if op == "close" {
		s.closed = true
	}
	return nil
}

type pocketGroupPolicy struct { // Value-owned role/extent contract; no caller closure.
	kind    string
	roles   []string
	extents []int
	step    *pocketGroupStepPolicy
}
type pocketGroupStepPolicy struct {
	batch  FlowLMTrainingBatch
	in     StepReductionInputs
	hidden []float32
	sample WeightSampleInputs
	cfg    TrainingStepConfig
}
type pocketGroupPlan struct {
	tape     train.Tape
	bound    map[int][]float32 // all surviving P, independent of selected G names
	bindings map[string]train.ResidentTapeBinding
	compact  map[string]train.ResidentCompactEmbeddingBinding
	names    []string
	results  []int
	storage  train.TapeStoragePlan
}
type pocketGroupLedger struct {
	OperationFrozen, FrozenP, RetainedPlanMetadata, MetadataReservation int64
	CandidatePayload, PublishedPayload, AccumulateScratch               int64
	Commands, Plans, FreezeCopies, Constructed, Closed, MaxLive         int
	Groups                                                              []pocketAdapterLedger
}
type pocketGroupOwner struct {
	plan             *pocketGroupPlan // mock lifetime retains its admitted plan/frozen borrows
	outputs          [][]float32
	closed, retained bool
}
type pocketGroupLocal struct {
	state                 *pocketGroupState
	callback              func(string, int, int)
	fault                 string
	faultGroup, faultItem int
	nilOwner              bool
	corrupt               func(int, [][]float32) // explicit test-only output fault injection
	metadataLimit         int64                  // smaller rejection fixture only; never > current ceiling
	ledger                pocketGroupLedger
	live                  int
}

func (p *pocketGroupLocal) stage(ctx context.Context, stage string, group, item int) error {
	if p.callback != nil {
		p.callback(stage, group, item)
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	if p.fault == stage && p.faultGroup == group && (p.faultItem < 0 || p.faultItem == item) {
		return fmt.Errorf("group %d %s[%d] fault", group, stage, item)
	}
	return nil
}
func pocketGroupAdd(total *int64, n, limit int64) error {
	if n < 0 || limit <= 0 || *total < 0 || *total > limit || n > limit-*total {
		return fmt.Errorf("group metadata byte ceiling: reserved=%d next=%d limit=%d", *total, n, limit)
	}
	*total += n
	return nil
}

// Retained structs plus nested copied arrays. Scratch has its own conservative
// high-water reservation; retained plans are not confused with adapter slots.
func pocketGroupMetadata(t train.Tape) (int64, error) {
	if len(t.Tensors) == 0 || len(t.Tensors) > 4096 || len(t.Commands) > 4096 || len(t.Outputs) == 0 || len(t.Outputs) > 4096 {
		return 0, fmt.Errorf("group tape cardinality")
	}
	n := int64(len(t.Tensors))*64 + int64(len(t.Commands))*96 + int64(len(t.Outputs))*8
	for _, s := range t.Tensors {
		if len(s.Shape) < 1 || len(s.Shape) > 3 {
			return 0, fmt.Errorf("group nested shape")
		}
		n += int64(len(s.Shape)) * 8
	}
	for _, c := range t.Commands {
		if len(c.Inputs) > 16 || len(c.Op) > 64 || !pocketAdapterTuple(c) {
			return 0, fmt.Errorf("group tuple/operand bound")
		}
		n += int64(len(c.Inputs))*8 + int64(len(c.Op))
	}
	return n, nil
}
func pocketGroupPayload(t train.Tape) int64 {
	var n int64
	for _, s := range t.Tensors {
		n += int64(len(s.Values)) * 4
	}
	return n
}
func pocketGroupFreezePolicy(v pocketGroupPolicy) pocketGroupPolicy {
	v.roles = append([]string(nil), v.roles...)
	v.extents = append([]int(nil), v.extents...)
	if v.step != nil {
		x := *v.step
		x.batch.NormalizedLatents = append([]float32(nil), x.batch.NormalizedLatents...)
		x.batch.VoiceLatents = append([]float32(nil), x.batch.VoiceLatents...)
		x.batch.TextTokens = append([]uint32(nil), x.batch.TextTokens...)
		x.hidden = append([]float32(nil), x.hidden...)
		x.sample.DiagonalSeed = append([]float32(nil), x.sample.DiagonalSeed...)
		x.sample.DistillSeed = append([]float32(nil), x.sample.DistillSeed...)
		x.sample.DiagonalTime = append([]float32(nil), x.sample.DiagonalTime...)
		x.sample.DistillS = append([]float32(nil), x.sample.DistillS...)
		x.sample.DistillT = append([]float32(nil), x.sample.DistillT...)
		x.in.Mask = append([]bool(nil), x.in.Mask...)
		x.in.DiagonalCondition = append([]float32(nil), x.in.DiagonalCondition...)
		x.in.DistillCondition = append([]float32(nil), x.in.DistillCondition...)
		x.in.DiagonalTarget = append([]float32(nil), x.in.DiagonalTarget...)
		x.in.DistillTarget = append([]float32(nil), x.in.DistillTarget...)
		x.in.DiagonalLogvar = append([]float32(nil), x.in.DiagonalLogvar...)
		x.in.DistillLogvar = append([]float32(nil), x.in.DistillLogvar...)
		x.in.RowMetrics = append([]float32(nil), x.in.RowMetrics...)
		v.step = &x
	}
	return v
}
func (v pocketGroupPolicy) admit(t train.Tape, roots []int) error {
	if len(roots) < 1 || len(roots) > 16 || len(v.roles) != len(roots) || len(v.extents) != len(roots) {
		return fmt.Errorf("group result policy count")
	}
	declared := map[int]int{}
	for _, id := range t.Outputs {
		declared[id]++
	}
	seen := map[int]bool{}
	roles := map[string]bool{}
	for i, id := range roots {
		if id < 0 || id >= len(t.Tensors) || declared[id] != 1 || seen[id] || roles[v.roles[i]] || len(v.roles[i]) == 0 || len(v.roles[i]) > 64 {
			return fmt.Errorf("group result policy root/role")
		}
		seen[id] = true
		roles[v.roles[i]] = true
		n, e := t.Tensors[id].Elements()
		if e != nil || n != v.extents[i] {
			return fmt.Errorf("group result policy extent")
		}
	}
	switch v.kind {
	case "component":
		if v.step != nil {
			return fmt.Errorf("group component policy")
		}
	case "step":
		if v.step == nil {
			return fmt.Errorf("group missing typed step")
		}
		x := v.step
		if x.batch.Frames != x.in.Frames || x.in.Frames < 1 || x.in.Frames > 1024 || x.in.Hidden < 1 || x.in.Hidden > 16384 || x.in.Latent < 1 || x.in.Latent > 16384 || x.batch.VoiceFrames < 0 || x.batch.VoiceFrames > 1024 || len(x.batch.NormalizedLatents) != x.in.Frames*x.in.Latent || len(x.batch.VoiceLatents) != x.batch.VoiceFrames*x.in.Latent || len(x.hidden) != x.in.Frames*x.in.Hidden || len(x.batch.TextTokens) > 4096 || !finiteF32(x.hidden) || !finiteF32(x.batch.NormalizedLatents) || !finiteF32(x.batch.VoiceLatents) {
			return fmt.Errorf("group typed geometry")
		}
		if _, e := validateStepReductionInputs(x.in, x.cfg); e != nil {
			return e
		}
		if e := x.sample.validate(2 * x.in.Frames); e != nil {
			return e
		}
		want := []string{"diagonal-logvar", "distill-logvar", "metrics"}
		if v.step.batch.VoiceFrames > 0 {
			want = append(want, "voice-vjp")
		}
		want = append(want, "normalized-vjp", "direct-tail")
		if len(want) != len(v.roles) {
			return fmt.Errorf("group step role count")
		}
		for i, n := range want {
			if n != v.roles[i] {
				return fmt.Errorf("group step role order")
			}
		}
	default:
		return fmt.Errorf("group unknown policy")
	}
	return nil
}
func (v pocketGroupPolicy) check(values [][]float32) error {
	if len(values) != len(v.extents) {
		return fmt.Errorf("group result count")
	}
	for i, x := range values {
		if len(x) != v.extents[i] || !finiteF32(x) {
			return fmt.Errorf("group result extent/finite")
		}
		for _, y := range values[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("group result alias")
			}
		}
	}
	if v.kind == "step" {
		_, e := decodeStepResultsBoundary(values, v.step.batch, v.step.hidden, v.step.in, v.step.sample)
		return e
	}
	return nil
}
func pocketGroupBindings(t train.Tape, b map[string]train.ResidentTapeBinding, c map[string]train.ResidentCompactEmbeddingBinding) error {
	if len(c) > 0 {
		return train.ValidateResidentCompactTapeBindings(t, b, c, pocketGroupLimit)
	}
	return train.ValidateResidentTapeBindings(t, b, pocketGroupLimit)
}

func (p *pocketGroupLocal) run(ctx context.Context, source train.Tape, all map[string]train.ResidentTapeBinding, compact map[string]train.ResidentCompactEmbeddingBinding, results []int, groups [][]string, scale float32, policy pocketGroupPolicy) (published [][]float32, err error) {
	if p == nil {
		return nil, fmt.Errorf("group nil coordinator")
	}
	s := p.state
	if e := s.admit(ctx); e != nil {
		return nil, e
	}
	if !isFinite(scale) {
		return nil, fmt.Errorf("group scale")
	}
	s.active = true
	defer func() { s.active = false }()
	p.ledger = pocketGroupLedger{}
	p.live = 0
	limit := p.metadataLimit
	if limit == 0 {
		limit = pocketGroupLimit
	}
	if limit > pocketGroupLimit {
		return nil, fmt.Errorf("group ceiling override forbidden")
	}
	base, e := pocketGroupMetadata(source)
	if e != nil {
		return nil, e
	}
	if len(s.ready) > 4096 || len(s.retained) > 4096 || s.ready == nil || len(all) == 0 || len(all) > 4096 || len(compact) > 4096 || len(results) == 0 || len(results) > 16 || len(groups) == 0 || len(groups) > 128 || len(s.p) != len(all) || len(s.g) != len(all) || len(policy.roles) != len(results) || len(policy.extents) != len(results) {
		return nil, fmt.Errorf("group upfront inventory cardinality")
	}
	// Reserve effective headers AND second validation scratch BEFORE any copy.
	// Public pure planners need temporary scratch. Keep its high-water reserve
	// alongside ALL retained plans; do not allocate/plan another group uncharged.
	reserve := base + int64(len(source.Tensors))*144 + int64(len(source.Tensors)+len(source.Commands))*256 + int64(len(results))*16 + int64(len(groups))*64 + int64(len(all)+len(compact))*256 + int64(len(source.Outputs)+len(all))*8
	for name := range all {
		if len(name) < 1 || len(name) > 512 {
			return nil, fmt.Errorf("group name bound")
		}
		reserve += int64(len(name)) * 2
	}
	for name := range compact {
		if len(name) < 1 || len(name) > 512 {
			return nil, fmt.Errorf("group compact name bound")
		}
		reserve += int64(len(name))
	}
	for _, g := range groups {
		if len(g) == 0 || len(g) > 8 {
			return nil, fmt.Errorf("group size")
		}
		for _, n := range g {
			if len(n) < 1 || len(n) > 512 {
				return nil, fmt.Errorf("group selected name")
			}
			reserve += 32 + int64(len(n))
		}
	}
	for _, role := range policy.roles {
		if len(role) > 64 {
			return nil, fmt.Errorf("group role bound")
		}
		reserve += 64 + int64(len(role))
	}
	if e = pocketGroupAdd(&p.ledger.MetadataReservation, reserve, limit); e != nil {
		return nil, e
	}
	if e = pocketGroupBindings(source, all, compact); e != nil {
		return nil, e
	}
	if e = policy.admit(source, results); e != nil {
		return nil, e
	}
	counts := map[int]int{}
	for _, id := range source.Outputs {
		counts[id]++
		if counts[id] != 1 {
			return nil, fmt.Errorf("group duplicate source output")
		}
	}
	for _, b := range all {
		if counts[b.Gradient] != 1 {
			return nil, fmt.Errorf("group exact declared G output")
		}
	}
	if policy.kind == "step" {
		x := policy.step
		need := 6
		if x.batch.VoiceFrames > 0 {
			need = 7
		}
		if len(source.Outputs) < need {
			return nil, fmt.Errorf("group malformed step outputs")
		}
		if e = adapterStepRootRoles(StepBackboneTrainingTape{Tape: source, ResidentBindings: all}, x.batch, x.in, x.sample, results); e != nil {
			return nil, e
		}
	}
	// Bound typed-policy payload before copying any of its arrays.
	var policyBytes int64
	if policy.step != nil {
		x := policy.step
		policyBytes = int64(len(x.batch.NormalizedLatents)+len(x.batch.VoiceLatents)+len(x.in.DiagonalCondition)+len(x.in.DistillCondition)+len(x.in.DiagonalTarget)+len(x.in.DistillTarget)+len(x.in.DiagonalLogvar)+len(x.in.DistillLogvar)+len(x.in.RowMetrics)+len(x.hidden)+len(x.sample.DiagonalSeed)+len(x.sample.DistillSeed)+len(x.sample.DiagonalTime)+len(x.sample.DistillS)+len(x.sample.DistillT)) * 4
		policyBytes += int64(len(x.batch.TextTokens))*4 + int64(len(x.in.Mask))
		if policyBytes > pocketGroupLimit {
			return nil, fmt.Errorf("group typed policy payload")
		}
		if e = pocketGroupAdd(&p.ledger.MetadataReservation, int64(len(x.batch.TextTokens))*4+int64(len(x.in.Mask))+1024, limit); e != nil {
			return nil, e
		}
	}
	// Admit the complete combined payload and mock inventories before any
	// operation-wide source/P/policy copy, not cumulatively after allocations.
	var pBytes int64
	for name, b := range all {
		n, _ := source.Tensors[b.Parameter].Elements()
		pv, ok := s.p[name]
		gv, gok := s.g[name]
		if !ok || !gok || len(pv) != n || len(gv) != n || !finiteF32(pv) || !finiteF32(gv) {
			return nil, fmt.Errorf("group P/G admission")
		}
		pBytes += int64(n) * 4
		// Stable writable G must never mutate ANY live parameter backing,
		// including same-name, cross-name and partially overlapping views.
		// Frozen P only protects this invocation's reads, not the live state.
		for _, liveP := range s.p {
			if f32SlicesOverlap(gv, liveP) {
				return nil, fmt.Errorf("group live G/P alias")
			}
		}
		for other, og := range s.g {
			if other != name && f32SlicesOverlap(gv, og) {
				return nil, fmt.Errorf("group G alias")
			}
		}
	}
	if pBytes+pocketGroupPayload(source)+policyBytes > pocketGroupLimit {
		return nil, fmt.Errorf("group frozen payload ceiling")
	}
	frozen := pocketAdapterFreeze(source)
	p.ledger.FreezeCopies++
	p.ledger.OperationFrozen = pocketGroupPayload(frozen) + policyBytes
	allCopy := map[string]train.ResidentTapeBinding{}
	cc := map[string]train.ResidentCompactEmbeddingBinding{}
	for n, v := range all {
		allCopy[n] = v
	}
	for n, v := range compact {
		cc[n] = v
	}
	all, compact = allCopy, cc
	results = append([]int(nil), results...)
	fg := make([][]string, len(groups))
	for i, g := range groups {
		fg[i] = append([]string(nil), g...)
	}
	groups = fg
	policy = pocketGroupFreezePolicy(policy)
	allP := map[int][]float32{}
	stableG := map[string][]float32{} // frozen headers; never re-resolve after callbacks
	seenG := map[int]bool{}
	for name, b := range all {
		n, _ := frozen.Tensors[b.Parameter].Elements()
		pv, ok := s.p[name]
		gv, gok := s.g[name]
		if !ok || !gok || len(pv) != n || len(gv) != n || !finiteF32(pv) || !finiteF32(gv) || seenG[b.Gradient] {
			return nil, fmt.Errorf("group P/G inventory/extent")
		}
		for other, otherG := range s.g {
			if other != name && f32SlicesOverlap(gv, otherG) {
				return nil, fmt.Errorf("group G alias")
			}
		}
		seenG[b.Gradient] = true
		p.ledger.FrozenP += int64(n) * 4
		if p.ledger.FrozenP+p.ledger.OperationFrozen > pocketGroupLimit {
			return nil, fmt.Errorf("group frozen payload ceiling")
		}
		allP[b.Parameter] = append([]float32(nil), pv...)
		stableG[name] = gv
	}
	if e = train.ValidateExternalTapeBindings(frozen, allP); e != nil {
		return nil, e
	}
	effective := frozen
	effective.Tensors = append([]train.TensorSpec(nil), frozen.Tensors...)
	for id, v := range allP {
		effective.Tensors[id].Values = v
		effective.Tensors[id].External = false
	}
	if e = effective.Validate(pocketGroupLimit); e != nil {
		return nil, fmt.Errorf("group effective source: %w", e)
	}
	// PruneTape accepts only declared roots. Selected P must survive even
	// if that gradient has no P dependency; declare only in this owned closure
	// view, after admitting the unchanged complete original/effective source.
	closure := frozen
	closure.Outputs = append([]int(nil), frozen.Outputs...)
	declared := map[int]bool{}
	for _, id := range closure.Outputs {
		declared[id] = true
	}
	for _, b := range all {
		if !declared[b.Parameter] {
			closure.Outputs = append(closure.Outputs, b.Parameter)
			declared[b.Parameter] = true
		}
	}
	if len(closure.Outputs) > 4096 {
		return nil, fmt.Errorf("group closure output ceiling")
	}
	plans := make([]pocketGroupPlan, 0, len(groups))
	seen := map[string]bool{}
	for gi, names := range groups {
		sort.Strings(names)
		roots := append([]int(nil), results...)
		for _, name := range names {
			b, ok := all[name]
			if !ok || seen[name] {
				return nil, fmt.Errorf("group duplicate/foreign G")
			}
			seen[name] = true
			roots = append(roots, b.Gradient, b.Parameter)
			if c, ok := compact[name]; ok {
				roots = append(roots, c.Index)
			}
		}
		rootSet := map[int]bool{}
		for _, id := range roots {
			if rootSet[id] {
				return nil, fmt.Errorf("group duplicate result/P/G/index union")
			}
			rootSet[id] = true
		}
		// Conservative full-source upper bound on retained pruned metadata and maps.
		// Scratch was reserved above. No one-group-at-a-time replanning contract.
		reservation := base + int64(len(frozen.Tensors))*16 + int64(len(allP)+len(names))*128 + int64(len(roots)+len(results))*8 + 256
		if e = pocketGroupAdd(&p.ledger.MetadataReservation, reservation, limit); e != nil {
			return nil, e
		}
		p.ledger.RetainedPlanMetadata += reservation
		pruned, remap, e := train.PruneTape(closure, roots, pocketGroupLimit)
		if e != nil {
			return nil, e
		}
		q := pocketGroupPlan{tape: pruned, bound: map[int][]float32{}, bindings: map[string]train.ResidentTapeBinding{}, compact: map[string]train.ResidentCompactEmbeddingBinding{}, names: names, results: make([]int, len(results))}
		protected := map[int]bool{}
		for id, v := range allP {
			if remap[id] >= 0 {
				q.bound[remap[id]] = v
				protected[remap[id]] = true
			}
		}
		for i, id := range results {
			q.results[i] = remap[id]
		}
		for _, name := range names {
			b := all[name]
			q.bindings[name] = train.ResidentTapeBinding{Parameter: remap[b.Parameter], Gradient: remap[b.Gradient]}
			if c, ok := compact[name]; ok {
				c.Index = remap[c.Index]
				q.compact[name] = c
			}
		}
		if e = pocketGroupBindings(pruned, q.bindings, q.compact); e != nil {
			return nil, e
		}
		if e = policy.admit(pruned, q.results); e != nil {
			return nil, e
		}
		if e = train.ValidateExternalTapeBindings(pruned, q.bound); e != nil {
			return nil, e
		}
		eff := pruned
		eff.Tensors = append([]train.TensorSpec(nil), pruned.Tensors...)
		for id, v := range q.bound {
			eff.Tensors[id].Values = v
			eff.Tensors[id].External = false
		}
		if e = eff.Validate(pocketGroupLimit); e != nil {
			return nil, e
		}
		if _, e = pocketGroupMetadata(pruned); e != nil {
			return nil, e
		}
		// Mirror the accepted adapter's complete metadata reservation here so
		// no later group discovers an adapter limit only after prior commits.
		adapterMeta := int64(len(pruned.Tensors))*64 + int64(len(pruned.Commands))*32 + int64(len(pruned.Tensors)+len(pruned.Commands))*256 + int64(len(pruned.Outputs))*8 + int64(len(q.bound))*128 + int64(len(pruned.Tensors))*144
		for _, s := range pruned.Tensors {
			adapterMeta += int64(len(s.Shape)) * 8
		}
		for _, c := range pruned.Commands {
			adapterMeta += int64(len(c.Inputs))*8 + int64(len(c.Op))
		}
		if adapterMeta > pocketGroupLimit {
			return nil, fmt.Errorf("group adapter metadata ceiling")
		}
		q.storage, e = train.PlanZeroTapeStorage(pruned, pocketGroupLimit, protected)
		if e != nil {
			return nil, e
		}
		var outBytes int64
		for _, id := range pruned.Outputs {
			n, _ := pruned.Tensors[id].Elements()
			outBytes += 4 * int64(n)
		}
		if outBytes > pocketGroupLimit {
			return nil, fmt.Errorf("group planned output ceiling")
		}
		p.ledger.Commands += len(pruned.Commands)
		if p.ledger.Commands > 65536 {
			return nil, fmt.Errorf("group work bound")
		}
		p.ledger.Plans++
		if e = p.stage(ctx, "plan", gi, -1); e != nil {
			return nil, e
		}
		plans = append(plans, q)
	}
	if len(seen) != len(all) {
		return nil, fmt.Errorf("group missing exact G union")
	}
	var owner *pocketGroupOwner
	progress := false
	fail := func(e error) ([][]float32, error) {
		if progress || owner != nil {
			s.terminal = e
			if owner != nil && !owner.closed && !owner.retained {
				owner.retained = true
				s.retained = append(s.retained, owner)
			}
		}
		return nil, e
	}
	var candidate [][]float32
	for gi, q := range plans {
		if e = p.stage(ctx, "construct", gi, -1); e != nil {
			if !p.nilOwner && ctx.Err() == nil {
				owner = &pocketGroupOwner{plan: &q}
				p.live++
			}
			return fail(e)
		}
		if p.nilOwner && p.fault == "nil" && p.faultGroup == gi {
			return fail(fmt.Errorf("group nil owner"))
		}
		owner = &pocketGroupOwner{plan: &q}
		p.live++
		p.ledger.Constructed++
		p.ledger.MaxLive = max(p.ledger.MaxLive, p.live)
		if e = p.stage(ctx, "constructed", gi, -1); e != nil {
			return fail(e)
		}
		if e = p.stage(ctx, "run", gi, -1); e != nil {
			return fail(e)
		}
		progress = true
		values, ledger, e := runPocketAdapter(ctx, q.tape, q.bound, nil, pocketAdapterOptions{reuse: true, shareZero: true, before: func(i int) {
			if p.callback != nil {
				p.callback("dispatch", gi, i)
			}
		}, after: func(i int) {
			if p.callback != nil {
				p.callback("dispatched", gi, i)
			}
		}})
		p.ledger.Groups = append(p.ledger.Groups, ledger)
		owner.outputs = values
		if e != nil {
			return fail(e)
		}
		if p.corrupt != nil {
			p.corrupt(gi, values)
		}
		if e = p.stage(ctx, "witness", gi, -1); e != nil {
			return fail(e)
		}
		positions := map[int]int{}
		for i, id := range q.tape.Outputs {
			positions[id] = i
		}
		selected := make([][]float32, len(q.results))
		for i, id := range q.results {
			selected[i] = values[positions[id]]
		}
		if e = policy.check(selected); e != nil {
			return fail(e)
		}
		if e = p.stage(ctx, "download", gi, -1); e != nil {
			return fail(e)
		}
		if candidate == nil {
			candidate = make([][]float32, len(selected))
			for i, v := range selected {
				candidate[i] = append([]float32(nil), v...)
				p.ledger.CandidatePayload += int64(len(v)) * 4
			}
		} else {
			for i, v := range selected {
				for j, x := range v {
					if math.Abs(float64(x)-float64(candidate[i][j])) > 2e-6 {
						return fail(fmt.Errorf("group recomputed result drift"))
					}
				}
			}
		}
		gradients := map[string][]float32{}
		indices := map[string][]float32{}
		for ni, name := range q.names {
			if e = p.stage(ctx, "lookup", gi, ni); e != nil {
				return fail(e)
			}
			b := q.bindings[name]
			v := values[positions[b.Gradient]]
			n, _ := q.tape.Tensors[b.Gradient].Elements()
			if len(v) != n || !finiteF32(v) {
				return fail(fmt.Errorf("group actual G extent/finite"))
			}
			gradients[name] = v
			if c, ok := q.compact[name]; ok {
				ix := values[positions[c.Index]]
				if e = train.ValidateCompactEmbeddingIndex(ix, c.Rows, c.Vocabulary); e != nil {
					return fail(e)
				}
				indices[name] = ix
			}
		}
		if e = p.stage(ctx, "actual", gi, -1); e != nil {
			return fail(e)
		}
		for ni, name := range q.names {
			g := stableG[name]
			v := gradients[name]
			next := make([]float32, len(g))
			p.ledger.AccumulateScratch += int64(len(g)) * 4
			copy(next, g)
			if c, ok := q.compact[name]; ok {
				ix := indices[name]
				for slot := 0; slot < c.Rows; slot++ {
					token := int(ix[slot])
					if token < 0 {
						continue
					}
					for ch := 0; ch < c.Width; ch++ {
						i := token*c.Width + ch
						next[i] = g[i] + scale*v[slot*c.Width+ch]
					}
				}
			} else {
				for i := range g {
					next[i] = g[i] + scale*v[i]
				}
			}
			if !finiteF32(next) {
				return fail(fmt.Errorf("group accumulate overflow"))
			}
			if e = p.stage(ctx, "copy", gi, ni); e != nil {
				return fail(e)
			}
			copy(g, next)
			s.ready[name] = true
		}
		if e = p.stage(ctx, "close", gi, -1); e != nil {
			return fail(e)
		}
		owner.closed = true
		owner.outputs = nil
		owner.plan = nil
		p.live--
		p.ledger.Closed++
		owner = nil
		if e = p.stage(ctx, "closed", gi, -1); e != nil {
			return fail(e)
		}
	}
	if e = p.stage(ctx, "publish", len(plans)-1, -1); e != nil {
		return fail(e)
	}
	published = make([][]float32, len(candidate))
	for i, v := range candidate {
		published[i] = append([]float32(nil), v...)
		p.ledger.PublishedPayload += int64(len(v)) * 4
	}
	return published, nil
}
