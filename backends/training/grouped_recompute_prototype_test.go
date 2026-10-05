package training

// TEST-ONLY ownership prototype. No callable production API or native adapter.
import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
)

type groupedProbe struct {
	state         *ResidentState
	driver        *groupedProbeDriver
	metadataLimit int64 // test override; default accounts 1 MiB of metadata bytes.
}
type groupedProbePlan struct {
	tape     Tape
	bound    map[int]ResidentBuffer // ALL live P sources, separate from selected G.
	names    []string
	bindings map[string]ResidentTapeBinding
	compact  map[string]ResidentCompactEmbeddingBinding
	results  []int
}
type groupedProbeDriver struct {
	sharedHostDriver
	callback              func(string, int)
	group                 int
	live, maxLive         int
	freezeCopies          int
	constructedBoundSizes []int
	failStage             string
	failGroup             int
	nilOwner              bool
	corruptWitness        bool
}

func (d *groupedProbeDriver) stage(stage string) error {
	if d.callback != nil {
		d.callback(stage, d.group)
	}
	if d.failStage == stage && d.failGroup == d.group {
		return fmt.Errorf("group%d %s fault", d.group, stage)
	}
	return nil
}
func (d *groupedProbeDriver) construct(ctx context.Context, p groupedProbePlan) (ResidentBoundTape, error) {
	if err := d.stage("construct"); err != nil {
		if d.nilOwner {
			return nil, err
		}
		x := &sharedHostTape{driver: &d.sharedHostDriver, tape: p.tape, bound: p.bound}
		d.live++
		return &groupedProbeTape{d: d, ResidentBoundTape: x, ctx: ctx}, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if d.live != 0 {
		return nil, errors.New("previous group not closed")
	}
	d.constructedBoundSizes = append(d.constructedBoundSizes, len(p.bound))
	x, err := d.sharedHostDriver.NewBoundTape(ctx, p.tape, p.bound, 1<<20)
	if err != nil {
		return x, err
	}
	d.live++
	d.maxLive = max(d.maxLive, d.live)
	if d.callback != nil {
		d.callback("constructed", d.group)
	}
	return &groupedProbeTape{d: d, ResidentBoundTape: x, ctx: ctx}, nil
}

type groupedProbeTape struct {
	d *groupedProbeDriver
	ResidentBoundTape
	ctx context.Context
}

func (t *groupedProbeTape) RunDevice(ctx context.Context) error {
	if err := t.d.stage("run"); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return t.ResidentBoundTape.RunDevice(ctx)
}
func (t *groupedProbeTape) OutputBuffer(id int) (ResidentBuffer, error) {
	if err := t.d.stage("lookup"); err != nil {
		return nil, err
	}
	if err := t.ctx.Err(); err != nil {
		return nil, err
	}
	return t.ResidentBoundTape.OutputBuffer(id)
}
func (t *groupedProbeTape) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	if err := t.d.stage("download"); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	v, err := t.ResidentBoundTape.DownloadOutputs(ctx, ids)
	if err == nil && t.d.corruptWitness && t.d.group == t.d.failGroup {
		v[len(v)-1][0] = float32(math.NaN())
	}
	return v, err
}
func (t *groupedProbeTape) Close() error {
	if err := t.d.stage("close"); err != nil {
		return err
	}
	if err := t.ctx.Err(); err != nil {
		return err
	}
	if err := t.ResidentBoundTape.Close(); err != nil {
		return err
	}
	t.d.live--
	// Notification after successful underlying close is deliberately allowed to
	// cancel. The coordinator must detect that without resurrecting this owner.
	if t.d.callback != nil {
		t.d.callback("closed", t.d.group)
	}
	return nil
}
func freezeGroupedProbeTape(src Tape) Tape {
	t := Tape{Tensors: make([]TensorSpec, len(src.Tensors)), Commands: make([]Command, len(src.Commands)), Outputs: append([]int(nil), src.Outputs...)}
	for i, s := range src.Tensors {
		s.Shape = append([]int(nil), s.Shape...)
		if s.Values != nil {
			s.Values = append([]float32{}, s.Values...)
		}
		t.Tensors[i] = s
	}
	for i, c := range src.Commands {
		c.Inputs = append([]int(nil), c.Inputs...)
		t.Commands[i] = c
	}
	return t
}
func cloneProbeBindings(b map[string]ResidentTapeBinding) map[string]ResidentTapeBinding {
	out := map[string]ResidentTapeBinding{}
	for n, v := range b {
		out[n] = v
	}
	return out
}
func cloneProbeCompact(b map[string]ResidentCompactEmbeddingBinding) map[string]ResidentCompactEmbeddingBinding {
	if b == nil {
		return nil
	}
	out := map[string]ResidentCompactEmbeddingBinding{}
	for n, v := range b {
		out[n] = v
	}
	return out
}

// Pure-plan admission uses extents/topology only. Actual driver alias/liveness
// validation is intentionally performed later against completed tape buffers.
func (p *groupedProbe) run(ctx context.Context, source Tape, all map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, results []int, groups [][]string, scale float32) (published [][]float32, err error) {
	s, d := p.state, p.driver
	if err = s.admit(ctx); err != nil {
		return nil, err
	}
	if s.shared == nil || !residentFinite(scale) {
		return nil, errors.New("probe state/scale")
	}
	s.borrowed = true
	defer func() { s.borrowed = false }() // ALL callbacks, groups, close, publication.
	limit := p.metadataLimit
	if limit == 0 {
		limit = 1 << 20
	}
	requestMetadata, sourceMetadata, e := admitGroupedProbeMetadata(source, all, compact, results, groups, limit)
	if e != nil {
		return nil, e
	}
	if err = validateResidentTapeBindings(source, all, compact, 1<<20); err != nil {
		return nil, err
	}
	if len(groups) == 0 || len(groups) > 128 {
		return nil, errors.New("probe group count")
	}
	frozenGroups := make([][]string, len(groups))
	for i, g := range groups {
		if len(g) == 0 || len(g) > 8 {
			return nil, errors.New("probe group size")
		}
		frozenGroups[i] = append([]string(nil), g...)
	}
	groups = frozenGroups
	d.freezeCopies++ // Instrument copies without a mutating pre-freeze callback.
	frozen := freezeGroupedProbeTape(source)
	all = cloneProbeBindings(all)
	compact = cloneProbeCompact(compact)
	results = append([]int(nil), results...)
	if err = validateResidentTapeBindings(frozen, all, compact, 1<<20); err != nil {
		return nil, err
	}
	if len(groups) == 0 || len(groups) > 128 || len(results) < 2 || len(results) > 16 {
		return nil, errors.New("probe group/result bound")
	}
	declared := map[int]bool{}
	for _, id := range frozen.Outputs {
		declared[id] = true
	}
	seenResults := map[int]bool{}
	for _, id := range results {
		if !declared[id] || seenResults[id] {
			return nil, errors.New("probe result root")
		}
		seenResults[id] = true
	}
	if n, _ := frozen.Tensors[results[len(results)-1]].Elements(); n != 1 {
		return nil, errors.New("probe scalar witness")
	}
	for _, c := range frozen.Commands {
		if c.Op != "vector" && !(c.Op == "embedding" && c.Mode == 3) {
			return nil, errors.New("probe unsupported/nonpure opcode")
		}
		if c.Op == "vector" && c.Mode != 0 {
			return nil, errors.New("probe mock vector mode")
		}
	}
	// Freeze P into private host snapshots for the test mock. They never expose G.
	// Charge these copies in reporting; do not pretend they are native borrows.
	allP := map[int]ResidentBuffer{}
	for name, b := range all {
		i, ok := s.index[name]
		if !ok {
			return nil, errors.New("probe foreign parameter")
		}
		e := s.entries[i]
		n, _ := frozen.Tensors[b.Parameter].Elements()
		if n != e.n {
			return nil, errors.New("probe parameter extent")
		}
		raw, e2 := d.raw(ctx, e.parameter, false)
		if e2 != nil {
			return nil, e2
		}
		allP[b.Parameter] = &residentHostBuffer{values: append([]float32(nil), raw.values...), driver: &d.residentHostDriver}
	}
	if err = ValidateExternalTapeBindings(frozen, allP); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	plans := make([]groupedProbePlan, 0, len(groups))
	commands, metadata := int64(0), requestMetadata
	for gi, names := range groups {
		if len(names) == 0 || len(names) > 8 {
			return nil, errors.New("probe group size")
		}
		names = append([]string(nil), names...)
		sort.Strings(names)
		roots := append([]int(nil), results...)
		for _, name := range names {
			b, ok := all[name]
			if !ok || seen[name] {
				return nil, errors.New("probe duplicate/foreign gradient")
			}
			seen[name] = true
			roots = append(roots, b.Gradient)
			if c, ok := compact[name]; ok {
				roots = append(roots, c.Index)
			}
		}
		// Reserve a conservative full-source-sized closure, remap and planner
		// scratch BEFORE allocating another pruned plan. Overestimation is
		// deliberate; actual struct layout/allocator peak remains unmeasured.
		reservation := sourceMetadata + int64(len(frozen.Tensors))*64 + int64(len(allP)+len(names))*128 + int64(len(roots)+len(results))*8 + 256
		if e := addGroupedProbeMetadata(&metadata, reservation, limit); e != nil {
			return nil, e
		}
		pruned, remap, e := PruneTape(frozen, roots, 1<<20)
		if e != nil {
			return nil, e
		}
		plan := groupedProbePlan{tape: pruned, bound: map[int]ResidentBuffer{}, names: names, bindings: map[string]ResidentTapeBinding{}, compact: map[string]ResidentCompactEmbeddingBinding{}, results: make([]int, len(results))}
		protected := map[int]bool{}
		for id, b := range allP {
			if remap[id] >= 0 {
				plan.bound[remap[id]] = b
				protected[remap[id]] = true
			}
		}
		for i, id := range results {
			plan.results[i] = remap[id]
		}
		for _, name := range names {
			b := all[name]
			plan.bindings[name] = ResidentTapeBinding{Parameter: remap[b.Parameter], Gradient: remap[b.Gradient]}
			if c, ok := compact[name]; ok {
				c.Index = remap[c.Index]
				plan.compact[name] = c
				dense, _, _, e := CompactEmbeddingAccumulateShape(c.Rows, c.Width, c.Vocabulary, scale)
				if e != nil || dense != s.entries[s.index[name]].n {
					return nil, errors.New("probe compact plan")
				}
			}
		}
		if _, e = PlanZeroTapeStorage(pruned, 1<<20, protected); e != nil {
			return nil, e
		}
		commands += int64(len(pruned.Commands))
		if commands > 65536 {
			return nil, errors.New("probe checked work bound")
		}
		d.group = gi
		if e = d.stage("plan"); e != nil {
			return nil, e
		}
		if e = ctx.Err(); e != nil {
			return nil, e
		}
		plans = append(plans, plan)
	}
	if len(seen) != len(all) {
		return nil, errors.New("probe missing gradient union")
	}
	executed := false
	var candidate [][]float32
	for gi, plan := range plans {
		d.group = gi
		var owner ResidentBoundTape
		// Any between-group error after progress poisons even a nil-owner failure.
		fail := func(e error) ([][]float32, error) {
			if executed || owner != nil {
				s.terminal = e
				if owner != nil {
					s.retained = append(s.retained, owner)
				}
			}
			return nil, e
		}
		if err = ctx.Err(); err != nil {
			return fail(err)
		}
		owner, err = d.construct(ctx, plan)
		if err != nil {
			return fail(err)
		}
		if owner == nil {
			return fail(errors.New("probe nil tape"))
		}
		if err = ctx.Err(); err != nil {
			return fail(err)
		}
		executed = true
		if err = owner.RunDevice(ctx); err != nil {
			return fail(err)
		}
		values, e := owner.DownloadOutputs(ctx, plan.results)
		if e != nil {
			return fail(e)
		}
		if err = d.stage("witness"); err != nil {
			return fail(err)
		}
		if len(values) != len(plan.results) {
			return fail(errors.New("probe result count"))
		}
		for i, v := range values {
			n, _ := plan.tape.Tensors[plan.results[i]].Elements()
			if len(v) != n {
				return fail(errors.New("probe result extent"))
			}
			for _, x := range v {
				if !residentFinite(x) {
					return fail(errors.New("probe result nonfinite"))
				}
			}
		}
		if values[len(values)-1][0] != 0 {
			return fail(errors.New("probe witness rejected"))
		}
		if candidate == nil {
			candidate = make([][]float32, len(values))
			for i, v := range values {
				candidate[i] = append([]float32(nil), v...)
			}
		} else {
			for i, v := range values {
				for j, x := range v {
					if math.Abs(float64(x)-float64(candidate[i][j])) > 2e-6 {
						return fail(errors.New("probe recomputed result drift"))
					}
				}
			}
		}
		sources, indices := map[string]ResidentBuffer{}, map[string]ResidentBuffer{}
		for _, name := range plan.names {
			b, e := owner.OutputBuffer(plan.bindings[name].Gradient)
			if e != nil {
				return fail(e)
			}
			sources[name] = b
			// Output buffer postchecks happen after execution but before any G
			// mutation; a pure validator failure here is still terminal.
			raw, e := d.raw(ctx, b, false)
			if e != nil {
				return fail(e)
			}
			gradientN, _ := plan.tape.Tensors[plan.bindings[name].Gradient].Elements()
			if len(raw.values) != gradientN {
				return fail(errors.New("probe gradient extent"))
			}
			for _, v := range raw.values {
				if !residentFinite(v) {
					return fail(errors.New("probe gradient nonfinite"))
				}
			}
			if c, ok := plan.compact[name]; ok {
				ix, e := owner.OutputBuffer(c.Index)
				if e != nil {
					return fail(e)
				}
				rawIndex, e := d.raw(ctx, ix, false)
				if e != nil {
					return fail(e)
				}
				if e = ValidateCompactEmbeddingIndex(rawIndex.values, c.Rows, c.Vocabulary); e != nil {
					return fail(e)
				}
				indices[name] = ix
			}
		}
		if err = d.stage("actual"); err != nil {
			return fail(err)
		}
		if err = s.sharedAccumulate(ctx, plan.names, sources, indices, plan.compact, nil, scale); err != nil {
			return fail(err)
		}
		if err = ctx.Err(); err != nil {
			return fail(err)
		}
		if err = owner.Close(); err != nil {
			return fail(err)
		}
		owner = nil
		if err = ctx.Err(); err != nil {
			return fail(err)
		}
	}
	if err = d.stage("publish"); err != nil {
		s.terminal = err
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		s.terminal = err
		return nil, err
	}
	return candidate, nil
}
func newGroupedProbe(t *testing.T, f ResidentSnapshot) *groupedProbe {
	t.Helper()
	d := &groupedProbeDriver{}
	d.sentinel = errors.New("probe execution fault")
	s, err := NewSharedWorkspaceResidentState(context.Background(), d, f, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	return &groupedProbe{state: s, driver: d}
}
func probeResults(tape Tape) (Tape, []int) {
	// Metrics remain a single owned result and a mandatory scalar witness.
	metric := len(tape.Tensors)
	tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{.75}}, TensorSpec{Shape: []int{1}, Values: []float32{0}})
	tape.Outputs = append(tape.Outputs, metric, metric+1)
	return tape, []int{metric, metric + 1}
}
func assertProbeClose(t *testing.T, p *groupedProbe) {
	t.Helper()
	if p.driver.live != 0 || p.driver.maxLive != 1 {
		t.Fatal("probe group lifetime", p.driver.live, p.driver.maxLive)
	}
	if p.state.Close() != nil {
		t.Fatal("probe close")
	}
}
func TestTrainingGroupedProbeNumerical(t *testing.T) {
	ctx := context.Background()
	var comparisons int
	var sum, sq, maximum float64
	for _, scale := range []float32{0, .25, 1} {
		for _, prior := range []bool{false, true} {
			f := sharedCompactFixture()
			p := newGroupedProbe(t, f)
			tape, b, c := compactBindingFixture()
			tape, results := probeResults(tape)
			previous := map[string][]float32{"a": {.1, -.2, .3, -.4, .5, -.6, .7, -.8}, "b": {.9, -1}}
			if prior {
				if err := p.state.Accumulate(ctx, previous, 1); err != nil {
					t.Fatal(err)
				}
			}
			identities := []ResidentBuffer{p.state.entries[0].gradient, p.state.entries[1].gradient}
			out, err := p.run(ctx, tape, b, c, results, [][]string{{"a"}, {"b"}}, scale)
			if err != nil || out[0][0] != .75 || out[1][0] != 0 {
				t.Fatal(err, out)
			}
			expected := map[string][]float64{"a": {3, 4, 0, 0, 6, 8, 0, 0}, "b": {3, 5}}
			for i, e := range p.state.entries {
				if e.gradient != identities[i] {
					t.Fatal("G swapped")
				}
				v := e.gradient.(*residentHostBuffer).values
				for j, x := range v {
					want := float64(scale) * expected[e.name][j]
					if prior {
						want += float64(previous[e.name][j])
					}
					error := math.Abs(float64(x) - want)
					if !residentFinite(x) || error > 2e-6 {
						t.Fatal("independent F64 grouped gradient", e.name, j, x, want)
					}
					comparisons++
					sum += error
					sq += error * error
					maximum = math.Max(maximum, error)
				}
			}
			out[0][0] = 99
			assertProbeClose(t, p)
		}
	}
	t.Logf("independent F64 grouped gradients comparisons=%d meanAbs=%.9g RMS=%.9g maxAbs=%.9g unchanged2e-6", comparisons, sum/float64(comparisons), math.Sqrt(sq/float64(comparisons)), maximum)
}
func TestTrainingGroupedProbeAdmissionAndFreeze(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []string{"empty", "missing", "duplicate", "foreign", "size", "late-plan", "stateful", "results"} {
		t.Run(kind, func(t *testing.T) {
			p := newGroupedProbe(t, sharedCompactFixture())
			tape, b, c := compactBindingFixture()
			tape, r := probeResults(tape)
			groups := [][]string{{"a"}, {"b"}}
			switch kind {
			case "empty":
				groups = nil
			case "missing":
				groups = groups[:1]
			case "duplicate":
				groups = [][]string{{"a"}, {"a", "b"}}
			case "foreign":
				groups[1] = []string{"z"}
			case "size":
				groups[0] = make([]string, 9)
			case "late-plan":
				p.driver.failStage = "plan"
				p.driver.failGroup = 1
			case "stateful":
				tape.Commands[0].Op = "unapproved"
			case "results":
				r = []int{r[0], r[0]}
			}
			if out, err := p.run(ctx, tape, b, c, r, groups, 1); err == nil || out != nil || p.state.terminal != nil || p.state.dirty || p.driver.events["tape-new"] != 0 {
				t.Fatal("pure allgroup rejection", kind, err)
			}
			if p.state.Close() != nil {
				t.Fatal("close")
			}
		})
	}
	p := newGroupedProbe(t, sharedCompactFixture())
	tape, b, c := compactBindingFixture()
	tape, r := probeResults(tape)
	groups := [][]string{{"a"}, {"b"}}
	p.driver.callback = func(stage string, _ int) {
		if stage == "plan" {
			tape.Tensors[1].Values[0] = 1234
			tape.Tensors[r[0]].Values[0] = 88
			tape.Commands[0].Inputs[0] = 0
			p.state.entries[0].parameter.(*residentHostBuffer).values[0] = 77
			delete(b, "b")
			delete(c, "a")
			groups[1][0] = "forged"
		}
	}
	out, err := p.run(ctx, tape, b, c, r, groups, 1)
	if err != nil || out[0][0] != .75 {
		t.Fatal("frozen callback inputs", err, out)
	}
	if p.state.entries[0].gradient.(*residentHostBuffer).values[4] != 6 {
		t.Fatal("source values not frozen")
	}
	assertProbeClose(t, p)
}
func TestTrainingGroupedProbeFailureFences(t *testing.T) {
	ctx := context.Background()
	for _, stage := range []string{"construct", "nil-construct", "run", "download", "witness", "lookup", "actual", "accumulate", "copy", "close", "publish", "cancel"} {
		t.Run(stage, func(t *testing.T) {
			fixture := sharedCompactFixture()
			fixture.Parameters["z"] = ResidentParameter{Parameter: []float32{1}, M: []float32{0}, V: []float32{0}, EMA: []float32{1}}
			p := newGroupedProbe(t, fixture)
			tape, b, c := compactBindingFixture()
			id := len(tape.Tensors)
			tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{1}}, TensorSpec{Shape: []int{1}, Values: []float32{2}})
			tape.Outputs = append(tape.Outputs, id+1)
			b["z"] = ResidentTapeBinding{id, id + 1}
			tape, r := probeResults(tape)
			context, cancel := context.WithCancel(ctx)
			defer cancel()
			p.driver.failGroup = 1
			p.driver.failStage = stage
			if stage == "publish" {
				p.driver.failGroup = 2
			}
			if stage == "nil-construct" {
				p.driver.failStage = "construct"
				p.driver.nilOwner = true
			}
			if stage == "witness" {
				p.driver.failStage = ""
				p.driver.corruptWitness = true
			}
			if stage == "copy" || stage == "accumulate" {
				p.driver.failStage = ""
				p.driver.fail = stage
				p.driver.at = 2
				if stage == "accumulate" {
					p.driver.at = 1
				}
			}
			if stage == "cancel" {
				p.driver.failStage = ""
				p.driver.callback = func(where string, g int) {
					if g == 1 && where == "construct" {
						cancel()
					}
				}
			}
			out, err := p.run(context, tape, b, c, r, [][]string{{"a"}, {"b"}, {"z"}}, 1)
			if err == nil || out != nil || p.state.terminal == nil {
				t.Fatal("late terminal/publication", stage, err, out)
			}
			if !p.state.entries[0].ready || sharedGradientContentsError(p.state.entries[0].gradient.(*residentHostBuffer).values, []float32{3, 4, 0, 0, 6, 8, 0, 0}) != nil {
				t.Fatal("earlier G lost")
			}
			copies := p.driver.events["copy"]
			if p.state.Close() == nil || p.state.Accumulate(ctx, map[string][]float32{"b": {0, 0}}, 1) == nil {
				t.Fatal("terminal retry")
			}
			if _, err = p.state.Update(ctx, 1); err == nil {
				t.Fatal("terminal update")
			}
			if _, err = p.state.Snapshot(ctx); err == nil {
				t.Fatal("terminal snapshot")
			}
			if p.driver.events["copy"] != copies {
				t.Fatal("retry dispatched")
			}
			if stage != "nil-construct" && stage != "cancel" && stage != "publish" && len(p.state.retained) == 0 {
				t.Fatal("uncertain owner missing")
			}
			expectedGroup := 1
			if stage == "publish" {
				expectedGroup = 2
			}
			if p.driver.group != expectedGroup {
				t.Fatal("later group started")
			}
			if stage != "publish" && p.state.entries[p.state.index["z"]].ready {
				t.Fatal("third group gradient committed after failure")
			}
		})
	}
}
func TestTrainingGroupedProbeReentry(t *testing.T) {
	ctx := context.Background()
	p := newGroupedProbe(t, sharedCompactFixture())
	tape, b, c := compactBindingFixture()
	tape, r := probeResults(tape)
	calls := 0
	fence := func() {
		calls++
		if !p.state.borrowed || p.state.Close() == nil || p.state.Accumulate(ctx, map[string][]float32{"b": {1, 2}}, 1) == nil {
			t.Fatal("whole operation fence")
		}
		if _, err := p.state.Update(ctx, 1); err == nil {
			t.Fatal("update fence")
		}
		if _, err := p.state.Snapshot(ctx); err == nil {
			t.Fatal("snapshot fence")
		}
		if _, err := p.run(ctx, tape, b, c, r, [][]string{{"a"}, {"b"}}, 1); err == nil {
			t.Fatal("nested group fence")
		}
	}
	p.driver.callback = func(string, int) { fence() }
	p.driver.hook = func(string) { fence() }
	if _, err := p.run(ctx, tape, b, c, r, [][]string{{"a"}, {"b"}}, .25); err != nil || calls < 14 {
		t.Fatal(err, calls)
	}
	assertProbeClose(t, p)
}

// Independent pinned fixture tests the coordinator/update contract, not newly
// computed full-backward kernels. Gradients are immutable oracle source roots.
func TestTrainingGroupedProbePyTorchUpdate(t *testing.T) {
	data, err := os.ReadFile("../../model/pockettts/testdata/training_step_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var o struct {
		Schema                int    `json:"schema"`
		Revision              string `json:"upstream_revision"`
		Parameters, Gradients map[string][]float32
		Adam                  struct {
			LR       float32              `json:"learning_rate"`
			B1       float32              `json:"beta1"`
			B2       float32              `json:"beta2"`
			Eps      float32              `json:"epsilon"`
			WD       float32              `json:"weight_decay"`
			EMA      float32              `json:"ema_decay"`
			After    map[string][]float32 `json:"parameters_after_step"`
			EMAAfter map[string][]float32 `json:"ema_after_step"`
		} `json:"adamw"`
	}
	if err = json.Unmarshal(data, &o); err != nil || o.Schema != 1 || o.Revision != "0acce6b2f390150267557770d2098c5caa9a18ac" {
		t.Fatal("fixture provenance", err)
	}
	f := ResidentSnapshot{Config: ResidentConfig{o.Adam.LR, o.Adam.B1, o.Adam.B2, o.Adam.Eps, o.Adam.WD, o.Adam.EMA}, Parameters: map[string]ResidentParameter{}}
	var names []string
	for n := range o.Gradients {
		names = append(names, n)
	}
	sort.Strings(names)
	tape := Tape{}
	b := map[string]ResidentTapeBinding{}
	var groups [][]string
	for _, name := range names {
		v := o.Parameters[name]
		f.Parameters[name] = ResidentParameter{Parameter: v, M: make([]float32, len(v)), V: make([]float32, len(v)), EMA: append([]float32(nil), v...)}
		id := len(tape.Tensors)
		tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{len(v)}, Values: v}, TensorSpec{Shape: []int{len(v)}, Values: o.Gradients[name]})
		tape.Outputs = append(tape.Outputs, id+1)
		b[name] = ResidentTapeBinding{id, id + 1}
	}
	for first := 0; first < len(names); first += 8 {
		groups = append(groups, append([]string(nil), names[first:min(first+8, len(names))]...))
	}
	tape, r := probeResults(tape)
	p := newGroupedProbe(t, f)
	if _, err = p.run(context.Background(), tape, b, nil, r, groups, 1); err != nil {
		t.Fatal(err)
	}
	if _, err = p.state.Update(context.Background(), math.MaxFloat32); err != nil {
		t.Fatal(err)
	}
	snap, err := p.state.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	var sum, sq, maximum float64
	for name, v := range snap.Parameters {
		for j, x := range append(append([]float32(nil), v.Parameter...), v.EMA...) {
			want := o.Adam.After[name]
			if j >= len(v.Parameter) {
				want = o.Adam.EMAAfter[name]
			}
			k := j % len(v.Parameter)
			e := math.Abs(float64(x) - float64(want[k]))
			if !residentFinite(x) || e > 4e-6 {
				t.Fatal("pinned grouped P/EMA", name, j, x, want[k])
			}
			count++
			sum += e
			sq += e * e
			maximum = math.Max(maximum, e)
		}
	}
	assertProbeClose(t, p)
	t.Logf("pinned independent PyTorch P/EMA comparisons=%d meanAbs=%.9g RMS=%.9g maxAbs=%.9g unchanged4e-6", count, sum/float64(count), math.Sqrt(sq/float64(count)), maximum)
}

func TestTrainingGroupedProbeClippedMultistep(t *testing.T) {
	ctx := context.Background()
	f := sharedCompactFixture()
	p := newGroupedProbe(t, f)
	reference, err := NewStateCheckResidentState(ctx, &deviceScaleHostDriver{}, f, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	tape, b, c := compactBindingFixture()
	tape, r := probeResults(tape)
	raw := map[string][]float32{"a": {3, 4, 0, 0, 6, 8, 0, 0}, "b": {3, 5}}
	count := 0
	maximum := 0.0
	sum, squares := 0.0, 0.0
	for _, scale := range []float32{.25, 1, 0} {
		before, e := reference.Snapshot(ctx)
		if e != nil {
			t.Fatal(e)
		}
		raw["b"] = []float32{before.Parameters["b"].Parameter[0] + 1, before.Parameters["b"].Parameter[1] + 2}
		if _, err = p.run(ctx, tape, b, c, r, [][]string{{"b"}, {"a"}}, scale); err != nil {
			t.Fatal(err)
		}
		if err = reference.Accumulate(ctx, raw, scale); err != nil {
			t.Fatal(err)
		}
		norm, err := p.state.Update(ctx, .1)
		if err != nil {
			t.Fatal(err)
		}
		rn, err := reference.Update(ctx, .1)
		if err != nil {
			t.Fatal(err)
		}
		f64 := 0.0
		for _, v := range raw {
			for _, x := range v {
				f64 = math.Hypot(f64, float64(x)*float64(scale))
			}
		}
		if math.Abs(norm-f64) > 3e-7*max(1, f64) || math.Abs(norm-rn) > 3e-7*max(1, f64) {
			t.Fatal("independentF64 clipped norm", norm, rn, f64)
		}
		got, err := p.state.Snapshot(ctx)
		if err != nil {
			t.Fatal(err)
		}
		want, err := reference.Snapshot(ctx)
		if err != nil {
			t.Fatal(err)
		}
		for name, a := range got.Parameters {
			z := want.Parameters[name]
			for k, v := range [][]float32{a.Parameter, a.M, a.V, a.EMA} {
				w := [][]float32{z.Parameter, z.M, z.V, z.EMA}[k]
				if len(v) != len(w) {
					t.Fatal("clipped extent")
				}
				for j, x := range v {
					e := math.Abs(float64(x) - float64(w[j]))
					if !residentFinite(x) || e > 2e-6 {
						t.Fatal("clipped P/M/V/EMA", name, k, j, e)
					}
					count++
					sum += e
					squares += e * e
					maximum = math.Max(maximum, e)
				}
			}
		}
	}
	if reference.Close() != nil {
		t.Fatal("reference close")
	}
	assertProbeClose(t, p)
	t.Logf("clipped multistep established host P/M/V/EMA comparisons=%d meanAbs%.9g RMS%.9g maxAbs%.9g unchanged2e-6; independentF64norm unchanged3e-7", count, sum/float64(count), math.Sqrt(squares/float64(count)), maximum)
}
func TestTrainingGroupedProbeTransportAndInitialFailure(t *testing.T) {
	ctx := context.Background()
	p := newGroupedProbe(t, residentFixture())
	tape, b := residentBindingFixture()
	tape, r := probeResults(tape)
	// Only a is bound by this fixture; this probe admits an exact gradient union
	// for the supplied subgraph without claiming all state parameters ready.
	tape.Tensors[b["a"].Gradient].Values = []float32{float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32}
	tape.Commands = nil
	out, err := p.run(ctx, tape, b, nil, r, [][]string{{"a"}}, 1)
	if err != nil || out[0][0] != .75 {
		t.Fatal(err, out)
	}
	// Accumulation is arithmetic (zero sign may change); the existing transport
	// regression covers exact copy bits. The subnormal must survive mock math.
	if p.state.entries[0].gradient.(*residentHostBuffer).values[1] != math.SmallestNonzeroFloat32 {
		t.Fatal("subnormal transport")
	}
	assertProbeClose(t, p)
	p = newGroupedProbe(t, sharedCompactFixture())
	tape, b, c := compactBindingFixture()
	tape, r = probeResults(tape)
	p.driver.failStage = "construct"
	p.driver.failGroup = 0
	p.driver.nilOwner = true
	if out, err = p.run(ctx, tape, b, c, r, [][]string{{"a"}, {"b"}}, 1); err == nil || out != nil || p.state.terminal != nil || p.state.dirty {
		t.Fatal("initial nil-owner no progress", err)
	}
	if p.state.Close() != nil {
		t.Fatal("initial failure close")
	}
}

func TestTrainingGroupedProbeTokenMaskResults(t *testing.T) {
	ctx := context.Background()
	for _, unique := range []bool{false, true} {
		for _, valid := range []int{1, 3} {
			p := newGroupedProbe(t, sharedCompactFixture())
			tape, b, c := compactBindingFixture()
			tokens := []float32{2, 0, 2}
			if unique {
				tokens = []float32{0, 1, 2}
			}
			for row := valid; row < 3; row++ {
				clear(tape.Tensors[1].Values[row*2 : row*2+2])
			}
			index, err := BuildCompactEmbeddingIndex(tokens, 4)
			if err != nil {
				t.Fatal(err)
			}
			tape.Tensors[2].Values = index
			tape, r := probeResults(tape)
			r = []int{r[0], 6, r[1]}
			out, err := p.run(ctx, tape, b, c, r, [][]string{{"a", "b"}}, .25)
			if err != nil {
				t.Fatal(err)
			}
			if sharedGradientContentsError(out[1], []float32{3, 5}) != nil || out[0][0] != .75 || out[2][0] != 0 {
				t.Fatal("independent selected input/result", out)
			}
			want := make([]float32, 8)
			for row := 0; row < valid; row++ {
				for ch := 0; ch < 2; ch++ {
					want[int(tokens[row])*2+ch] += .25 * float32(row*2+ch+1)
				}
			}
			if err = sharedGradientContentsError(p.state.entries[0].gradient.(*residentHostBuffer).values, want); err != nil {
				t.Fatal("F64 token/mask gradient", unique, valid, err)
			}
			out[1][0] = 99
			assertProbeClose(t, p)
		}
	}
}

func TestTrainingGroupedProbeWithinGroupFailure(t *testing.T) {
	p := newGroupedProbe(t, sharedCompactFixture())
	tape, b, c := compactBindingFixture()
	tape, r := probeResults(tape)
	p.driver.fail = "copy"
	p.driver.at = 2
	out, err := p.run(context.Background(), tape, b, c, r, [][]string{{"a", "b"}}, 1)
	if err == nil || out != nil || p.state.terminal == nil || !p.state.entries[0].ready || p.state.entries[1].ready || len(p.state.retained) == 0 {
		t.Fatal("within-group partial commit", err)
	}
	if err = sharedGradientContentsError(p.state.entries[0].gradient.(*residentHostBuffer).values, []float32{3, 4, 0, 0, 6, 8, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err = sharedGradientContentsError(p.state.entries[1].gradient.(*residentHostBuffer).values, []float32{3, 0}); err != nil {
		t.Fatal(err)
	}
	assertSharedTerminalFences(t, p.state, &p.driver.sharedHostDriver)
}

func TestTrainingGroupedProbeEarlyExecutedFailure(t *testing.T) {
	for _, stage := range []string{"run", "witness", "actual"} {
		t.Run(stage, func(t *testing.T) {
			p := newGroupedProbe(t, sharedCompactFixture())
			tape, b, c := compactBindingFixture()
			tape, r := probeResults(tape)
			p.driver.failStage = stage
			p.driver.failGroup = 0
			out, err := p.run(context.Background(), tape, b, c, r, [][]string{{"a"}, {"b"}}, 1)
			if err == nil || out != nil || p.state.terminal == nil || len(p.state.retained) == 0 || p.state.dirty || p.driver.events["copy"] != 0 || p.driver.group != 0 {
				t.Fatal("early executed terminal/witness before G", stage, err)
			}
			for _, e := range p.state.entries {
				if e.ready {
					t.Fatal("early rejected gradient ready")
				}
			}
			assertSharedTerminalFences(t, p.state, &p.driver.sharedHostDriver)
		})
	}
}

// Byte-charge accounting is conservative: 256 bytes per tensor/command covers
// value structs, slice/map headers and planner records; nested IDs use 8 bytes,
// copied string payloads are charged too. F32 payload has its separate 1 MiB
// source limit. These are prototype bounds, not measured Go allocator bytes.
func addGroupedProbeMetadata(total *int64, n, limit int64) error {
	if n < 0 || limit <= 0 || *total < 0 || *total > limit || n > limit-*total {
		return errors.New("probe metadata byte ceiling")
	}
	*total += n
	return nil
}
func admitGroupedProbeMetadata(source Tape, all map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, results []int, groups [][]string, limit int64) (request, sourceBytes int64, err error) {
	// Cardinalities checked before validation walks or copying unbounded slices.
	if len(results) < 2 || len(results) > 16 || len(source.Outputs) == 0 || len(source.Outputs) > 4096 || len(source.Tensors) == 0 || len(source.Tensors) > 4096 || len(source.Commands) > 4096 || len(all) == 0 || len(all) > 4096 || len(compact) > 4096 || len(groups) == 0 || len(groups) > 128 {
		return 0, 0, errors.New("probe upfront cardinality")
	}
	add := func(n int64) bool { err = addGroupedProbeMetadata(&sourceBytes, n, limit); return err == nil }
	if !add(int64(len(source.Tensors)+len(source.Commands))*256 + int64(len(source.Outputs))*8) {
		return
	}
	for _, s := range source.Tensors {
		if len(s.Shape) < 1 || len(s.Shape) > 3 {
			return 0, 0, errors.New("probe nested shape bound")
		}
		if !add(int64(len(s.Shape)) * 8) {
			return
		}
	}
	for _, c := range source.Commands {
		if len(c.Inputs) > 16 || len(c.Op) > 64 {
			return 0, 0, errors.New("probe nested operand/op bound")
		}
		if !add(int64(len(c.Inputs))*8 + int64(len(c.Op))) {
			return
		}
	}
	request = sourceBytes
	for name := range all {
		if len(name) == 0 || len(name) > 512 {
			return 0, 0, errors.New("probe binding name bound")
		}
		if err = addGroupedProbeMetadata(&request, 128+int64(len(name)), limit); err != nil {
			return
		}
	}
	for name := range compact {
		if len(name) == 0 || len(name) > 512 {
			return 0, 0, errors.New("probe compact name bound")
		}
		if err = addGroupedProbeMetadata(&request, 128+int64(len(name)), limit); err != nil {
			return
		}
	}
	if err = addGroupedProbeMetadata(&request, int64(len(results))*8+int64(len(groups))*64, limit); err != nil {
		return
	}
	for _, g := range groups {
		if len(g) == 0 || len(g) > 8 {
			return 0, 0, errors.New("probe group size")
		}
		for _, name := range g {
			if len(name) == 0 || len(name) > 512 {
				return 0, 0, errors.New("probe group name bound")
			}
			if err = addGroupedProbeMetadata(&request, 32+int64(len(name)), limit); err != nil {
				return
			}
		}
	}
	return
}

func groupedCancellationFixture(t *testing.T) (*groupedProbe, Tape, map[string]ResidentTapeBinding, []int) {
	t.Helper()
	p := newGroupedProbe(t, residentFixture())
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{9, 9}}, {Shape: []int{2}, Values: []float32{2, 4}}, {Shape: []int{1}, Values: []float32{9}}, {Shape: []int{1}, Values: []float32{6}}}, Outputs: []int{1, 3}}
	tape, r := probeResults(tape)
	b := map[string]ResidentTapeBinding{"a": {0, 1}, "b": {2, 3}}
	return p, tape, b, r
}
func TestTrainingGroupedCorrectionCancellation(t *testing.T) {
	for _, stage := range []string{"closed", "construct", "constructed", "run", "close"} {
		t.Run(stage, func(t *testing.T) {
			p, tape, b, r := groupedCancellationFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			targetGroup := 1
			if stage == "closed" || stage == "close" {
				targetGroup = 0
			}
			callbacks := map[string]int{}
			p.driver.callback = func(where string, g int) {
				callbacks[where]++
				if where == stage && g == targetGroup {
					cancel()
				}
			}
			out, err := p.run(ctx, tape, b, nil, r, [][]string{{"a"}, {"b"}}, 1)
			if !errors.Is(err, context.Canceled) || out != nil || p.state.terminal == nil || !p.state.entries[0].ready || p.state.entries[1].ready {
				t.Fatal("cancel terminal/output/readiness", stage, err)
			}
			if e := sharedGradientContentsError(p.state.entries[0].gradient.(*residentHostBuffer).values, []float32{2, 4}); e != nil {
				t.Fatal(e)
			}
			expectedNew, retained, live := 1, 0, 0
			if stage == "constructed" || stage == "run" {
				expectedNew, retained, live = 2, 1, 1
			}
			if stage == "close" {
				retained, live = 1, 1
			}
			if p.driver.events["tape-new"] != expectedNew || p.driver.events["tape-run"] != 1 || p.driver.events["copy"] != 1 || len(p.state.retained) != retained || p.driver.live != live {
				t.Fatal("cancel exact dispatch/retention", stage, p.driver.events, len(p.state.retained), p.driver.live)
			}
			constructCalls := 1
			if stage == "construct" || stage == "constructed" || stage == "run" {
				constructCalls = 2
			}
			if callbacks["construct"] != constructCalls || callbacks["publish"] != 0 {
				t.Fatal("cancel constructor/publication callbacks", stage, callbacks)
			}
			for _, n := range p.driver.constructedBoundSizes {
				if n != 0 {
					t.Fatal("constant gradient cancellation fixture unexpectedly binds P")
				}
			}
			if retained > 0 && p.state.retained[0].(*groupedProbeTape).ResidentBoundTape.(*sharedHostTape).closed {
				t.Fatal("retained owner already closed")
			}
			if stage == "closed" && p.driver.events["tape-close"] != 1 {
				t.Fatal("successful prior owner not closed")
			}
			assertSharedTerminalFences(t, p.state, &p.driver.sharedHostDriver)
		})
	}
}
func TestTrainingGroupedCorrectionMetadata(t *testing.T) {
	for _, kind := range []string{"results", "outputs", "nested-shape", "nested-inputs", "cumulative"} {
		t.Run(kind, func(t *testing.T) {
			p, tape, b, r := groupedCancellationFixture(t)
			groups := [][]string{{"a"}, {"b"}}
			plans := 0
			p.driver.callback = func(where string, _ int) {
				if where == "plan" {
					plans++
				}
			}
			switch kind {
			case "results":
				r = make([]int, 17)
			case "outputs":
				tape.Outputs = make([]int, 4097)
				for i := range tape.Outputs {
					tape.Outputs[i] = 1
				}
			case "nested-shape":
				tape.Tensors[0].Shape = make([]int, 4)
			case "nested-inputs":
				tape.Commands = []Command{{Op: "vector", Inputs: make([]int, 17), Output: 1}}
			case "cumulative":
				request, src, e := admitGroupedProbeMetadata(tape, b, nil, r, groups, 1<<20)
				if e != nil {
					t.Fatal(e)
				}
				reservation := src + int64(len(tape.Tensors))*64 + int64(len(b)+1)*128 + int64(3+len(r))*8 + 256
				p.metadataLimit = request + reservation
			}
			out, err := p.run(context.Background(), tape, b, nil, r, groups, 1)
			expectedPlans, expectedFreezes := 0, 0
			if kind == "cumulative" {
				expectedPlans, expectedFreezes = 1, 1
			}
			if err == nil || out != nil || p.state.terminal != nil || p.state.dirty || p.driver.freezeCopies != expectedFreezes || plans != expectedPlans || p.driver.events["tape-new"] != 0 {
				t.Fatal("bounded rejection before copy/next plan", kind, err, plans)
			}
			if p.state.Close() != nil {
				t.Fatal("pure rejection close")
			}
		})
	}
}
func TestTrainingGroupedCorrectionFrozenCrossParameter(t *testing.T) {
	// A gradient reads B.P while only A is selected for accumulation in group0.
	f := residentFixture()
	f.Parameters["a"] = ResidentParameter{Parameter: []float32{.3}, M: []float32{0}, V: []float32{0}, EMA: []float32{.3}}
	p := newGroupedProbe(t, f)
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{1}, Values: []float32{9}}, {Shape: []int{1}, Values: []float32{9}}, {Shape: []int{1}, Values: []float32{3}}, {Shape: []int{1}}, {Shape: []int{1}, Values: []float32{6}}}, Commands: []Command{{Op: "vector", Inputs: []int{1, 2}, Output: 3, Scale: 1}}, Outputs: []int{3, 4}}
	tape, r := probeResults(tape)
	b := map[string]ResidentTapeBinding{"a": {0, 3}, "b": {1, 4}}
	p.driver.callback = func(where string, g int) {
		if where == "plan" && g == 0 {
			p.state.entries[p.state.index["b"]].parameter.(*residentHostBuffer).values[0] = 77
			tape.Tensors[2].Values[0] = 99
		}
	}
	out, err := p.run(context.Background(), tape, b, nil, r, [][]string{{"a"}, {"b"}}, 1)
	if err != nil || out[0][0] != .75 {
		t.Fatal(err, out)
	}
	if e := sharedGradientContentsError(p.state.entries[p.state.index["a"]].gradient.(*residentHostBuffer).values, []float32{3.5}); e != nil {
		t.Fatal("A gradient did not read frozen B.P", e)
	}
	if p.state.entries[p.state.index["b"]].parameter.(*residentHostBuffer).values[0] != 77 || tape.Tensors[2].Values[0] != 99 {
		t.Fatal("adversarial callback not executed")
	}
	if e := sharedGradientContentsError(p.state.entries[p.state.index["b"]].gradient.(*residentHostBuffer).values, []float32{6}); e != nil {
		t.Fatal(e)
	}
	assertProbeClose(t, p)
}
