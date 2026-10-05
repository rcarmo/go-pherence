package training

import (
	"context"
	"fmt"
)

// ResidentLeaseEdge borrows a declared, pinned output from an exact live owner
// in the same lease tree. Input is a constant target in the complete consumer.
// No gradient crosses it implicitly. All referenced owners stay live until the
// child borrowing them has closed; unrelated/sibling borrows are explicit.
type ResidentLeaseEdge struct {
	Source        *ResidentTapeLease
	Output, Input int
}

func (l *ResidentTapeLease) ownEdges(edges []TapeHandoffEdge) []ResidentLeaseEdge {
	out := make([]ResidentLeaseEdge, len(edges))
	for i, e := range edges {
		out[i] = ResidentLeaseEdge{Source: l, Output: e.Output, Input: e.Input}
	}
	return out
}

// RunSourcesTape admits read-only edges from multiple live producers, executes
// one pruned consumer and optionally accumulates its named gradients once.
func (l *ResidentTapeLease) RunSourcesTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, outputs []int, edges []ResidentLeaseEdge, accumulate bool, scale float32) (values [][]float32, err error) {
	if l == nil {
		return nil, fmt.Errorf("nil resident lease")
	}
	if err = l.admit(ctx, l.state); err != nil {
		if ctx != nil && ctx.Err() != nil {
			l.RetainFailure(ctx.Err())
		}
		return nil, err
	}
	defer func() {
		if err != nil {
			l.RetainFailure(err)
		}
	}()
	if bindings == nil && (accumulate || compact != nil) {
		return nil, fmt.Errorf("lease unbound accumulation/compact inventory")
	}
	return l.state.runTapeBindingsLease(ctx, t, bindings, compact, outputs, accumulate, scale, l.maxBytes, bindings == nil, true, l.reuse, l.shareZero, l, edges)
}

// RunGradientGroupTape validates the complete source/binding inventory but
// executes and accumulates only explicitly selected named gradient roots. Other
// parameter sources remain bound if those roots depend on them. Caller groups
// must be disjoint; no implicit retry or accumulation deduplication occurs.
func (l *ResidentTapeLease) RunGradientGroupTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, outputs []int, names []string, scale float32) (values [][]float32, err error) {
	if l == nil {
		return nil, fmt.Errorf("nil resident lease")
	}
	if err = l.admit(ctx, l.state); err != nil {
		if ctx != nil && ctx.Err() != nil {
			l.RetainFailure(ctx.Err())
		}
		return nil, err
	}
	defer func() {
		if err != nil {
			l.RetainFailure(err)
		}
	}()
	if len(names) == 0 || len(names) > 4096 {
		return nil, fmt.Errorf("gradient group count")
	}
	return l.state.runSelectedTapeBindingsLease(ctx, t, bindings, compact, outputs, true, scale, l.maxBytes, false, true, l.reuse, l.shareZero, l, nil, names)
}

// BeginChildTape pins read-only child outputs without accumulation. Its complete
// source charge joins every other live source in the root's combined ceiling.
// Parent and all explicitly borrowed producer owners cannot close before child.
// Errors after parent execution poison the whole tree; a non-nil partial child
// stays root-owned for coordinated recovery, never blind cleanup/retry.
func (l *ResidentTapeLease) BeginChildTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, edges []ResidentLeaseEdge) (child *ResidentTapeLease, err error) {
	if l == nil {
		return nil, fmt.Errorf("nil parent lease")
	}
	if err = l.admit(ctx, l.state); err != nil {
		if ctx != nil && ctx.Err() != nil {
			l.RetainFailure(ctx.Err())
		}
		return nil, err
	}
	defer func() {
		if err != nil {
			l.RetainFailure(err)
		}
	}()
	if len(l.root.live) >= 4096 {
		return nil, fmt.Errorf("resident lease owner ceiling")
	}
	s := l.state
	s.borrowed = true
	defer func() { s.borrowed = false }()
	if bindings == nil {
		err = t.Validate(l.maxBytes)
	} else {
		err = ValidateResidentTapeBindings(t, bindings, l.maxBytes)
	}
	if err != nil {
		return nil, err
	}
	bytes, err := leaseTapeBytes(t, l.maxBytes-l.root.liveBytes)
	if err != nil {
		return nil, err
	}
	if len(outputs) == 0 || len(outputs) > 4096 {
		return nil, fmt.Errorf("child pinned output count")
	}
	declared := map[int]bool{}
	for _, id := range t.Outputs {
		declared[id] = true
	}
	pinned := map[int]int{}
	for _, id := range outputs {
		if !declared[id] {
			return nil, fmt.Errorf("child output not declared")
		}
		if _, ok := pinned[id]; ok {
			return nil, fmt.Errorf("child duplicate root")
		}
		pinned[id] = id
	}
	bound := map[int]ResidentBuffer{}
	for name, b := range bindings {
		i, ok := s.index[name]
		if !ok {
			return nil, fmt.Errorf("child unknown parameter")
		}
		n, _ := t.Tensors[b.Parameter].Elements()
		if n != s.entries[i].n {
			return nil, fmt.Errorf("child parameter extent")
		}
		bound[b.Parameter] = s.entries[i].parameter
	}
	edgeBound, err := l.consumerBindings(ctx, t, edges, bound)
	if err != nil {
		return nil, err
	}
	for id, b := range edgeBound {
		bound[id] = b
	}
	source := copyLeaseMetadata(t)
	pruned, mapping, err := PruneTape(t, outputs, l.maxBytes)
	if err != nil {
		return nil, err
	}
	rebound := map[int]ResidentBuffer{}
	for id, b := range bound {
		if mapping[id] >= 0 {
			rebound[mapping[id]] = b
		}
	}
	for id := range pinned {
		pinned[id] = mapping[id]
	}
	dependencies := map[*ResidentTapeLease]bool{l: true}
	for _, e := range edges {
		if mapping[e.Input] >= 0 {
			dependencies[e.Source] = true
		}
	}
	child = &ResidentTapeLease{state: s, parent: l, root: l.root, source: source, outputs: pinned, logicalBytes: bytes, maxBytes: l.maxBytes, reuse: l.reuse, shareZero: l.shareZero, dependencies: dependencies}
	// Register before allocation so even partial construction retains ownership.
	l.root.live[child] = true
	l.root.liveBytes += bytes
	if child.shareZero {
		child.producer, err = s.driver.(ResidentZeroTapeDriver).NewZeroBoundTape(ctx, pruned, rebound, l.maxBytes)
	} else if child.reuse {
		child.producer, err = s.driver.(ResidentReuseTapeDriver).NewReuseBoundTape(ctx, pruned, rebound, l.maxBytes)
	} else {
		child.producer, err = s.driver.(ResidentTapeDriver).NewBoundTape(ctx, pruned, rebound, l.maxBytes)
	}
	if err != nil {
		return child, err
	}
	if child.producer == nil {
		err = fmt.Errorf("resident child nil producer")
		return child, err
	}
	err = child.producer.RunDevice(ctx)
	return child, err
}
