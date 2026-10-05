package training

import (
	"context"
	"fmt"
)

// ResidentTapeLease pins a completed read-only producer under one ResidentState.
// Only its consumer methods may execute/accumulate while it is live; public
// state update, snapshot, accumulation, tape execution and close stay blocked.
// All methods and borrowed work are caller-serialised, including Close versus
// RunTape. Successful driver execution/accumulation must be device-complete.
// No implicit gradients cross an edge. Kernel modules outlive the lease tree.
// The root owns every registered child, including partial failures. Parents and
// explicitly borrowed sibling owners cannot close until dependants close.
type ResidentTapeLease struct {
	state                  *ResidentState
	producer               ResidentBoundTape
	source                 Tape
	outputs                map[int]int // admitted source output -> pruned output
	logicalBytes, maxBytes int64
	reuse, shareZero       bool
	closed                 bool
	parent, root           *ResidentTapeLease
	live                   map[*ResidentTapeLease]bool // root-owned identity registry
	liveBytes              int64                       // complete sources of all live owners
	dependencies           map[*ResidentTapeLease]bool // live borrowed owners plus parent
}

// BeginPrunedTapeLease admits a complete producer and parameter inventory, then
// pins selected declared roots without accumulation. Consumer admission charges
// its complete source plus every live producer's complete source against maxBytes.
// A non-nil lease returned with an error retains uncertain producer resources.
func (s *ResidentState) BeginPrunedTapeLease(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, maxBytes int64) (*ResidentTapeLease, error) {
	return s.beginTapeLease(ctx, t, bindings, outputs, maxBytes, false, false)
}

// BeginZeroPrunedTapeLease additionally selects lifetime allocation and immutable
// positive-zero sharing. Borrowed parameter/edge constants remain protected.
func (s *ResidentState) BeginZeroPrunedTapeLease(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, maxBytes int64) (*ResidentTapeLease, error) {
	return s.beginTapeLease(ctx, t, bindings, outputs, maxBytes, true, true)
}
func (s *ResidentState) beginTapeLease(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, maxBytes int64, reuse, shareZero bool) (lease *ResidentTapeLease, err error) {
	if err = s.admit(ctx); err != nil {
		return nil, err
	}
	driver, ok := s.driver.(ResidentHandoffTapeDriver)
	if !ok {
		return nil, fmt.Errorf("resident driver lacks handoff preflight")
	}
	if reuse {
		if _, ok = s.driver.(ResidentReuseTapeDriver); !ok {
			return nil, fmt.Errorf("resident driver lacks lifetime storage")
		}
	}
	if shareZero {
		if _, ok = s.driver.(ResidentZeroTapeDriver); !ok {
			return nil, fmt.Errorf("resident driver lacks zero-constant sharing")
		}
	}
	if bindings == nil {
		err = t.Validate(maxBytes)
	} else {
		err = ValidateResidentTapeBindings(t, bindings, maxBytes)
	}
	if err != nil {
		return nil, err
	}
	bytes, err := leaseTapeBytes(t, maxBytes)
	if err != nil {
		return nil, err
	}
	if len(outputs) == 0 || len(outputs) > 4096 {
		return nil, fmt.Errorf("lease pinned output count")
	}
	declared := map[int]bool{}
	for _, id := range t.Outputs {
		declared[id] = true
	}
	pinned := map[int]int{}
	for _, id := range outputs {
		if !declared[id] {
			return nil, fmt.Errorf("lease output not declared")
		}
		if _, ok = pinned[id]; ok {
			return nil, fmt.Errorf("lease duplicate root")
		}
		pinned[id] = id
	}
	bound := map[int]ResidentBuffer{}
	for name, b := range bindings {
		i, ok := s.index[name]
		if !ok {
			return nil, fmt.Errorf("lease unknown parameter %s", name)
		}
		n, _ := t.Tensors[b.Parameter].Elements()
		if n != s.entries[i].n {
			return nil, fmt.Errorf("lease parameter extent")
		}
		bound[b.Parameter] = s.entries[i].parameter
	}
	if err = ValidateExternalTapeBindings(t, bound); err != nil {
		return nil, err
	}
	if err = driver.ValidateHandoffTape(ctx, t, bound, maxBytes); err != nil {
		return nil, err
	}
	// Admit/prune the complete source, consume constants synchronously below,
	// and retain frozen metadata only for future edge/extent provenance.
	source := copyLeaseMetadata(t)
	pruned, mapping, err := PruneTape(t, outputs, maxBytes)
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
	lease = &ResidentTapeLease{state: s, source: source, outputs: pinned, logicalBytes: bytes, maxBytes: maxBytes, reuse: reuse, shareZero: shareZero}
	lease.root = lease
	lease.live = map[*ResidentTapeLease]bool{lease: true}
	lease.liveBytes = bytes
	s.lease = lease
	s.borrowed = true
	defer func() {
		s.borrowed = false
		if err != nil {
			lease.RetainFailure(err)
		}
	}()
	if shareZero {
		lease.producer, err = s.driver.(ResidentZeroTapeDriver).NewZeroBoundTape(ctx, pruned, rebound, maxBytes)
	} else if reuse {
		lease.producer, err = s.driver.(ResidentReuseTapeDriver).NewReuseBoundTape(ctx, pruned, rebound, maxBytes)
	} else {
		lease.producer, err = driver.NewBoundTape(ctx, pruned, rebound, maxBytes)
	}
	if err != nil {
		return lease, err
	}
	if lease.producer == nil {
		err = fmt.Errorf("resident lease nil producer")
		return lease, err
	}
	err = lease.producer.RunDevice(ctx)
	return lease, err
}

// leaseTapeBytes requires already validated tensor extents, and uses subtraction
// before addition so the conservative full-source ceiling cannot overflow.
func leaseTapeBytes(t Tape, maxBytes int64) (int64, error) {
	if maxBytes <= 0 {
		return 0, fmt.Errorf("lease byte ceiling required")
	}
	var bytes int64
	for _, spec := range t.Tensors {
		n, err := spec.Elements()
		if err != nil {
			return 0, err
		}
		part := int64(n) * 4
		if part > maxBytes-bytes {
			return 0, fmt.Errorf("lease combined byte ceiling")
		}
		bytes += part
	}
	return bytes, nil
}
func (l *ResidentTapeLease) admit(ctx context.Context, s *ResidentState) error {
	if l == nil || s == nil || l.state != s || l.root == nil || s.lease != l.root || !l.root.live[l] || l.closed || s.closed || s.terminal != nil || s.borrowed || l.producer == nil || ctx == nil {
		return fmt.Errorf("unavailable/foreign/terminal resident lease")
	}
	return ctx.Err()
}

// RunTape executes one pruned consumer under this lease. Parameter gradients
// accumulate only when explicitly requested, with the supplied scale once.
// A nil binding inventory permits an unbound read-only utility consumer.
func (l *ResidentTapeLease) RunTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, edges []TapeHandoffEdge, accumulate bool, scale float32) ([][]float32, error) {
	return l.RunCompactTape(ctx, t, bindings, nil, outputs, edges, accumulate, scale)
}

// RunCompactTape preserves explicit compact-gradient index provenance and roots.
// Consumer output IDs and edges refer to the complete, unpruned input graphs.
func (l *ResidentTapeLease) RunCompactTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, outputs []int, edges []TapeHandoffEdge, accumulate bool, scale float32) (values [][]float32, err error) {
	if l == nil {
		return nil, fmt.Errorf("nil resident lease")
	}
	if err = l.admit(ctx, l.state); err != nil {
		// Cancellation with a live producer abandons a training boundary;
		// keep it terminal rather than permitting publication or retry.
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
	return l.state.runTapeBindingsLease(ctx, t, bindings, compact, outputs, accumulate, scale, l.maxBytes, bindings == nil, true, l.reuse, l.shareZero, l, l.ownEdges(edges))
}
func (l *ResidentTapeLease) consumerBindings(ctx context.Context, t Tape, edges []ResidentLeaseEdge, bound map[int]ResidentBuffer) (map[int]ResidentBuffer, error) {
	// t and parameter inventory were already fully validated by the shared runner.
	if _, err := leaseTapeBytes(t, l.maxBytes-l.root.liveBytes); err != nil {
		return nil, err
	}
	// An authorised no-edge consumer may accumulate unrelated branches while
	// the producer stays pinned; it still charges the combined live ceiling.
	if len(edges) > 4096 {
		return nil, fmt.Errorf("lease edge count")
	}
	families := map[*ResidentTapeLease][]int{}
	used := map[int]bool{}
	sourceUsed := map[*ResidentTapeLease]map[int]bool{}
	for _, e := range edges {
		source := e.Source
		if source == nil || source.state != l.state || source.root != l.root || !l.root.live[source] || source.closed || source.producer == nil {
			return nil, fmt.Errorf("foreign/expired lease edge source")
		}
		family, exists := families[source]
		if !exists {
			family = make([]int, len(source.source.Tensors))
			for id := range family {
				family[id] = id
			}
			for _, c := range source.source.Commands {
				if c.Op == "reshape" {
					family[c.Output] = family[c.Inputs[0]]
				}
			}
			families[source] = family
			sourceUsed[source] = map[int]bool{}
		}
		if _, ok := source.outputs[e.Output]; !ok {
			return nil, fmt.Errorf("lease source root not pinned")
		}
		if e.Input < 0 || e.Input >= len(t.Tensors) || (t.Tensors[e.Input].Values == nil && !t.Tensors[e.Input].External) || used[e.Input] || sourceUsed[source][family[e.Output]] {
			return nil, fmt.Errorf("lease constant target/duplicate edge")
		}
		if _, ok := bound[e.Input]; ok {
			return nil, fmt.Errorf("lease parameter/edge conflict")
		}
		a, _ := source.source.Tensors[e.Output].Elements()
		b, _ := t.Tensors[e.Input].Elements()
		if a != b {
			return nil, fmt.Errorf("lease edge extent")
		}
		used[e.Input] = true
		sourceUsed[source][family[e.Output]] = true
	}
	edgesBound := make(map[int]ResidentBuffer, len(edges))
	all := make(map[int]ResidentBuffer, len(bound)+len(edges))
	for id, b := range bound {
		all[id] = b
	}
	for _, e := range edges {
		b, err := e.Source.producer.OutputBuffer(e.Source.outputs[e.Output])
		if err != nil {
			return nil, err
		}
		edgesBound[e.Input] = b
		all[e.Input] = b
	}
	if err := ValidateExternalTapeBindings(t, all); err != nil {
		return nil, err
	}
	if err := l.state.driver.(ResidentHandoffTapeDriver).ValidateHandoffTape(ctx, t, all, l.maxBytes); err != nil {
		return nil, err
	}
	return edgesBound, nil
}

// DownloadOutputs keeps the lease live for later row consumers. It accepts only
// pinned original source IDs; host copies are owned by the caller.
func (l *ResidentTapeLease) DownloadOutputs(ctx context.Context, ids []int) (values [][]float32, err error) {
	if l == nil {
		return nil, fmt.Errorf("nil resident lease")
	}
	if err = l.admit(ctx, l.state); err != nil {
		if ctx != nil && ctx.Err() != nil {
			l.RetainFailure(ctx.Err())
		}
		return nil, err
	}
	mapped := make([]int, len(ids))
	for i, id := range ids {
		var ok bool
		mapped[i], ok = l.outputs[id]
		if !ok {
			err = fmt.Errorf("lease download not pinned")
			l.RetainFailure(err)
			return nil, err
		}
	}
	l.state.borrowed = true
	defer func() {
		l.state.borrowed = false
		if err != nil {
			l.RetainFailure(err)
		}
	}()
	values, err = l.producer.DownloadOutputs(ctx, mapped)
	if err != nil {
		return nil, err
	}
	if len(values) != len(ids) {
		return nil, fmt.Errorf("lease output count")
	}
	for i, id := range ids {
		n, _ := l.source.Tensors[id].Elements()
		if len(values[i]) != n {
			return nil, fmt.Errorf("lease output extent")
		}
		for _, v := range values[i] {
			if !residentFinite(v) {
				return nil, fmt.Errorf("lease nonfinite output")
			}
		}
	}
	return values, nil
}

// RetainFailure fences every state publication/mutation after a late failure.
// The state retains this lease and any failed consumers; it never retries/free.
func (l *ResidentTapeLease) RetainFailure(err error) error {
	if err != nil && l != nil && l.state != nil && l.root != nil && l.state.lease == l.root && l.root.live[l] && !l.closed && l.state.terminal == nil {
		l.state.terminal = err
	}
	return err
}

// Close releases the producer only after synchronous consumers/accumulation and
// retained child/dependant owners finish. A live-dependant rejection is safe and
// leaves the tree usable; a native close error retains all owners terminally.
func (l *ResidentTapeLease) Close() (err error) {
	if l == nil || l.closed {
		return nil
	}
	if err = l.admit(context.Background(), l.state); err != nil {
		return err
	}
	for owner := range l.root.live {
		if owner != l && (owner.parent == l || owner.dependencies[l]) {
			return fmt.Errorf("resident lease has live child/dependent")
		}
	}
	l.state.borrowed = true
	defer func() {
		l.state.borrowed = false
		if err != nil {
			l.RetainFailure(err)
		}
	}()
	if err = l.producer.Close(); err != nil {
		return err
	}
	l.closed = true
	delete(l.root.live, l)
	l.root.liveBytes -= l.logicalBytes
	if l == l.root {
		l.state.lease = nil
	}
	return nil
}
