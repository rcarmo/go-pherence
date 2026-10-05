package training

import (
	"context"
	"errors"
	"fmt"
	"sort"
)

// ResidentTapeBinding identifies a read-only parameter tensor and its gradient
// tensor in a validated graph. Only constant parameter inputs may be rebound;
// graph intermediates/gradients remain tape-owned until accumulation completes.
type ResidentTapeBinding struct{ Parameter, Gradient int }

// ResidentBoundTape retains borrowed parameter owners without freeing them.
// Run executes commands only; downloads are explicitly selected afterwards.
// OutputBuffer exposes a borrowed contiguous tensor only for accumulation
// while this tape remains alive. Native failure/cleanup returns a retained
// owner to the coordinator, never automatically retries uncertain work.
type ResidentBoundTape interface {
	RunDevice(context.Context) error
	OutputBuffer(int) (ResidentBuffer, error)
	DownloadOutputs(context.Context, []int) ([][]float32, error)
	Close() error
}

// ResidentTapeDriver requires all borrowed inputs to be disjoint allocations.
// ValidateAccumulate is pure host admission of exact extents/output aliases;
// coordinator admits the whole accumulation pass before any gradient mutation.
type ResidentTapeDriver interface {
	NewBoundTape(context.Context, Tape, map[int]ResidentBuffer, int64) (ResidentBoundTape, error)
	ValidateAccumulate(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, float32) error
}

// ResidentReuseTapeDriver explicitly opts into intra-tape lifetime storage.
type ResidentReuseTapeDriver interface {
	NewReuseBoundTape(context.Context, Tape, map[int]ResidentBuffer, int64) (ResidentBoundTape, error)
}

// ResidentZeroTapeDriver explicitly enables immutable zero-constant sharing.
type ResidentZeroTapeDriver interface {
	NewZeroBoundTape(context.Context, Tape, map[int]ResidentBuffer, int64) (ResidentBoundTape, error)
}

func ValidateResidentTapeBindings(t Tape, bindings map[string]ResidentTapeBinding, maxBytes int64) error {
	return validateResidentTapeBindings(t, bindings, nil, maxBytes)
}
func validateResidentTapeBindings(t Tape, bindings map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, maxBytes int64) error {
	if err := t.Validate(maxBytes); err != nil {
		return err
	}
	if len(bindings) == 0 || len(bindings) > 4096 {
		return fmt.Errorf("resident tape bindings count")
	}
	declared := map[int]bool{}
	for _, id := range t.Outputs {
		declared[id] = true
	}
	used := map[int]bool{}
	// Reject compact output (including reshape aliases) in ordinary bindings,
	// even when rows happens to equal vocabulary and extents match. Build the
	// marker lazily so legacy graphs without mode3 allocate no extra map.
	var compactOutputs map[int]bool
	for _, c := range t.Commands {
		if c.Op == "embedding" && c.Mode == 3 {
			if compactOutputs == nil {
				compactOutputs = map[int]bool{}
			}
			compactOutputs[c.Output] = true
		} else if c.Op == "reshape" && compactOutputs[c.Inputs[0]] {
			compactOutputs[c.Output] = true
		}
	}
	for name, b := range bindings {
		if name == "" || len(name) > 512 || b.Parameter < 0 || b.Parameter >= len(t.Tensors) || b.Gradient < 0 || b.Gradient >= len(t.Tensors) || b.Gradient == b.Parameter || !declared[b.Gradient] || used[b.Parameter] {
			return fmt.Errorf("resident tape binding name/id/duplicate")
		}
		used[b.Parameter] = true
		parameter, gradient := t.Tensors[b.Parameter], t.Tensors[b.Gradient]
		n, _ := parameter.Elements()
		g, _ := gradient.Elements()
		_, isCompact := compact[name]
		if !isCompact && compactOutputs[b.Gradient] {
			return fmt.Errorf("compact gradient requires explicit binding")
		}
		if parameter.Values == nil || (!isCompact && n != g) {
			return fmt.Errorf("resident tape binding constant/gradient extent")
		}
	}
	if compact != nil {
		return validateResidentCompactBindings(t, bindings, compact)
	}
	return nil
}

// RunTape binds current resident parameters read-only, executes one tape, then
// optionally accumulates its parameter gradients on device. Only requested
// tensor IDs are downloaded. A non-accumulating forward/endpoint pass never
// contributes parameter gradients (minimal-LSD stop-gradient boundary).
// Logical tape admission remains conservative and includes borrowed inputs;
// maxBytes is per-tape, additional to resident admission. Construction and
// execution are single-owner, non-concurrent. The borrow blocks Update,
// Snapshot, Close and re-entrant accumulation for the complete tape lifetime.
func (s *ResidentState) RunTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, accumulate bool, scale float32, maxBytes int64) ([][]float32, error) {
	return s.runTape(ctx, t, bindings, outputs, accumulate, scale, maxBytes, false, false, false, false)
}

// RunUnboundTape executes a loss/utility tape under the same borrow, terminal
// retention and download validation as resident parameter tapes. It cannot
// accumulate parameters; there are no fake binding/gradient placeholders.
func (s *ResidentState) RunUnboundTape(ctx context.Context, t Tape, outputs []int, maxBytes int64) ([][]float32, error) {
	return s.runTape(ctx, t, nil, outputs, false, 1, maxBytes, true, false, false, false)
}

// RunPrunedTape preserves full source admission and all binding inventory checks,
// then executes only requested outputs plus accumulated parameter gradients.
func (s *ResidentState) RunPrunedTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, accumulate bool, scale float32, maxBytes int64) ([][]float32, error) {
	return s.runTape(ctx, t, bindings, outputs, accumulate, scale, maxBytes, false, true, false, false)
}
func (s *ResidentState) RunPrunedUnboundTape(ctx context.Context, t Tape, outputs []int, maxBytes int64) ([][]float32, error) {
	return s.runTape(ctx, t, nil, outputs, false, 1, maxBytes, true, true, false, false)
}

// RunReusePrunedTape preserves full-source/binding admission and terminal
// retention while selecting explicit intra-tape lifetime allocation.
func (s *ResidentState) RunReusePrunedTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, accumulate bool, scale float32, maxBytes int64) ([][]float32, error) {
	return s.runTape(ctx, t, bindings, outputs, accumulate, scale, maxBytes, false, true, true, false)
}
func (s *ResidentState) RunReusePrunedUnboundTape(ctx context.Context, t Tape, outputs []int, maxBytes int64) ([][]float32, error) {
	return s.runTape(ctx, t, nil, outputs, false, 1, maxBytes, true, true, true, false)
}
func (s *ResidentState) RunZeroPrunedTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, accumulate bool, scale float32, maxBytes int64) ([][]float32, error) {
	return s.runTape(ctx, t, bindings, outputs, accumulate, scale, maxBytes, false, true, true, true)
}

// RunGradientGroupTape admits the complete original source and inventory,
// prunes to requested boundary/selected gradient roots, then accumulates once.
func (s *ResidentState) RunGradientGroupTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, outputs []int, names []string, scale float32, maxBytes int64) ([][]float32, error) {
	if len(names) == 0 || len(names) > 4096 {
		return nil, fmt.Errorf("gradient group count")
	}
	return s.runSelectedTapeBindingsLease(ctx, t, bindings, compact, outputs, true, scale, maxBytes, false, true, true, true, nil, nil, names)
}

func (s *ResidentState) RunZeroPrunedUnboundTape(ctx context.Context, t Tape, outputs []int, maxBytes int64) ([][]float32, error) {
	return s.runTape(ctx, t, nil, outputs, false, 1, maxBytes, true, true, true, true)
}
func (s *ResidentState) runTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, outputs []int, accumulate bool, scale float32, maxBytes int64, unbound, prune, reuse, shareZero bool) ([][]float32, error) {
	return s.runTapeBindings(ctx, t, bindings, nil, outputs, accumulate, scale, maxBytes, unbound, prune, reuse, shareZero)
}
func (s *ResidentState) runTapeBindings(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, compactBindings map[string]ResidentCompactEmbeddingBinding, outputs []int, accumulate bool, scale float32, maxBytes int64, unbound, prune, reuse, shareZero bool) ([][]float32, error) {
	return s.runTapeBindingsLease(ctx, t, bindings, compactBindings, outputs, accumulate, scale, maxBytes, unbound, prune, reuse, shareZero, nil, nil)
}
func (s *ResidentState) runTapeBindingsLease(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, compactBindings map[string]ResidentCompactEmbeddingBinding, outputs []int, accumulate bool, scale float32, maxBytes int64, unbound, prune, reuse, shareZero bool, lease *ResidentTapeLease, edges []ResidentLeaseEdge) ([][]float32, error) {
	return s.runSelectedTapeBindingsLease(ctx, t, bindings, compactBindings, outputs, accumulate, scale, maxBytes, unbound, prune, reuse, shareZero, lease, edges, nil)
}
func (s *ResidentState) runSelectedTapeBindingsLease(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, compactBindings map[string]ResidentCompactEmbeddingBinding, outputs []int, accumulate bool, scale float32, maxBytes int64, unbound, prune, reuse, shareZero bool, lease *ResidentTapeLease, edges []ResidentLeaseEdge, selectedNames []string) (values [][]float32, err error) {
	if lease != nil {
		err = lease.admit(ctx, s)
	} else {
		err = s.admit(ctx)
	}
	if err != nil {
		return nil, err
	}
	if lease != nil {
		// Protect producer output borrowing and consumer preflight as well as
		// execution against re-entrant lease close/download/consumer calls.
		s.borrowed = true
		defer func() { s.borrowed = false }()
	}
	if shareZero {
		if _, ok := s.driver.(ResidentZeroTapeDriver); !ok {
			return nil, fmt.Errorf("resident driver lacks zero-constant sharing")
		}
	}
	if reuse {
		if _, ok := s.driver.(ResidentReuseTapeDriver); !ok {
			return nil, fmt.Errorf("resident driver lacks lifetime storage")
		}
	}
	driver, ok := s.driver.(ResidentTapeDriver)
	if !ok {
		return nil, fmt.Errorf("resident driver has no tape binding support")
	}
	if !residentFinite(scale) {
		return nil, fmt.Errorf("resident tape accumulation scale")
	}
	var compactDriver ResidentCompactEmbeddingDriver
	if compactBindings != nil {
		var supported bool
		compactDriver, supported = s.driver.(ResidentCompactEmbeddingDriver)
		if !supported {
			return nil, fmt.Errorf("resident driver lacks compact embedding accumulation")
		}
	}
	if unbound {
		err = t.Validate(maxBytes)
	} else {
		err = validateResidentTapeBindings(t, bindings, compactBindings, maxBytes)
	}
	if err != nil {
		return nil, err
	}
	// Requested downloads must be declared graph outputs. No uninitialised or
	// unrelated mutable owner can escape through a tensor ID.
	declared := map[int]bool{}
	for _, id := range t.Outputs {
		declared[id] = true
	}
	for _, id := range outputs {
		if !declared[id] {
			return nil, fmt.Errorf("resident tape requested output")
		}
	}
	bound := map[int]ResidentBuffer{}
	names := make([]string, 0, len(bindings))
	for name, b := range bindings {
		index, exists := s.index[name]
		if !exists {
			return nil, fmt.Errorf("resident tape unknown parameter %s", name)
		}
		n, _ := t.Tensors[b.Parameter].Elements()
		if n != s.entries[index].n {
			return nil, fmt.Errorf("resident tape parameter %s extent", name)
		}
		bound[b.Parameter] = s.entries[index].parameter
		names = append(names, name)
	}
	sort.Strings(names)
	allNames := names
	if selectedNames != nil {
		if !accumulate || !prune || unbound || len(selectedNames) == 0 {
			return nil, fmt.Errorf("selected gradient execution requires bound pruning/accumulation")
		}
		seen := make(map[string]bool, len(selectedNames))
		for _, name := range selectedNames {
			if _, ok := bindings[name]; !ok || seen[name] {
				return nil, fmt.Errorf("selected gradient unknown/duplicate")
			}
			seen[name] = true
		}
		names = append([]string(nil), selectedNames...)
		sort.Strings(names)
	}
	var edgeBindings map[int]ResidentBuffer
	if lease != nil {
		edgeBindings, err = lease.consumerBindings(ctx, t, edges, bound)
		if err != nil {
			return nil, err
		}
		for id, buffer := range edgeBindings {
			bound[id] = buffer
		}
	}
	if err = ValidateExternalTapeBindings(t, bound); err != nil {
		return nil, err
	}
	if prune {
		roots := append([]int(nil), outputs...)
		if accumulate {
			for _, name := range names {
				roots = append(roots, bindings[name].Gradient)
				if c, ok := compactBindings[name]; ok {
					roots = append(roots, c.Index)
				}
			}
		}
		if len(roots) > 0 {
			compact, mapping, e := PruneTape(t, roots, maxBytes)
			if e != nil {
				return nil, e
			}
			selected := make([]int, len(outputs))
			for i, id := range outputs {
				selected[i] = mapping[id]
			}
			rebound := map[int]ResidentBuffer{}
			rebindings := map[string]ResidentTapeBinding{}
			var remappedCompact map[string]ResidentCompactEmbeddingBinding
			if compactBindings != nil {
				remappedCompact = map[string]ResidentCompactEmbeddingBinding{}
			}
			for _, name := range allNames {
				b := bindings[name]
				if mapping[b.Parameter] >= 0 {
					rebound[mapping[b.Parameter]] = bound[b.Parameter]
				}
			}
			for _, name := range names {
				b := bindings[name]
				if accumulate {
					rebindings[name] = ResidentTapeBinding{Parameter: mapping[b.Parameter], Gradient: mapping[b.Gradient]}
					if c, ok := compactBindings[name]; ok {
						c.Index = mapping[c.Index]
						remappedCompact[name] = c
					}
				}
			}
			for id, buffer := range edgeBindings {
				if mapping[id] >= 0 {
					rebound[mapping[id]] = buffer
				}
			}
			t, bound, outputs = compact, rebound, selected
			if accumulate {
				bindings = rebindings
				compactBindings = remappedCompact
			}
		}
	}
	s.borrowed = true
	defer func() { s.borrowed = false }()
	var tape ResidentBoundTape
	if shareZero {
		tape, err = s.driver.(ResidentZeroTapeDriver).NewZeroBoundTape(ctx, t, bound, maxBytes)
	} else if reuse {
		tape, err = s.driver.(ResidentReuseTapeDriver).NewReuseBoundTape(ctx, t, bound, maxBytes)
	} else {
		tape, err = driver.NewBoundTape(ctx, t, bound, maxBytes)
	}
	if err != nil {
		if tape != nil {
			s.retained = append(s.retained, tape)
			s.terminal = err
		}
		return nil, err
	}
	if tape == nil {
		err = fmt.Errorf("resident driver returned nil tape")
		s.terminal = err
		return nil, err
	}
	defer func() {
		// A live producer already executed. Any consumer failure must retain
		// this consumer before unwinding, even pure late accumulation admission.
		if lease != nil && err != nil {
			lease.RetainFailure(err)
		}
		if s.terminal != nil {
			s.retained = append(s.retained, tape)
			return
		}
		if closeErr := tape.Close(); closeErr != nil {
			s.retained = append(s.retained, tape)
			s.terminal = closeErr
			err = errors.Join(err, closeErr)
			values = nil
		}
	}()
	if err = tape.RunDevice(ctx); err != nil {
		s.terminal = err
		return nil, err
	}
	if accumulate {
		sources := make(map[string]ResidentBuffer, len(names))
		var indices map[string]ResidentBuffer
		if compactBindings != nil {
			indices = make(map[string]ResidentBuffer, len(compactBindings))
		}
		for _, name := range names {
			source, e := tape.OutputBuffer(bindings[name].Gradient)
			if e != nil {
				s.terminal = e
				return nil, e
			}
			sources[name] = source
			if s.shared != nil {
				if c, compact := compactBindings[name]; compact {
					index, e := tape.OutputBuffer(c.Index)
					if e != nil {
						s.terminal = e
						return nil, e
					}
					indices[name] = index
				}
				continue
			}
			entry := &s.entries[s.index[name]]
			previous := entry.zero
			if entry.ready {
				previous = entry.gradient
			}
			if c, compact := compactBindings[name]; compact {
				index, e := tape.OutputBuffer(c.Index)
				if e != nil {
					s.terminal = e
					return nil, e
				}
				indices[name] = index
				err = compactDriver.ValidateCompactEmbeddingAccumulate(ctx, entry.scratch, previous, source, index, c.Rows, c.Width, c.Vocabulary, scale)
			} else {
				err = driver.ValidateAccumulate(ctx, entry.scratch, previous, source, entry.n, scale)
			}
			if err != nil {
				return nil, err
			}
		}
		if s.shared != nil {
			if err = s.sharedAccumulate(ctx, names, sources, indices, compactBindings, nil, scale); err != nil {
				s.terminal = err
				return nil, err
			}
		} else {
			for _, name := range names {
				entry := &s.entries[s.index[name]]
				previous := entry.zero
				if entry.ready {
					previous = entry.gradient
				}
				if c, compact := compactBindings[name]; compact {
					err = compactDriver.CompactEmbeddingAccumulate(ctx, entry.scratch, previous, sources[name], indices[name], c.Rows, c.Width, c.Vocabulary, scale)
				} else {
					err = s.driver.Accumulate(ctx, entry.scratch, previous, sources[name], entry.n, scale)
				}
				if err != nil {
					s.terminal = err
					return nil, err
				}
				entry.gradient, entry.scratch = entry.scratch, entry.gradient
				entry.ready = true
				s.dirty = true
			}
		}
	}
	values, err = tape.DownloadOutputs(ctx, outputs)
	if err != nil {
		s.terminal = err
		return nil, err
	}
	if len(values) != len(outputs) {
		err = fmt.Errorf("resident tape output count")
		s.terminal = err
		return nil, err
	}
	for i, x := range values {
		n, _ := t.Tensors[outputs[i]].Elements()
		if len(x) != n {
			err = fmt.Errorf("resident tape output extent")
			s.terminal = err
			return nil, err
		}
		for _, v := range x {
			if !residentFinite(v) {
				err = fmt.Errorf("resident tape nonfinite output")
				s.terminal = err
				return nil, err
			}
		}
	}
	return values, nil
}
