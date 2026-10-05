package training

import (
	"context"
	"fmt"
	"reflect"
	"sort"
)

// ResidentSharedWorkspaceDriver is optional. Prefix is pure metadata creation:
// it returns an exact bounded alias, never a new allocation or dispatch. Every
// operation must validate the supplied SharedResidentAccess token and
// validate owning driver/context/allocation identity, checked physical ranges,
// disjointness and live parents. Copy is a completed exact-bit transfer, never
// arithmetic emulation. Validation is pure; execution is device-complete.
// Existing/native drivers need not implement this generic experimental contract.
type ResidentSharedWorkspaceDriver interface {
	ResidentDriver
	// ValidateSharedInventory checks exact physical extent, driver/context and
	// allocation identity/disjointness for all owners, before initial uploads.
	ValidateSharedInventory(context.Context, []ResidentBuffer, []int) error
	Prefix(context.Context, ResidentBuffer, int, *SharedResidentAccess) (ResidentBuffer, error)
	ValidateSharedAccumulate(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, float32) error
	ValidateCopy(context.Context, ResidentBuffer, ResidentBuffer, int) error
	Copy(context.Context, ResidentBuffer, ResidentBuffer, int) error
	ValidateNormStats(context.Context, ResidentBuffer, ResidentBuffer, int, bool) error
}

type SharedResidentPlan struct {
	Bytes                 int64
	Elements              int64
	MaxElements, MaxStats int
	Parameters            int
}

// SharedResidentBytes validates complete numerical inventory as before; charges
// P/G/M/V/EMA per parameter, scratch/branch/zero/maxstats once and 48 globals.
// It does not estimate native allocator/tape/host coexistence or change CLI caps.
func SharedResidentBytes(snapshot ResidentSnapshot) (SharedResidentPlan, error) {
	if _, err := ResidentBytes(snapshot); err != nil {
		return SharedResidentPlan{}, err
	}
	p := SharedResidentPlan{Parameters: len(snapshot.Parameters)}
	for _, v := range snapshot.Parameters {
		n := len(v.Parameter)
		stats, _ := NormStatsShape(n)
		p.Elements += int64(n)
		p.MaxElements = max(p.MaxElements, n)
		p.MaxStats = max(p.MaxStats, stats)
	}
	if p.Elements > (1<<38)/4 {
		return SharedResidentPlan{}, fmt.Errorf("shared resident mutable-state ceiling")
	}
	if err := NormClipShape(p.Elements, 1); err != nil {
		return SharedResidentPlan{}, err
	}
	p.Bytes = 20*p.Elements + 12*int64(p.MaxElements) + 4*int64(p.MaxStats) + 48
	return p, nil
}

// SharedTapeResidentBytes omits only the host-upload branch reservation. Its
// constructor rejects host Accumulate; all gradients must come from bound tapes.
// The generic shared constructor retains its full host-upload contract.
func SharedTapeResidentBytes(snapshot ResidentSnapshot) (SharedResidentPlan, error) {
	p, err := SharedResidentBytes(snapshot)
	if err != nil {
		return SharedResidentPlan{}, err
	}
	p.Bytes -= 4 * int64(p.MaxElements)
	return p, nil
}

type residentSharedWorkspace struct {
	tapeOnly                     bool
	state                        *ResidentState
	driver                       ResidentSharedWorkspaceDriver
	scratch, branch, zero, stats ResidentBuffer
	generation                   uint64
	active                       bool
	ctx                          context.Context
}

// SharedResidentAccess is the mandatory immutable permission token captured by
// a driver-created bounded prefix. The prefix implements ResidentBuffer and
// MUST validate this token on upload/download/close and every driver operation.
// No unwrap/resolver returns the parent/raw buffer, and the prefix exposes no
// method that frees or escapes its physical storage. Native adapters must enforce
// token access as part of their capability; none are implemented here.
// A fresh pointer context per operation gives enforceable identity even when
// the caller uses a comparable value-type Context. It forwards cancellation.
type sharedOperationContext struct{ context.Context }

type SharedResidentAccess struct {
	workspace  *residentSharedWorkspace
	generation uint64
	writable   bool
}

func (a *SharedResidentAccess) Validate(ctx context.Context, driver ResidentDriver, write bool) error {
	if a == nil || a.workspace == nil || ctx == nil || driver == nil {
		return fmt.Errorf("nil shared access/context/driver")
	}
	w := a.workspace
	op, ok := ctx.(*sharedOperationContext)
	if !ok || op != w.ctx {
		return fmt.Errorf("foreign shared operation context")
	}
	if reflect.ValueOf(driver).Kind() != reflect.Pointer || reflect.ValueOf(driver).IsNil() || reflect.ValueOf(driver).Pointer() != reflect.ValueOf(w.driver).Pointer() {
		return fmt.Errorf("foreign shared prefix driver")
	}
	if !w.active || a.generation != w.generation || ctx != w.ctx || w.state.closed || w.state.terminal != nil || (write && !a.writable) {
		return fmt.Errorf("expired/foreign/read-only shared prefix")
	}
	return ctx.Err()
}

func (w *residentSharedWorkspace) begin(ctx context.Context) error {
	if ctx == nil || w.active || w.state.closed || w.state.terminal != nil {
		return fmt.Errorf("shared workspace reentry/owner")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if w.generation == ^uint64(0) {
		return fmt.Errorf("shared workspace generation ceiling")
	}
	w.generation++
	w.active = true
	w.ctx = &sharedOperationContext{ctx}
	return nil
}
func (w *residentSharedWorkspace) end() { w.active = false; w.ctx = nil }
func (w *residentSharedWorkspace) view(parent ResidentBuffer, n int, write bool) (ResidentBuffer, error) {
	if !w.active || n <= 0 {
		return nil, fmt.Errorf("shared prefix outside operation")
	}
	if parent == w.zero && write {
		return nil, fmt.Errorf("immutable shared zero")
	}
	access := &SharedResidentAccess{workspace: w, generation: w.generation, writable: write}
	b, err := w.driver.Prefix(w.ctx, parent, n, access)
	if err != nil {
		return nil, err
	}
	if b == nil {
		return nil, fmt.Errorf("nil shared prefix")
	}
	return b, nil
}

// NewSharedWorkspaceResidentState is a generic experimental constructor only.
// It enables global/device-scale/state checks, keeps stable G, and requires
// optional view/copy/stat validation capability before allocation. No native
// adapter/CLI route is supplied. Partial failures return retained terminal owner.
func NewSharedWorkspaceResidentState(ctx context.Context, driver ResidentDriver, snapshot ResidentSnapshot, maxBytes int64) (*ResidentState, error) {
	return newSharedWorkspaceResidentState(ctx, driver, snapshot, maxBytes, false)
}

// NewTapeOnlySharedWorkspaceResidentState explicitly admits gradients only from
// bound tapes. No host-upload branch is reserved and Accumulate rejects before
// any upload or state mutation. Tape, snapshot, failure and ownership contracts
// remain unchanged; there is no implicit fallback to host accumulation.
func NewTapeOnlySharedWorkspaceResidentState(ctx context.Context, driver ResidentDriver, snapshot ResidentSnapshot, maxBytes int64) (*ResidentState, error) {
	return newSharedWorkspaceResidentState(ctx, driver, snapshot, maxBytes, true)
}

func newSharedWorkspaceResidentState(ctx context.Context, driver ResidentDriver, snapshot ResidentSnapshot, maxBytes int64, tapeOnly bool) (s *ResidentState, err error) {
	if ctx == nil || driver == nil {
		return nil, fmt.Errorf("shared context/driver required")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	d, ok := driver.(ResidentSharedWorkspaceDriver)
	if !ok || reflect.ValueOf(driver).Kind() != reflect.Pointer || reflect.ValueOf(driver).IsNil() {
		return nil, fmt.Errorf("resident driver lacks shared workspace capability/identity")
	}
	if _, ok := driver.(ResidentDeviceScaleDriver); !ok {
		return nil, fmt.Errorf("shared driver lacks device scale")
	}
	if _, ok := driver.(ResidentGlobalNormDriver); !ok {
		return nil, fmt.Errorf("shared driver lacks global norm")
	}
	var plan SharedResidentPlan
	var planErr error
	if tapeOnly {
		plan, planErr = SharedTapeResidentBytes(snapshot)
	} else {
		plan, planErr = SharedResidentBytes(snapshot)
	}
	err = planErr
	if err != nil {
		return nil, err
	}
	if maxBytes <= 0 || plan.Bytes > maxBytes {
		return nil, fmt.Errorf("shared resident byte ceiling")
	}
	s = &ResidentState{driver: driver, config: snapshot.Config, step: snapshot.Step, index: map[string]int{}, deviceGlobalNorm: true, deviceScale: true, deviceStateCheck: true}
	w := &residentSharedWorkspace{state: s, driver: d, tapeOnly: tapeOnly}
	s.shared = w
	var extents []int
	var uploads []struct {
		b      ResidentBuffer
		values []float32
	}
	allocate := func(dst *ResidentBuffer, n int, values []float32) error {
		b, e := driver.NewBuffer(ctx, n)
		if b != nil {
			s.owned = append(s.owned, b)
			*dst = b
		}
		if e != nil {
			return e
		}
		if b == nil {
			return fmt.Errorf("nil shared allocation")
		}
		extents = append(extents, n)
		if values != nil {
			uploads = append(uploads, struct {
				b      ResidentBuffer
				values []float32
			}{b, values})
		}
		return nil
	}
	defer func() {
		if err != nil {
			s.terminal = err
		}
	}()
	names := make([]string, 0, len(snapshot.Parameters))
	for name := range snapshot.Parameters {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		p := snapshot.Parameters[name]
		entry := residentEntry{name: name, n: len(p.Parameter)}
		for _, slot := range []struct {
			dst *ResidentBuffer
			v   []float32
		}{{&entry.parameter, p.Parameter}, {&entry.gradient, nil}, {&entry.m, p.M}, {&entry.v, p.V}, {&entry.ema, p.EMA}} {
			if err = allocate(slot.dst, entry.n, slot.v); err != nil {
				return s, err
			}
		}
		s.index[name] = len(s.entries)
		s.entries = append(s.entries, entry)
	}
	for _, slot := range []struct {
		dst  *ResidentBuffer
		n    int
		zero bool
	}{{&w.scratch, plan.MaxElements, false}, {&w.branch, plan.MaxElements, false}, {&w.zero, plan.MaxElements, true}, {&w.stats, plan.MaxStats, false}, {&s.globalA, 4, false}, {&s.globalB, 4, false}, {&s.globalClip, 4, false}} {
		if tapeOnly && slot.dst == &w.branch {
			continue
		}
		var v []float32
		if slot.zero {
			v = make([]float32, slot.n)
		}
		if err = allocate(slot.dst, slot.n, v); err != nil {
			return s, err
		}
	}
	if err = d.ValidateSharedInventory(ctx, s.owned, extents); err != nil {
		return s, err
	}
	for _, u := range uploads {
		if err = u.b.Upload(ctx, u.values); err != nil {
			return s, err
		}
	}
	return s, nil
}

func (s *ResidentState) sharedAccumulate(ctx context.Context, names []string, sources, indices map[string]ResidentBuffer, compact map[string]ResidentCompactEmbeddingBinding, host map[string][]float32, scale float32) error {
	w := s.shared
	if w.tapeOnly && host != nil {
		return fmt.Errorf("tape-only shared workspace rejects host branches")
	}
	operation := func(name string, execute bool) error {
		if err := w.begin(ctx); err != nil {
			return err
		}
		defer w.end()
		opctx := w.ctx
		e := &s.entries[s.index[name]]
		dst, err := w.view(w.scratch, e.n, true)
		if err != nil {
			return err
		}
		previous := e.gradient
		if !e.ready {
			previous, err = w.view(w.zero, e.n, false)
			if err != nil {
				return err
			}
		}
		source := sources[name]
		if host != nil {
			source, err = w.view(w.branch, e.n, true)
			if err != nil {
				return err
			}
		}
		if !execute {
			if c, ok := compact[name]; ok {
				err = s.driver.(ResidentCompactEmbeddingDriver).ValidateCompactEmbeddingAccumulate(opctx, dst, previous, source, indices[name], c.Rows, c.Width, c.Vocabulary, scale)
			} else {
				err = w.driver.ValidateSharedAccumulate(opctx, dst, previous, source, e.n, scale)
			}
			if err != nil {
				return err
			}
			return w.driver.ValidateCopy(opctx, e.gradient, dst, e.n)
		}
		if host != nil {
			if err := source.Upload(opctx, host[name]); err != nil {
				return err
			}
		}
		if c, ok := compact[name]; ok {
			err = s.driver.(ResidentCompactEmbeddingDriver).CompactEmbeddingAccumulate(opctx, dst, previous, source, indices[name], c.Rows, c.Width, c.Vocabulary, scale)
		} else {
			err = s.driver.Accumulate(opctx, dst, previous, source, e.n, scale)
		}
		if err != nil {
			return err
		}
		if err = w.driver.Copy(opctx, e.gradient, dst, e.n); err != nil {
			return err
		}
		e.ready = true
		s.dirty = true
		return nil
	}
	for _, name := range names {
		if err := operation(name, false); err != nil {
			return err
		}
	}
	for _, name := range names {
		if err := operation(name, true); err != nil {
			s.terminal = err
			return err
		}
	}
	return nil
}

// sharedStats executes or validates one stats->merge pair within one generation.
// Both gradient passes and all state-check pairs must be purely validated before
// any optimiser write. Stats recomputation/merge completion precede view expiry.
func (s *ResidentState) sharedStats(ctx context.Context, source ResidentBuffer, n int, positive bool, mergeOut, mergeIn ResidentBuffer, previous int64, mode uint32, execute bool) error {
	w := s.shared
	if err := w.begin(ctx); err != nil {
		return err
	}
	defer w.end()
	ctx = w.ctx
	count, _ := NormStatsShape(n)
	stats, err := w.view(w.stats, count, true)
	if err != nil {
		return err
	}
	if !execute {
		if err = w.driver.ValidateNormStats(ctx, stats, source, n, positive); err != nil {
			return err
		}
		return s.driver.(ResidentGlobalNormDriver).ValidateNormMerge(ctx, mergeOut, mergeIn, stats, n, previous, mode)
	}
	if err = s.driver.NormStats(ctx, stats, source, n, positive); err != nil {
		return err
	}
	return s.driver.(ResidentGlobalNormDriver).NormMerge(ctx, mergeOut, mergeIn, stats, n, previous, mode)
}
