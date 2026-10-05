package training

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
)

const NormChunkElements = 256

// NormStatsShape bounds per-chunk (maximum absolute value, scaled sum of
// squares, invalid flag). Scaling avoids F32 square overflow for finite input.
func NormStatsShape(n int) (int, error) {
	if n < 1 || n > 1<<26 {
		return 0, fmt.Errorf("invalid training norm extent")
	}
	return ((n-1)/NormChunkElements + 1) * 3, nil
}

// ResidentBuffer owns a native allocation. Drivers require disjoint storage and
// synchronous completion. An error may indicate unfinished native work; state
// then retains every owner and permits no retry/snapshot/ordinary Close.
type ResidentBuffer interface {
	Upload(context.Context, []float32) error
	Download(context.Context, []float32) error
	Close() error
}

// ResidentDriver owns live kernels separately from the ResidentState buffers.
// ValidateUpdate must admit all ranges/constants without submitting work.
// Each method has no fallback; the caller explicitly selected this driver.
// Native metadata/allocator overhead is outside the logical byte ceiling.
type ResidentDriver interface {
	NewBuffer(context.Context, int) (ResidentBuffer, error)
	Accumulate(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, float32) error
	NormStats(context.Context, ResidentBuffer, ResidentBuffer, int, bool) error
	ValidateUpdate(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, AdamWStep) error
	Update(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, AdamWStep) error
}
type ResidentConfig struct{ LearningRate, Beta1, Beta2, Epsilon, WeightDecay, EMADecay float32 }
type ResidentParameter struct{ Parameter, M, V, EMA []float32 }
type ResidentSnapshot struct {
	Step       uint64
	Config     ResidentConfig
	Parameters map[string]ResidentParameter
}
type residentEntry struct {
	name                                                         string
	n                                                            int
	parameter, gradient, scratch, branch, zero, m, v, ema, stats ResidentBuffer
	ready                                                        bool
}

// ResidentState is single-owner, non-concurrent training state. Gradients are
// accumulated explicitly by name. Update requires every parameter touched;
// missing branches are never silently zeroed. Snapshot only succeeds at a
// completed clean boundary and returns owned host arrays. Restore constructs a
// fresh state from a fully validated snapshot; in-place restore is unsupported.
// Legacy states reduce norm summaries on CPU in F64. Explicit global-norm
// states compute the maximum, compensated sum and clipping scale on device;
// four diagnostic/scale scalars and post-update checks still transfer to host.
// Device-scale states pass the validated result buffer directly to AdamW;
// global-norm-only states still pass its scale through host launch constants.
// AdamW bias corrections and snapshot orchestration remain host work.
type ResidentState struct {
	driver                       ResidentDriver
	config                       ResidentConfig
	step                         uint64
	entries                      []residentEntry
	index                        map[string]int
	owned                        []ResidentBuffer
	terminal                     error
	closed, dirty                bool
	borrowed                     bool
	retained                     []ResidentBoundTape
	lease                        *ResidentTapeLease
	deviceGlobalNorm             bool
	deviceScale                  bool
	deviceStateCheck             bool
	globalA, globalB, globalClip ResidentBuffer
	shared                       *residentSharedWorkspace
}

func ResidentBytes(snapshot ResidentSnapshot) (int64, error) {
	if len(snapshot.Parameters) == 0 || len(snapshot.Parameters) > 4096 {
		return 0, fmt.Errorf("resident parameter count")
	}
	if _, err := NewAdamWStep(snapshot.Config.LearningRate, snapshot.Config.Beta1, snapshot.Config.Beta2, snapshot.Config.Epsilon, snapshot.Config.WeightDecay, snapshot.Config.EMADecay, 1, 1); err != nil {
		return 0, err
	}
	if snapshot.Step > uint64(math.MaxInt) {
		return 0, fmt.Errorf("resident completed step ceiling")
	}
	var bytes int64
	for name, p := range snapshot.Parameters {
		if len(name) == 0 || len(name) > 512 {
			return 0, fmt.Errorf("resident parameter name")
		}
		n := len(p.Parameter)
		stats, err := NormStatsShape(n)
		if err != nil {
			return 0, err
		}
		for _, x := range [][]float32{p.Parameter, p.M, p.V, p.EMA} {
			if len(x) != n {
				return 0, fmt.Errorf("resident %s state shape", name)
			}
			for _, v := range x {
				if !residentFinite(v) {
					return 0, fmt.Errorf("resident %s nonfinite state", name)
				}
			}
		}
		for _, v := range p.V {
			if v < 0 {
				return 0, fmt.Errorf("resident %s negative second moment", name)
			}
		}
		bytes += int64(n)*32 + int64(stats)*4
	}
	return bytes, nil
}

// NewResidentState never initialises a runtime. A nonnil owner accompanying an
// error must be retained, including failures during upload or rollback cleanup.
func NewResidentState(ctx context.Context, driver ResidentDriver, snapshot ResidentSnapshot, maxBytes int64) (*ResidentState, error) {
	return newResidentState(ctx, driver, snapshot, maxBytes, false, false, false)
}

// NewGlobalNormResidentState selects device summary merging/clipping. The
// legacy constructor retains host F64 reduction. Twelve extra F32 scalars owned.
func NewGlobalNormResidentState(ctx context.Context, driver ResidentDriver, snapshot ResidentSnapshot, maxBytes int64) (*ResidentState, error) {
	return newResidentState(ctx, driver, snapshot, maxBytes, true, false, false)
}
func newResidentState(ctx context.Context, driver ResidentDriver, snapshot ResidentSnapshot, maxBytes int64, deviceGlobalNorm, deviceScale, deviceStateCheck bool) (owner *ResidentState, err error) {
	if ctx == nil || driver == nil {
		return nil, fmt.Errorf("explicit resident context/driver required")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if deviceScale {
		if _, ok := driver.(ResidentDeviceScaleDriver); !ok {
			return nil, fmt.Errorf("resident driver lacks device clipping-scale update")
		}
	}
	bytes, err := ResidentBytes(snapshot)
	if err != nil {
		return nil, err
	}
	if deviceGlobalNorm {
		if _, ok := driver.(ResidentGlobalNormDriver); !ok {
			return nil, fmt.Errorf("resident driver lacks device global norm")
		}
		var elements int64
		for _, p := range snapshot.Parameters {
			elements += int64(len(p.Parameter))
		}
		if deviceStateCheck && elements > (1<<38)/4 {
			return nil, fmt.Errorf("resident mutable-state element ceiling")
		}
		if err := NormClipShape(elements, 1); err != nil {
			return nil, err
		}
		bytes += 48
	}
	if maxBytes <= 0 || bytes > maxBytes {
		return nil, fmt.Errorf("resident byte ceiling")
	}
	owner = &ResidentState{driver: driver, config: snapshot.Config, step: snapshot.Step, index: map[string]int{}, deviceGlobalNorm: deviceGlobalNorm, deviceScale: deviceScale, deviceStateCheck: deviceStateCheck}
	names := make([]string, 0, len(snapshot.Parameters))
	for name := range snapshot.Parameters {
		names = append(names, name)
	}
	sort.Strings(names)
	// Allocation errors are conservatively terminal: a partial native allocation
	// may have uncertain cleanup. Successfully created owners stay reachable.
	for _, name := range names {
		p := snapshot.Parameters[name]
		entry := residentEntry{name: name, n: len(p.Parameter)}
		stats, _ := NormStatsShape(entry.n)
		for _, slot := range []struct {
			dst    *ResidentBuffer
			n      int
			values []float32
		}{{&entry.parameter, entry.n, p.Parameter}, {&entry.gradient, entry.n, nil}, {&entry.scratch, entry.n, nil}, {&entry.branch, entry.n, nil}, {&entry.zero, entry.n, make([]float32, entry.n)}, {&entry.m, entry.n, p.M}, {&entry.v, entry.n, p.V}, {&entry.ema, entry.n, p.EMA}, {&entry.stats, stats, nil}} {
			b, e := driver.NewBuffer(ctx, slot.n)
			if b != nil {
				owner.owned = append(owner.owned, b)
				*slot.dst = b
			}
			if e != nil {
				owner.terminal = e
				return owner, e
			}
			if b == nil {
				err = fmt.Errorf("resident driver returned nil buffer")
				owner.terminal = err
				return owner, err
			}
			if slot.values != nil {
				if e = b.Upload(ctx, slot.values); e != nil {
					owner.terminal = e
					return owner, e
				}
			}
		}
		owner.index[name] = len(owner.entries)
		owner.entries = append(owner.entries, entry)
	}
	if deviceGlobalNorm {
		for _, slot := range []struct {
			dst *ResidentBuffer
			n   int
		}{{&owner.globalA, 4}, {&owner.globalB, 4}, {&owner.globalClip, 4}} {
			b, e := driver.NewBuffer(ctx, slot.n)
			if b != nil {
				owner.owned = append(owner.owned, b)
				*slot.dst = b
			}
			if e != nil || b == nil {
				if e == nil {
					e = fmt.Errorf("nil global norm buffer")
				}
				owner.terminal = e
				return owner, e
			}
		}
	}
	return owner, nil
}
func (s *ResidentState) admit(ctx context.Context) error {
	if s == nil || ctx == nil {
		return fmt.Errorf("nil resident state/context")
	}
	if s.terminal != nil {
		return fmt.Errorf("terminal resident state: %w", s.terminal)
	}
	if s.closed {
		return fmt.Errorf("closed resident state")
	}
	if s.borrowed || s.lease != nil {
		return fmt.Errorf("resident state has active tape borrow/lease")
	}
	if s.shared != nil && s.shared.active {
		return fmt.Errorf("resident shared workspace active")
	}
	return ctx.Err()
}
func (s *ResidentState) StepCount() uint64 {
	if s == nil {
		return 0
	}
	return s.step
}

// Accumulate validates every supplied branch before any upload. Branch values
// and scale must be finite; scaled sums must remain representable. Numerical
// overflow is rejected by NormStats before any optimiser update.
func (s *ResidentState) Accumulate(ctx context.Context, branches map[string][]float32, scale float32) error {
	if err := s.admit(ctx); err != nil {
		return err
	}
	if len(branches) == 0 || !residentFinite(scale) {
		return fmt.Errorf("invalid resident branch/scale")
	}
	if s.shared != nil && s.shared.tapeOnly {
		return fmt.Errorf("tape-only shared resident rejects host gradients")
	}
	for name, x := range branches {
		index, ok := s.index[name]
		if !ok || len(x) != s.entries[index].n {
			return fmt.Errorf("resident branch %s shape/name", name)
		}
		for _, v := range x {
			if !residentFinite(v) {
				return fmt.Errorf("resident branch %s nonfinite", name)
			}
		}
	}
	if s.shared != nil {
		s.borrowed = true
		defer func() { s.borrowed = false }()
		names := make([]string, 0, len(branches))
		for _, e := range s.entries {
			if _, ok := branches[e.name]; ok {
				names = append(names, e.name)
			}
		}
		return s.sharedAccumulate(ctx, names, nil, nil, nil, branches, scale)
	}
	// Deterministic parameter order; no borrowed branch storage survives upload.
	for i := range s.entries {
		e := &s.entries[i]
		x, ok := branches[e.name]
		if !ok {
			continue
		}
		if err := e.branch.Upload(ctx, x); err != nil {
			s.terminal = err
			return err
		}
		previous := e.zero
		if e.ready {
			previous = e.gradient
		}
		if err := s.driver.Accumulate(ctx, e.scratch, previous, e.branch, e.n, scale); err != nil {
			s.terminal = err
			return err
		}
		e.gradient, e.scratch = e.scratch, e.gradient
		e.ready = true
		s.dirty = true
	}
	return nil
}

// Update downloads per-parameter norm summaries in legacy states, or four
// device-global norm/scale scalars in explicit global-norm states, before AdamW.
// Complete-pass range validation precedes mutation; uncertain errors retain owners.
// The supplied ceiling clips the global norm across all named parameters.
func (s *ResidentState) Update(ctx context.Context, maxNorm float32) (norm float64, err error) {
	if err = s.admit(ctx); err != nil {
		return 0, err
	}
	if s.shared != nil {
		s.borrowed = true
		defer func() { s.borrowed = false }()
	}
	if !residentFinite(maxNorm) || maxNorm <= 0 {
		return 0, fmt.Errorf("invalid resident norm ceiling")
	}
	if s.step >= uint64(math.MaxInt) {
		return 0, fmt.Errorf("resident completed step ceiling")
	}
	for _, e := range s.entries {
		if !e.ready {
			return 0, fmt.Errorf("resident gradient %s missing", e.name)
		}
	}
	if s.deviceStateCheck {
		if err = s.validateDeviceStateCheck(ctx); err != nil {
			return 0, err
		}
	}
	scale := float32(1)
	// Preflight every new device-scale update before reduction or mutation.
	// The scale buffer is not read during validation; its contents are checked
	// by the diagnostic boundary after reduction, before the first update.
	c := s.config
	if s.deviceScale {
		p, e := NewAdamWStep(c.LearningRate, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay, c.EMADecay, 1, s.step+1)
		if e != nil {
			return 0, e
		}
		for _, entry := range s.entries {
			if e = s.driver.(ResidentDeviceScaleDriver).ValidateScaledUpdate(ctx, entry.parameter, entry.gradient, entry.m, entry.v, entry.ema, s.globalClip, entry.n, p); e != nil {
				return 0, e
			}
		}
	}
	if s.deviceGlobalNorm {
		norm, scale, err = s.reduceGlobalNorm(ctx, maxNorm)
		if err != nil {
			return 0, err
		}
	} else {
		var square, correction float64
		for _, e := range s.entries {
			if err = s.driver.NormStats(ctx, e.stats, e.gradient, e.n, false); err != nil {
				s.terminal = err
				return 0, err
			}
			count, _ := NormStatsShape(e.n)
			values := make([]float32, count)
			if err = e.stats.Download(ctx, values); err != nil {
				s.terminal = err
				return 0, err
			}
			part, eerr := NormSquareFromStats(values, e.n)
			if eerr != nil {
				s.terminal = eerr
				return 0, eerr
			}
			term := part - correction
			next := square + term
			correction = (next - square) - term
			square = next
		}
		norm = math.Sqrt(square)
		if math.IsNaN(norm) || math.IsInf(norm, 0) {
			err = fmt.Errorf("resident norm overflow")
			s.terminal = err
			return 0, err
		}
		if norm > float64(maxNorm) {
			scale = float32(float64(maxNorm) / norm)
		}
	}
	if s.deviceScale {
		scale = 1
	} // Never hand the downloaded scale back to the kernel.
	p, err := NewAdamWStep(c.LearningRate, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay, c.EMADecay, scale, s.step+1)
	if err != nil {
		return 0, err
	}
	if !s.deviceScale {
		for _, e := range s.entries {
			if err = s.driver.ValidateUpdate(ctx, e.parameter, e.gradient, e.m, e.v, e.ema, e.n, p); err != nil {
				return 0, err
			}
		}
	}
	for _, e := range s.entries {
		if s.deviceScale {
			err = s.driver.(ResidentDeviceScaleDriver).UpdateScaled(ctx, e.parameter, e.gradient, e.m, e.v, e.ema, s.globalClip, e.n, p)
		} else {
			err = s.driver.Update(ctx, e.parameter, e.gradient, e.m, e.v, e.ema, e.n, p)
		}
		if err != nil {
			s.terminal = err
			return 0, err
		}
	}
	// Verify finite mutable state and nonnegative V before committing a step.
	// Only small summaries transfer. Failure after mutation is terminal.
	if s.deviceStateCheck {
		if err = s.checkDeviceState(ctx); err != nil {
			s.terminal = err
			return 0, err
		}
	} else {
		for _, e := range s.entries {
			for _, slot := range []struct {
				b        ResidentBuffer
				positive bool
			}{{e.parameter, false}, {e.m, false}, {e.v, true}, {e.ema, false}} {
				if err = s.driver.NormStats(ctx, e.stats, slot.b, e.n, slot.positive); err != nil {
					s.terminal = err
					return 0, err
				}
				count, _ := NormStatsShape(e.n)
				values := make([]float32, count)
				if err = e.stats.Download(ctx, values); err != nil {
					s.terminal = err
					return 0, err
				}
				if _, err = NormSquareFromStats(values, e.n); err != nil {
					s.terminal = err
					return 0, err
				}
			}
		}
	}
	s.step++
	for i := range s.entries {
		s.entries[i].ready = false
	}
	s.dirty = false
	return norm, nil
}

// NormSquareFromStats validates chunk summaries and reduces in compensated F64.
// Device chunks contain <=256 values, so scaled F32 sums cannot square-overflow.
func NormSquareFromStats(values []float32, n int) (float64, error) {
	count, err := NormStatsShape(n)
	if err != nil || len(values) != count {
		return 0, fmt.Errorf("resident norm summary shape")
	}
	var sum, correction float64
	for i := 0; i < count; i += 3 {
		maximum, scaled, flag := values[i], values[i+1], values[i+2]
		chunk := min(NormChunkElements, n-(i/3)*NormChunkElements)
		if !residentFinite(maximum) || !residentFinite(scaled) || !residentFinite(flag) || flag != 0 || maximum < 0 || scaled < 0 || scaled > float32(chunk) || (maximum == 0 && scaled != 0) || (maximum > 0 && scaled < 1) {
			return 0, fmt.Errorf("invalid/nonfinite resident norm summary")
		}
		term := float64(maximum)*float64(maximum)*float64(scaled) - correction
		next := sum + term
		correction = (next - sum) - term
		sum = next
	}
	return sum, nil
}
func (s *ResidentState) Snapshot(ctx context.Context) (ResidentSnapshot, error) {
	var snapshot ResidentSnapshot
	if err := s.admit(ctx); err != nil {
		return snapshot, err
	}
	if s.dirty {
		return snapshot, fmt.Errorf("resident snapshot requires completed boundary")
	}
	if s.shared != nil {
		s.borrowed = true
		defer func() { s.borrowed = false }()
	}
	snapshot = ResidentSnapshot{Step: s.step, Config: s.config, Parameters: map[string]ResidentParameter{}}
	for _, e := range s.entries {
		p := ResidentParameter{Parameter: make([]float32, e.n), M: make([]float32, e.n), V: make([]float32, e.n), EMA: make([]float32, e.n)}
		for _, slot := range []struct {
			b ResidentBuffer
			x []float32
		}{{e.parameter, p.Parameter}, {e.m, p.M}, {e.v, p.V}, {e.ema, p.EMA}} {
			if err := slot.b.Download(ctx, slot.x); err != nil {
				s.terminal = err
				return ResidentSnapshot{}, err
			}
		}
		snapshot.Parameters[e.name] = p
	}
	if _, err := ResidentBytes(snapshot); err != nil {
		s.terminal = err
		return ResidentSnapshot{}, err
	}
	return snapshot, nil
}
func (s *ResidentState) Close() error {
	if s == nil || s.closed {
		return nil
	}
	if s.borrowed || s.lease != nil || (s.shared != nil && s.shared.active) {
		return fmt.Errorf("resident state has active tape borrow/lease/workspace")
	}
	if s.terminal != nil {
		return fmt.Errorf("resident uncertain resources retained: %w", s.terminal)
	}
	if s.shared != nil {
		s.borrowed = true
		defer func() { s.borrowed = false }()
	}
	var err error
	for _, b := range s.owned {
		err = errors.Join(err, b.Close())
	}
	if err != nil {
		s.terminal = err
		return err
	}
	s.closed = true
	return nil
}
func residentFinite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }
