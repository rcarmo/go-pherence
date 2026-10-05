package training

import (
	"context"
	"fmt"
	"sort"
	"unsafe"
)

// SnapshotInto overwrites independent caller-owned host storage at a completed
// shared-workspace boundary. It never allocates parameter-sized arrays or
// borrows native storage. Public Snapshot still returns fresh owned arrays.
// The destination must be private, pairwise disjoint and relinquished for this
// synchronous call. A download, cancellation or validation error after writes
// makes the state terminal and the destination unusable: no rollback or retry.
// Admission errors leave the destination unchanged and perform no transfers.
func (s *ResidentState) SnapshotInto(ctx context.Context, dst *ResidentSnapshot) error {
	if err := s.admit(ctx); err != nil {
		return err
	}
	if s.shared == nil || s.dirty || dst == nil {
		return fmt.Errorf("snapshot into requires completed shared state/destination")
	}
	if dst.Config != s.config || dst.Step > s.step || len(dst.Parameters) != len(s.entries) {
		return fmt.Errorf("shared snapshot destination config/step/inventory")
	}
	// Validate the previous host boundary and all physical host ranges before
	// any download. Driver-owned mapped arrays must never be supplied here.
	if _, err := ResidentBytes(*dst); err != nil {
		return err
	}
	type span struct{ start, end uintptr }
	ranges := make([]span, 0, 4*len(s.entries))
	for _, e := range s.entries {
		p, ok := dst.Parameters[e.name]
		if !ok {
			return fmt.Errorf("shared snapshot destination name")
		}
		for _, x := range [][]float32{p.Parameter, p.M, p.V, p.EMA} {
			if len(x) != e.n {
				return fmt.Errorf("shared snapshot destination extent")
			}
			start := uintptr(unsafe.Pointer(&x[0]))
			bytes := uintptr(len(x)) * 4
			if start == 0 || bytes/4 != uintptr(len(x)) || start > ^uintptr(0)-bytes {
				return fmt.Errorf("shared snapshot destination address overflow")
			}
			ranges = append(ranges, span{start, start + bytes})
		}
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].start < ranges[j].start })
	for i := 1; i < len(ranges); i++ {
		if ranges[i-1].end > ranges[i].start {
			return fmt.Errorf("shared snapshot destination storage aliases")
		}
	}
	s.borrowed = true
	defer func() { s.borrowed = false }()
	for _, e := range s.entries {
		p := dst.Parameters[e.name]
		for _, slot := range []struct {
			b ResidentBuffer
			x []float32
		}{{e.parameter, p.Parameter}, {e.m, p.M}, {e.v, p.V}, {e.ema, p.EMA}} {
			if err := slot.b.Download(ctx, slot.x); err != nil {
				s.terminal = err
				return err
			}
			if err := ctx.Err(); err != nil {
				s.terminal = err
				return err
			}
		}
	}
	if _, err := ResidentBytes(*dst); err != nil {
		s.terminal = err
		return err
	}
	if err := ctx.Err(); err != nil {
		s.terminal = err
		return err
	}
	dst.Step = s.step
	return nil
}
