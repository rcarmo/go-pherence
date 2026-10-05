package pockettts

import (
	"context"
	"fmt"
	"unsafe"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// A separate replay identity selects reduced storage; legacy scopes keep their
// constructors and memory policy. Only Vulkan has a native adapter so far.
func (b *ResidentTrainingBackend) sharedWorkspaceEnabled() bool {
	return b != nil && b.Scope == "vulkan-resident-sharedworkspace-hybrid-v1"
}

// sharedCheckpointState borrows the completed backend-owned host snapshot only
// for synchronous runner serialisation. Caller serialises all runner operations;
// no update/restore/close or mutation may interleave. It is never returned from
// a public State/snapshot API. Boundary validation checks live-state separation
// and current values before handing the immutable sources to the binary writer.
func (b *ResidentTrainingBackend) sharedCheckpointState(t *FullTrainer) (FullTrainingState, error) {
	if !b.sharedWorkspaceEnabled() || (b.state == nil && b.host.Params == nil) {
		return FullTrainingState{}, fmt.Errorf("shared checkpoint requires completed owned state")
	}
	if err := b.ValidateBoundary(t); err != nil {
		return FullTrainingState{}, err
	}
	if err := b.host.Validate(); err != nil {
		return FullTrainingState{}, err
	}
	if err := validateSharedSnapshotStorage(b.host, t); err != nil {
		return FullTrainingState{}, err
	}
	return b.host, nil
}

// restoreSharedOwnedState consumes only the fresh independently decoded runner
// checkpoint after association/step/config checks. No native owner exists yet.
// Complete numerical/topology/alias validation precedes live mutation. The
// backend retains the decoded arrays as its separate host mutation guard.
func (b *ResidentTrainingBackend) restoreSharedOwnedState(t *FullTrainer, source *FullTrainingState) error {
	if !b.sharedWorkspaceEnabled() || t == nil || source == nil || source.Step <= 0 || b.state != nil || b.host.Params != nil || b.trainer != nil || b.active || b.closed || b.terminal != nil {
		return fmt.Errorf("shared restore owner/source unavailable")
	}
	if err := t.loadSharedRestoredState(*source); err != nil {
		return err
	}
	b.host, b.trainer = *source, t
	*source = FullTrainingState{} // ownership transfers only after success
	return nil
}

// Retain only freshly allocated step-zero moments after synchronous initial
// uploads. Keep b.host.Adam empty until a successful completed update, so the
// exact host/live mutation guard and initial checkpoint contract stay unchanged.
func (b *ResidentTrainingBackend) rememberSharedInitialMoments(snapshot train.ResidentSnapshot) {
	if !b.sharedWorkspaceEnabled() || b.host.Step != 0 || len(b.host.Adam) != 0 {
		return
	}
	b.sharedInitialMoments = make([]NamedAdamState, 0, len(b.host.Params))
	for _, p := range b.host.Params {
		v := snapshot.Parameters[p.Name]
		b.sharedInitialMoments = append(b.sharedInitialMoments, NamedAdamState{Name: p.Name, M: v.M, V: v.V})
	}
}

// loadSharedCompletedSnapshot reuses only the private independent host boundary.
// Download may destroy its previous values; any later failure is terminal and
// cannot publish/retry that boundary. Live trainer arrays remain unchanged until
// complete download/numerical/alias admission succeeds in the load helper.
func (b *ResidentTrainingBackend) loadSharedCompletedSnapshot(ctx context.Context, t *FullTrainer) error {
	if b == nil || b.terminal != nil || b.closed || !b.sharedWorkspaceEnabled() || b.state == nil || b.trainer != t {
		return fmt.Errorf("shared completed snapshot owner unavailable")
	}
	destination := b.host
	if len(destination.Adam) == 0 && len(b.sharedInitialMoments) > 0 {
		destination.Adam = b.sharedInitialMoments
	}
	if err := validateSharedSnapshotStorage(destination, t); err != nil {
		return b.fail(err)
	}
	snapshot, err := sharedInitialSnapshot(destination)
	if err != nil {
		return b.fail(err)
	}
	if err := b.state.SnapshotInto(ctx, &snapshot); err != nil {
		return b.fail(err)
	}
	next := destination
	next.Step = int(snapshot.Step)
	// State() omits missing step-zero moments. Adopt the first independent
	// download storage; later boundaries reuse the same arrays.
	if len(next.Adam) == 0 {
		next.Adam = make([]NamedAdamState, 0, len(next.Params))
		for _, p := range next.Params {
			v := snapshot.Parameters[p.Name]
			next.Adam = append(next.Adam, NamedAdamState{Name: p.Name, M: v.M, V: v.V})
		}
	}
	if err := t.loadResidentCompletedState(next); err != nil {
		return b.fail(err)
	}
	b.host = next
	b.sharedInitialMoments = nil // adopted by completed host boundary
	return nil
}

// Check source/live separation before overwriting the host boundary. The later
// completed-state loader validates numerical data and topology before copying.
func validateSharedSnapshotStorage(host FullTrainingState, t *FullTrainer) error {
	if t == nil || t.refreshParameterBindings() != nil {
		return fmt.Errorf("shared snapshot live trainer unavailable")
	}
	var ranges []residentLoadRange
	add := func(x []float32) error {
		if len(x) == 0 {
			return fmt.Errorf("shared snapshot empty storage")
		}
		start := uintptr(unsafe.Pointer(&x[0]))
		bytes := uintptr(len(x)) * 4
		if bytes/4 != uintptr(len(x)) || start > ^uintptr(0)-bytes {
			return fmt.Errorf("shared snapshot storage overflow")
		}
		ranges = append(ranges, residentLoadRange{start, start + bytes})
		return nil
	}
	for _, list := range [][]NamedTrainingTensor{host.Params, host.Buffers, host.EMA} {
		for _, p := range list {
			if err := add(p.Values); err != nil {
				return err
			}
		}
	}
	for _, m := range host.Adam {
		if err := add(m.M); err != nil {
			return err
		}
		if err := add(m.V); err != nil {
			return err
		}
	}
	for _, values := range []map[string][]float32{t.params, fullBufferMap(t.FlowLM, t.Flow)} {
		for _, x := range values {
			if err := add(x); err != nil {
				return err
			}
		}
	}
	for _, m := range t.Moments {
		if err := add(m.M); err != nil {
			return err
		}
		if err := add(m.V); err != nil {
			return err
		}
	}
	if t.EMADecay > 0 {
		for _, x := range t.EMA {
			if err := add(x); err != nil {
				return err
			}
		}
	}
	return validateResidentLoadRanges(ranges)
}
