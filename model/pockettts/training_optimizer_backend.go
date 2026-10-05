package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// TrainingOptimizerBackend updates a full trainer and synchronises coherent
// host state before returning success. Caller commits replay only afterwards.
// Failure can follow native mutation and must poison the runner. No fallback.
type TrainingOptimizerBackend interface {
	TrainingOptimizerBackendName() string
	UpdateTrainer(context.Context, *FullTrainer, TrainingStepGradients, float32) error
	ValidateBoundary(*FullTrainer) error
	Close() error
}

// ResidentOptimizerBackend is an explicit optimiser-only hybrid. CPU computes
// gradients; device owns accumulation/clipping/AdamW/EMA; a complete validated
// snapshot returns to host each update for subsequent CPU training/checkpoints.
// Small global norm summaries reduce on CPU. Driver/runtime is caller-owned.
// Initialise lazily after runner restore so moments/EMA resume coherently.
type ResidentOptimizerBackend struct {
	Driver   train.ResidentDriver
	Scope    string
	MaxBytes int64
	state    *train.ResidentState
	host     FullTrainingState
	terminal error
	closed   bool
}

func (b *ResidentOptimizerBackend) TrainingOptimizerBackendName() string {
	if b == nil {
		return "unknown"
	}
	return b.Scope
}
func (b *ResidentOptimizerBackend) UpdateTrainer(ctx context.Context, t *FullTrainer, g TrainingStepGradients, maxNorm float32) error {
	if b == nil || b.Driver == nil || ctx == nil || t == nil || b.closed || b.terminal != nil {
		return fmt.Errorf("nil/terminal resident optimiser backend")
	}
	if b.Scope != "ptx-optimizer-hybrid-v1" && b.Scope != "vulkan-optimizer-hybrid-v1" {
		return fmt.Errorf("unsupported resident optimiser scope")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !isFinite(maxNorm) || maxNorm <= 0 {
		return fmt.Errorf("invalid resident gradient ceiling")
	}
	if !flowLMGradientMatches(g.FlowLM, t.FlowLM) || g.Flow == nil || len(g.Flow.Time) != len(t.Flow.Time) || len(g.Flow.Blocks) != len(t.Flow.Blocks) || g.Weighting == nil || len(g.Weighting.Layers) != len(t.Weighting.Layers) {
		return fmt.Errorf("resident optimiser gradient topology")
	}
	if b.state == nil {
		host, err := t.State()
		if err != nil {
			return err
		}
		snapshot, err := ResidentSnapshotFromFullState(host)
		if err != nil {
			return err
		}
		b.state, err = train.NewResidentState(ctx, b.Driver, snapshot, b.MaxBytes)
		if err != nil {
			if b.state != nil {
				b.terminal = err
			}
			return err
		}
		b.host = host
	} else if !sameResidentLiveBoundary(b.host, t) {
		return fmt.Errorf("host trainer changed against resident completed boundary")
	}
	gradients := map[string][]float32{}
	for _, binding := range t.bindings {
		values := binding.gradient(g)
		if len(values) != len(binding.parameter) || !finiteF32(values) {
			return fmt.Errorf("resident optimiser gradient %s shape/nonfinite", binding.name)
		}
		gradients[binding.name] = values
	}
	if err := b.state.Accumulate(ctx, gradients, 1); err != nil {
		b.terminal = err
		return err
	}
	if _, err := b.state.Update(ctx, maxNorm); err != nil {
		b.terminal = err
		return err
	}
	snapshot, err := b.state.Snapshot(ctx)
	if err != nil {
		b.terminal = err
		return err
	}
	coherent, err := takeFullStateFromResidentSnapshot(b.host, &snapshot)
	if err != nil {
		b.terminal = err
		return err
	}
	if err = t.loadResidentCompletedState(coherent); err != nil {
		b.terminal = err
		return err
	}
	b.host = coherent
	return nil
}

func (b *ResidentOptimizerBackend) ValidateBoundary(t *FullTrainer) error {
	if b == nil || t == nil || b.closed || b.terminal != nil {
		return fmt.Errorf("nil/closed/terminal resident optimiser boundary")
	}
	if b.state == nil {
		return nil
	} // Fresh pre-restore owner: device state not created.
	if b.state.StepCount() != uint64(t.StepCount) || !sameResidentLiveBoundary(b.host, t) {
		return fmt.Errorf("host/resident checkpoint boundary mismatch")
	}
	return nil
}

// Values here have an exact checkpoint-state contract: host state was copied
// from the completed snapshot, so any caller mutation invalidates residency.
// This does not compare generated audio or establish numerical equivalence.
func sameResidentHostBoundary(a, b FullTrainingState) bool {
	if a.Step != b.Step || a.AdamW != b.AdamW || a.EMADecay != b.EMADecay {
		return false
	}
	same := func(x, y []NamedTrainingTensor) bool {
		if len(x) != len(y) {
			return false
		}
		for i, p := range x {
			q := y[i]
			if p.Name != q.Name || len(p.Values) != len(q.Values) {
				return false
			}
			for j, v := range p.Values {
				if v != q.Values[j] {
					return false
				}
			}
		}
		return true
	}
	if !same(a.Params, b.Params) || !same(a.Buffers, b.Buffers) || !same(a.EMA, b.EMA) || len(a.Adam) != len(b.Adam) {
		return false
	}
	for i, m := range a.Adam {
		n := b.Adam[i]
		if m.Name != n.Name || len(m.M) != len(n.M) || len(m.V) != len(n.V) {
			return false
		}
		for j, v := range m.M {
			if v != n.M[j] || m.V[j] != n.V[j] {
				return false
			}
		}
	}
	return true
}
func (b *ResidentOptimizerBackend) Close() error {
	if b == nil || b.closed {
		return nil
	}
	if b.terminal != nil {
		return fmt.Errorf("resident optimiser resources retained: %w", b.terminal)
	}
	if err := b.state.Close(); err != nil {
		b.terminal = err
		return err
	}
	if owner, ok := b.Driver.(interface{ Close() error }); ok {
		if err := owner.Close(); err != nil {
			b.terminal = err
			return err
		}
	}
	b.closed = true
	return nil
}
