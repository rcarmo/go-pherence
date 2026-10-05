//go:build linux

package pockettts

import (
	"context"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func TestTrainingReleasedDeviceResidentRejectBeforeAllocation(t *testing.T) {
	cfg, flow, pool := releasedDeviceAuditFixture(t)
	inventory := releasedDeviceAuditInventory(t, cfg, flow)
	snapshot := train.ResidentSnapshot{Config: train.ResidentConfig{LearningRate: .001, Beta1: .9, Beta2: .99, Epsilon: 1e-8, EMADecay: .9}, Parameters: map[string]train.ResidentParameter{}}
	for name, n := range inventory {
		if pool[n] == nil {
			pool[n] = make([]float32, n)
		}
		p := pool[n]
		snapshot.Parameters[name] = train.ResidentParameter{Parameter: p, M: p, V: p, EMA: p}
	}
	d := &hostResidentGraphDriver{t: t}
	// Exercise admission with a fake driver only. Full logical inventory exceeds
	// both limits; no construction, upload, model evaluation or native call occurs.
	for _, ceiling := range []int64{256 << 20, 2 << 30} {
		state, err := train.NewStateCheckResidentState(context.Background(), d, snapshot, ceiling)
		if err == nil || state != nil || d.allocations != 0 || d.uploads != 0 || d.tapes != 0 {
			t.Fatalf("oversized released state allocated: limit%d err%v alloc%d upload%d", ceiling, err, d.allocations, d.uploads)
		}
		t.Logf("resident preflight rejects ceiling%d before allocation/upload/tape: %v", ceiling, err)
	}
}
