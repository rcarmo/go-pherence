//go:build linux

package pockettts

import (
	"context"
	"testing"
)

func TestTrainingResidentLoadReuseRunner(t *testing.T) {
	for _, scope := range []string{"ptx-resident-sampleweight-hybrid-v1", "vulkan-resident-sampleweight-hybrid-v1", "ptx-optimizer-hybrid-v1", "vulkan-optimizer-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) {
			var r *TrainingRunner
			var err error
			if scope == "ptx-optimizer-hybrid-v1" || scope == "vulkan-optimizer-hybrid-v1" {
				r, _, _ = optimizerRunner(t, scope)
			} else {
				r, _, _ = residentGraphRunner(t, scope)
			}
			if _, err = r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			refs := make(map[string]NamedAdamState, len(r.trainer.Moments))
			for n, x := range r.trainer.Moments {
				refs[n] = x
			}
			ema := make(map[string][]float32, len(r.trainer.EMA))
			for n, x := range r.trainer.EMA {
				ema[n] = x
			}
			if _, err = r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			for n, x := range refs {
				y := r.trainer.Moments[n]
				if &x.M[0] != &y.M[0] || &x.V[0] != &y.V[0] {
					t.Fatal("runner moment replacement", n)
				}
			}
			for n, x := range ema {
				if &x[0] != &r.trainer.EMA[n][0] {
					t.Fatal("runner EMA replacement", n)
				}
			}
			if err = r.trainer.OptimizerBackend.ValidateBoundary(r.trainer); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			if _, err = r.Checkpoint(dir, "reuse", TrainingBundleLimits{1 << 20, 1 << 16}); err != nil {
				t.Fatal(err)
			}
		})
	}
}
