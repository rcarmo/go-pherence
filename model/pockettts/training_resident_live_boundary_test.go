//go:build linux

package pockettts

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func TestTrainingResidentLiveBoundaryMutation(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*FullTrainer)
	}{
		{"parameter", func(t *FullTrainer) { t.params[t.names[0]][0] += .1 }},
		{"parameter-nan", func(t *FullTrainer) { t.params[t.names[0]][0] = float32(math.NaN()) }},
		{"buffer", func(t *FullTrainer) {
			for _, x := range fullBufferMap(t.FlowLM, t.Flow) {
				x[0] += .1
				break
			}
		}},
		{"buffer-nan", func(t *FullTrainer) {
			for _, x := range fullBufferMap(t.FlowLM, t.Flow) {
				x[0] = float32(math.NaN())
				break
			}
		}},
		{"M", func(t *FullTrainer) { x := t.Moments[t.names[0]]; x.M[0] += .1 }},
		{"V", func(t *FullTrainer) { x := t.Moments[t.names[0]]; x.V[0] += .1 }},
		{"negative-V", func(t *FullTrainer) { x := t.Moments[t.names[0]]; x.V[0] = -1 }},
		{"moment-name", func(t *FullTrainer) {
			x := t.Moments[t.names[0]]
			delete(t.Moments, t.names[0])
			t.Moments["foreign"] = x
		}},
		{"moment-shape", func(t *FullTrainer) { x := t.Moments[t.names[0]]; x.M = x.M[:len(x.M)-1]; t.Moments[t.names[0]] = x }},
		{"EMA", func(t *FullTrainer) { t.EMA[t.names[0]][0] += .1 }},
		{"EMA-name", func(t *FullTrainer) { x := t.EMA[t.names[0]]; delete(t.EMA, t.names[0]); t.EMA["foreign"] = x }},
		{"config", func(t *FullTrainer) { t.Config.Epsilon *= 2 }},
		{"decay", func(t *FullTrainer) { t.EMADecay = .5 }},
		{"step", func(t *FullTrainer) { t.StepCount++ }},
		{"topology", func(t *FullTrainer) { t.Weighting.Layers = append(t.Weighting.Layers, t.Weighting.Layers[0]) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-sampleweight-hybrid-v1")
			if _, err := r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			if !sameResidentLiveBoundary(b.host, r.trainer) {
				t.Fatal("completed boundary rejected")
			}
			owned := b.host
			old, err := r.trainer.State()
			if err != nil {
				t.Fatal(err)
			}
			if !sameResidentHostBoundary(owned, old) {
				t.Fatal("reference mismatch")
			}
			c.mutate(r.trainer)
			got := sameResidentLiveBoundary(owned, r.trainer)
			state, err := r.trainer.State()
			want := err == nil && sameResidentHostBoundary(owned, state)
			if got != want || got {
				t.Fatal("live/reference mutation acceptance", got, want, err)
			}
			events := d.updates
			if b.ValidateBoundary(r.trainer) == nil || b.PrepareTrainingStep(context.Background(), r.trainer) == nil || d.updates != events {
				t.Fatal("mutated boundary launched")
			}
		})
	}
}
func TestTrainingResidentLiveBoundaryOptionalState(t *testing.T) {
	for _, ema := range []float32{0, .9} {
		t.Run(map[bool]string{true: "EMA", false: "noEMA"}[ema > 0], func(t *testing.T) {
			oracle := loadTrainingStepOracle(t)
			lm, flow, w := trainingStepModelsFromOracle(t, oracle)
			trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), ema)
			if err != nil {
				t.Fatal(err)
			}
			host, err := trainer.State()
			if err != nil {
				t.Fatal(err)
			}
			if !sameResidentLiveBoundary(host, trainer) {
				t.Fatal("step0 empty moments")
			}
			// The checkpoint contract uses map keys, not the redundant moment Name field;
			// preserve that behavior exactly at completed boundaries.
			snap, err := ResidentSnapshotFromFullState(host)
			if err != nil {
				t.Fatal(err)
			}
			coherent, err := FullStateFromResidentSnapshot(host, snap)
			if err != nil {
				t.Fatal(err)
			}
			if err = trainer.LoadState(coherent); err != nil {
				t.Fatal(err)
			}
			for name, x := range trainer.Moments {
				x.Name = "ignored"
				trainer.Moments[name] = x
			}
			if !sameResidentLiveBoundary(coherent, trainer) {
				t.Fatal("redundant name behavior changed")
			}
			if ema == 0 {
				trainer.EMA = map[string][]float32{"ignored": {float32(math.NaN())}}
				if !sameResidentLiveBoundary(coherent, trainer) {
					t.Fatal("disabled EMA behavior changed")
				}
			}
			before := append([]float32(nil), coherent.Params[0].Values...)
			trainer.params[coherent.Params[0].Name][0] += .1
			if !reflect.DeepEqual(before, coherent.Params[0].Values) {
				t.Fatal("boundary borrowed live array")
			}
		})
	}
}
func TestTrainingResidentLiveBoundaryAllocation(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-sampleweight-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	borrowed := testing.AllocsPerRun(10, func() {
		if !sameResidentLiveBoundary(b.host, r.trainer) {
			panic("boundary")
		}
	})
	prior := testing.AllocsPerRun(10, func() {
		x, err := r.trainer.State()
		if err != nil || !sameResidentHostBoundary(b.host, x) {
			panic("boundary")
		}
	})
	if borrowed >= prior {
		t.Fatal("no allocation reduction", borrowed, prior)
	}
	t.Logf("completed boundary allocations/op owned State reference=%g -> live=%g; native downloads unchanged", prior, borrowed)
}
