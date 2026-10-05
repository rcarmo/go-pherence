package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestTrainingResidentLoadReuseParity(t *testing.T) {
	for _, ema := range []float32{0, .9} {
		t.Run(map[bool]string{true: "EMA", false: "noEMA"}[ema > 0], func(t *testing.T) {
			trainer, template, snapshot := snapshotHandoffFixture(t, ema)
			state, err := FullStateFromResidentSnapshot(template, snapshot)
			if err != nil {
				t.Fatal(err)
			}
			// Compare with public restore using a separate topology and owned arrays.
			reference, _, _ := snapshotHandoffFixture(t, ema)
			if err = reference.LoadState(state); err != nil {
				t.Fatal(err)
			}
			if err = trainer.loadResidentCompletedState(state); err != nil {
				t.Fatal(err)
			}
			got, err := trainer.State()
			if err != nil {
				t.Fatal(err)
			}
			want, err := reference.State()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatal("public load parity")
			}
			moments, shadows := map[string]NamedAdamState{}, map[string][]float32{}
			for n, x := range trainer.Moments {
				moments[n] = x
			}
			for n, x := range trainer.EMA {
				shadows[n] = x
			}
			for _, p := range state.Params {
				if f32SlicesOverlap(p.Values, trainer.params[p.Name]) {
					t.Fatal("parameter source aliases")
				}
			}
			for _, p := range state.Adam {
				if f32SlicesOverlap(p.M, trainer.Moments[p.Name].M) || f32SlicesOverlap(p.V, trainer.Moments[p.Name].V) {
					t.Fatal("moment source aliases")
				}
			}
			for _, p := range state.EMA {
				if f32SlicesOverlap(p.Values, trainer.EMA[p.Name]) {
					t.Fatal("EMA source aliases")
				}
			}
			for step := 2; step <= 4; step++ {
				state.Step = step
				for _, p := range state.Params {
					for i := range p.Values {
						p.Values[i] += .001
					}
				}
				for _, p := range state.Adam {
					for i := range p.M {
						p.M[i] += .002
						p.V[i] += .003
					}
				}
				for _, p := range state.EMA {
					for i := range p.Values {
						p.Values[i] += .004
					}
				}
				if err = trainer.loadResidentCompletedState(state); err != nil {
					t.Fatal(step, err)
				}
				if err = reference.LoadState(state); err != nil {
					t.Fatal(err)
				}
				got, err = trainer.State()
				if err != nil {
					t.Fatal(err)
				}
				want, err = reference.State()
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatal("multistep public parity", step, err)
				}
				for n, x := range moments {
					y := trainer.Moments[n]
					if &x.M[0] != &y.M[0] || &x.V[0] != &y.V[0] {
						t.Fatal("moment array replaced", n)
					}
				}
				for n, x := range shadows {
					if &x[0] != &trainer.EMA[n][0] {
						t.Fatal("EMA array replaced", n)
					}
				}
			}
			before := state.Params[0].Values[0]
			trainer.params[state.Params[0].Name][0] += 1
			if state.Params[0].Values[0] != before {
				t.Fatal("owned boundary changed with live trainer")
			}
		})
	}
}
func residentLoadCompletedFixture(t *testing.T) (*FullTrainer, FullTrainingState) {
	t.Helper()
	trainer, template, snapshot := snapshotHandoffFixture(t, .9)
	s, err := FullStateFromResidentSnapshot(template, snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if err = trainer.LoadState(s); err != nil {
		t.Fatal(err)
	}
	s.Step = 2
	return trainer, s
}
func TestTrainingResidentLoadReuseAdmission(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*FullTrainer, *FullTrainingState)
	}{
		{"version", func(t *FullTrainer, s *FullTrainingState) { s.Version = 0 }},
		{"step", func(t *FullTrainer, s *FullTrainingState) { s.Step = 4 }},
		{"config-nan", func(t *FullTrainer, s *FullTrainingState) { s.AdamW.Epsilon = float32(math.NaN()) }},
		{"decay-nan", func(t *FullTrainer, s *FullTrainingState) { s.EMADecay = float32(math.NaN()) }},
		{"source-missing-moments", func(t *FullTrainer, s *FullTrainingState) { s.Adam = nil }},
		{"source-missing-EMA", func(t *FullTrainer, s *FullTrainingState) { s.EMA = nil }},
		{"source-duplicate", func(t *FullTrainer, s *FullTrainingState) { s.Params[1] = s.Params[0] }},
		{"live-parameter-nan", func(t *FullTrainer, s *FullTrainingState) { t.params[s.Params[0].Name][0] = float32(math.NaN()) }},
		{"live-empty-buffer", func(t *FullTrainer, s *FullTrainingState) {
			t.FlowLM.LatentMean = nil
			for i := range s.Buffers {
				if s.Buffers[i].Name == "flow_lm.emb_mean" {
					s.Buffers[i].Values = nil
				}
			}
		}},
		{"live-buffer-nan", func(t *FullTrainer, s *FullTrainingState) {
			for _, x := range fullBufferMap(t.FlowLM, t.Flow) {
				x[0] = float32(math.NaN())
				break
			}
		}},
		{"live-EMA-nan", func(t *FullTrainer, s *FullTrainingState) { t.EMA[s.Params[0].Name][0] = float32(math.NaN()) }},
		{"config", func(t *FullTrainer, s *FullTrainingState) { s.AdamW.Epsilon *= 2 }},
		{"decay", func(t *FullTrainer, s *FullTrainingState) { s.EMADecay = .5 }},
		{"parameter-name", func(t *FullTrainer, s *FullTrainingState) { s.Params[0].Name = "foreign" }},
		{"buffer-name", func(t *FullTrainer, s *FullTrainingState) { s.Buffers[0].Name = "foreign" }},
		{"negative-V", func(t *FullTrainer, s *FullTrainingState) { s.Adam[0].V[0] = -1 }},
		{"source-nan", func(t *FullTrainer, s *FullTrainingState) { s.Params[0].Values[0] = float32(math.NaN()) }},
		{"source-alias", func(t *FullTrainer, s *FullTrainingState) { s.Adam[0].M = s.Params[0].Values }},
		{"source-live-alias", func(t *FullTrainer, s *FullTrainingState) { s.Params[0].Values = t.params[s.Params[0].Name] }},
		{"live-moment-alias", func(t *FullTrainer, s *FullTrainingState) {
			n := s.Params[0].Name
			x := t.Moments[n]
			x.M = t.params[n]
			t.Moments[n] = x
		}},
		{"live-EMA-alias", func(t *FullTrainer, s *FullTrainingState) { n := s.Params[0].Name; t.EMA[n] = t.Moments[n].M }},
		{"missing-moments", func(t *FullTrainer, s *FullTrainingState) { t.Moments = nil }},
		{"foreign-moment", func(t *FullTrainer, s *FullTrainingState) {
			n := s.Params[0].Name
			x := t.Moments[n]
			delete(t.Moments, n)
			t.Moments["foreign"] = x
		}},
		{"moment-shape", func(t *FullTrainer, s *FullTrainingState) {
			n := s.Params[0].Name
			x := t.Moments[n]
			x.M = x.M[:len(x.M)-1]
			t.Moments[n] = x
		}},
		{"moment-nan", func(t *FullTrainer, s *FullTrainingState) { t.Moments[s.Params[0].Name].M[0] = float32(math.NaN()) }},
		{"live-negative-V", func(t *FullTrainer, s *FullTrainingState) { t.Moments[s.Params[0].Name].V[0] = -1 }},
		{"missing-EMA", func(t *FullTrainer, s *FullTrainingState) { t.EMA = nil }},
		{"foreign-EMA", func(t *FullTrainer, s *FullTrainingState) {
			n := s.Params[0].Name
			x := t.EMA[n]
			delete(t.EMA, n)
			t.EMA["foreign"] = x
		}},
		{"EMA-shape", func(t *FullTrainer, s *FullTrainingState) {
			n := s.Params[0].Name
			t.EMA[n] = t.EMA[n][:len(t.EMA[n])-1]
		}},
		{"buffer-live-alias", func(t *FullTrainer, s *FullTrainingState) {
			s.Buffers[0].Values = fullBufferMap(t.FlowLM, t.Flow)[s.Buffers[0].Name]
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			trainer, s := residentLoadCompletedFixture(t)
			c.mutate(trainer, &s)
			// Capture raw destination storage, including malformed/nonfinite state.
			raw := map[string][]float32{}
			for n, x := range trainer.params {
				raw["p:"+n] = append([]float32(nil), x...)
			}
			for n, x := range fullBufferMap(trainer.FlowLM, trainer.Flow) {
				raw["b:"+n] = append([]float32(nil), x...)
			}
			for n, x := range trainer.Moments {
				raw["m:"+n] = append([]float32(nil), x.M...)
				raw["v:"+n] = append([]float32(nil), x.V...)
			}
			for n, x := range trainer.EMA {
				raw["e:"+n] = append([]float32(nil), x...)
			}
			step := trainer.StepCount
			if err := trainer.loadResidentCompletedState(s); err == nil {
				t.Fatal("bad load admitted")
			}
			sameBits := func(a, b []float32) {
				t.Helper()
				if len(a) != len(b) {
					t.Fatal("mutated shape")
				}
				for i, x := range a {
					if math.Float32bits(x) != math.Float32bits(b[i]) {
						t.Fatal("rejected load mutated destination")
					}
				}
			}
			// Exact bits here only check unchanged owned state, including NaN payloads;
			// they are not numerical/generated-signal equivalence gates.
			for n, x := range trainer.params {
				sameBits(x, raw["p:"+n])
			}
			for n, x := range fullBufferMap(trainer.FlowLM, trainer.Flow) {
				sameBits(x, raw["b:"+n])
			}
			for n, x := range trainer.Moments {
				sameBits(x.M, raw["m:"+n])
				sameBits(x.V, raw["v:"+n])
			}
			for n, x := range trainer.EMA {
				sameBits(x, raw["e:"+n])
			}
			if trainer.StepCount != step {
				t.Fatal("step advanced")
			}
		})
	}
	if err := (*FullTrainer)(nil).loadResidentCompletedState(FullTrainingState{}); err == nil {
		t.Fatal("nil trainer")
	}
}
func TestTrainingResidentLoadRanges(t *testing.T) {
	for _, r := range [][]residentLoadRange{{{0, 4}}, {{4, 4}}, {{5, 4}}, {{4, 12}, {8, 16}}, {{4, 16}, {5, 6}, {7, 8}}} {
		if validateResidentLoadRanges(r) == nil {
			t.Fatal("bad ranges", r)
		}
	}
	if err := validateResidentLoadRanges([]residentLoadRange{{12, 16}, {4, 8}, {8, 12}}); err != nil {
		t.Fatal("adjacent independent storage", err)
	}
}
func BenchmarkTrainingResidentLoadReuse(b *testing.B) {
	raw, err := os.ReadFile("testdata/training_step_pytorch.json")
	if err != nil {
		b.Fatal(err)
	}
	var oracle trainingStepOracle
	if err = json.Unmarshal(raw, &oracle); err != nil {
		b.Fatal(err)
	}
	for _, mode := range []string{"public-replace", "private-reuse"} {
		b.Run(mode, func(b *testing.B) {
			lm, flow, w := trainingStepModelsFromOracle(b, oracle)
			w.Layers = []LinearF32{{In: 2, Out: 256, Weight: make([]float32, 512), Bias: make([]float32, 256)}, {In: 256, Out: 256, Weight: make([]float32, 65536), Bias: make([]float32, 256)}, {In: 256, Out: 1, Weight: make([]float32, 256), Bias: make([]float32, 1)}}
			trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
			if err != nil {
				b.Fatal(err)
			}
			state, err := trainer.State()
			if err != nil {
				b.Fatal(err)
			}
			snapshot, err := ResidentSnapshotFromFullState(state)
			if err != nil {
				b.Fatal(err)
			}
			snapshot.Step = 1
			state, err = FullStateFromResidentSnapshot(state, snapshot)
			if err != nil {
				b.Fatal(err)
			}
			if err = trainer.LoadState(state); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				state.Step = trainer.StepCount + 1
				if mode == "private-reuse" {
					err = trainer.loadResidentCompletedState(state)
				} else {
					err = trainer.LoadState(state)
				}
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
