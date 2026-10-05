package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"testing"
)

func snapshotHandoffFixture(t *testing.T, ema float32) (*FullTrainer, FullTrainingState, train.ResidentSnapshot) {
	t.Helper()
	o := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, o)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), ema)
	if err != nil {
		t.Fatal(err)
	}
	template, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := ResidentSnapshotFromFullState(template)
	if err != nil {
		t.Fatal(err)
	}
	snapshot.Step = 1
	for name, p := range snapshot.Parameters {
		for i := range p.Parameter {
			p.Parameter[i] += .01
			p.M[i] = .02
			p.V[i] = .03
			p.EMA[i] = .04
		}
		snapshot.Parameters[name] = p
	}
	return trainer, template, snapshot
}
func TestTrainingResidentSnapshotHandoffOwnership(t *testing.T) {
	for _, ema := range []float32{0, .9} {
		t.Run(map[bool]string{true: "EMA", false: "noEMA"}[ema > 0], func(t *testing.T) {
			trainer, template, snapshot := snapshotHandoffFixture(t, ema)
			public, err := FullStateFromResidentSnapshot(template, snapshot)
			if err != nil {
				t.Fatal(err)
			}
			// Keep views solely to prove provenance/ownership transfer in this test.
			// Production callers relinquish the snapshot and all aliases after success.
			refs := snapshot.Parameters
			got, err := takeFullStateFromResidentSnapshot(template, &snapshot)
			if err != nil {
				t.Fatal(err)
			}
			if snapshot.Parameters != nil || snapshot.Step != 0 || snapshot.Config != (train.ResidentConfig{}) {
				t.Fatal("source not consumed")
			}
			if !reflect.DeepEqual(public, got) {
				t.Fatal("public conversion parity")
			}
			for i, p := range got.Params {
				src := refs[p.Name]
				if !f32SlicesOverlap(p.Values, src.Parameter) || !f32SlicesOverlap(got.Adam[i].M, src.M) || !f32SlicesOverlap(got.Adam[i].V, src.V) {
					t.Fatal("not moved", p.Name)
				}
				if f32SlicesOverlap(public.Params[i].Values, src.Parameter) || f32SlicesOverlap(public.Adam[i].M, src.M) || f32SlicesOverlap(public.Adam[i].V, src.V) {
					t.Fatal("public clone contract", p.Name)
				}
				if ema > 0 && (!f32SlicesOverlap(got.EMA[i].Values, src.EMA) || f32SlicesOverlap(public.EMA[i].Values, src.EMA)) {
					t.Fatal("EMA ownership", p.Name)
				}
			}
			for i, x := range template.Buffers {
				if f32SlicesOverlap(x.Values, got.Buffers[i].Values) {
					t.Fatal("borrowed frozen template buffer")
				}
			}
			if err = trainer.LoadState(got); err != nil {
				t.Fatal(err)
			}
			loaded, err := trainer.State()
			if err != nil || !sameResidentHostBoundary(got, loaded) {
				t.Fatal("load parity", err)
			}
			// The owned backend boundary must not alias mutable trainer state.
			for _, p := range got.Params {
				if f32SlicesOverlap(p.Values, trainer.params[p.Name]) {
					t.Fatal("trainer aliases boundary parameter")
				}
			}
			for _, p := range got.Adam {
				if f32SlicesOverlap(p.M, trainer.Moments[p.Name].M) || f32SlicesOverlap(p.V, trainer.Moments[p.Name].V) {
					t.Fatal("trainer aliases boundary moment")
				}
			}
			for _, p := range got.EMA {
				if f32SlicesOverlap(p.Values, trainer.EMA[p.Name]) {
					t.Fatal("trainer aliases boundary EMA")
				}
			}
			// Public conversion mutation cannot alter transferred state, and changing
			// live trainer parameters is still rejected by the exact boundary contract.
			public.Params[0].Values[0] += 1
			if reflect.DeepEqual(public.Params[0].Values, got.Params[0].Values) {
				t.Fatal("public aliases move")
			}
			trainer.params[got.Params[0].Name][0] += 1
			if sameResidentLiveBoundary(got, trainer) {
				t.Fatal("lost mutation guard")
			}
		})
	}
}
func TestTrainingResidentSnapshotHandoffRejection(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*FullTrainingState, *train.ResidentSnapshot)
	}{
		{"template", func(s *FullTrainingState, x *train.ResidentSnapshot) { s.Version = 0 }},
		{"config", func(s *FullTrainingState, x *train.ResidentSnapshot) { x.Config.Epsilon *= 2 }},
		{"step", func(s *FullTrainingState, x *train.ResidentSnapshot) { s.Step = 2 }},
		{"step-ceiling", func(s *FullTrainingState, x *train.ResidentSnapshot) { x.Step = math.MaxUint64 }},
		{"inventory", func(s *FullTrainingState, x *train.ResidentSnapshot) {
			for n := range x.Parameters {
				delete(x.Parameters, n)
				break
			}
		}},
		{"renamed", func(s *FullTrainingState, x *train.ResidentSnapshot) {
			for n, p := range x.Parameters {
				delete(x.Parameters, n)
				x.Parameters["unknown"] = p
				break
			}
		}},
		{"shape", func(s *FullTrainingState, x *train.ResidentSnapshot) {
			for n, p := range x.Parameters {
				p.Parameter = append(p.Parameter, 0)
				p.M = append(p.M, 0)
				p.V = append(p.V, 0)
				p.EMA = append(p.EMA, 0)
				x.Parameters[n] = p
				break
			}
		}},
		{"nonfinite", func(s *FullTrainingState, x *train.ResidentSnapshot) {
			for _, p := range x.Parameters {
				p.Parameter[0] = float32(math.Inf(1))
				break
			}
		}},
		{"negative-V", func(s *FullTrainingState, x *train.ResidentSnapshot) {
			for _, p := range x.Parameters {
				p.V[0] = -1
				break
			}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			trainer, template, snapshot := snapshotHandoffFixture(t, .9)
			c.mutate(&template, &snapshot)
			saved := snapshot
			params := snapshot.Parameters
			step := snapshot.Step
			before, err := trainer.State()
			if err != nil {
				t.Fatal(err)
			}
			if _, err = takeFullStateFromResidentSnapshot(template, &snapshot); err == nil {
				t.Fatal("invalid handoff")
			}
			if snapshot.Step != step || snapshot.Parameters == nil || len(snapshot.Parameters) != len(params) {
				t.Fatal("invalid source consumed")
			}
			for n, p := range params {
				q := snapshot.Parameters[n]
				if !f32SlicesOverlap(p.Parameter, q.Parameter) {
					t.Fatal("invalid source replaced")
				}
			}
			if !reflect.DeepEqual(saved, snapshot) {
				t.Fatal("source mutated on rejection")
			}
			after, err := trainer.State()
			if err != nil || !reflect.DeepEqual(before, after) {
				t.Fatal("trainer mutated")
			}
		})
	}
	if _, err := takeFullStateFromResidentSnapshot(FullTrainingState{}, nil); err == nil {
		t.Fatal("nil source")
	}
}
func TestTrainingResidentSnapshotHandoffAllocations(t *testing.T) {
	_, template, snapshot := snapshotHandoffFixture(t, .9)
	public := testing.AllocsPerRun(10, func() {
		if _, err := FullStateFromResidentSnapshot(template, snapshot); err != nil {
			panic(err)
		}
	})
	moved := testing.AllocsPerRun(10, func() {
		x := snapshot
		if _, err := takeFullStateFromResidentSnapshot(template, &x); err != nil {
			panic(err)
		}
	})
	if moved >= public {
		t.Fatal("no allocation reduction", moved, public)
	}
	t.Logf("tiny snapshot conversion allocations/op public owned clone=%g -> private ownership move=%g; full native snapshot downloads unchanged", public, moved)
}
func BenchmarkTrainingResidentSnapshotHandoff(b *testing.B) {
	s := validationStateFixture(32768)
	snapshot, err := ResidentSnapshotFromFullState(s)
	if err != nil {
		b.Fatal(err)
	}
	b.Run("public-clone", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := FullStateFromResidentSnapshot(s, snapshot); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("private-move", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			x := snapshot
			if _, err := takeFullStateFromResidentSnapshot(s, &x); err != nil {
				b.Fatal(err)
			}
		}
	})
}
