package training

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

type residentHostBuffer struct {
	values []float32
	driver *residentHostDriver
	closed bool
}

func (b *residentHostBuffer) Upload(ctx context.Context, x []float32) error {
	if err := b.driver.event("upload"); err != nil {
		return err
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if len(x) != len(b.values) {
		return errors.New("upload shape")
	}
	copy(b.values, x)
	return nil
}
func (b *residentHostBuffer) Download(ctx context.Context, x []float32) error {
	if err := b.driver.event("download"); err != nil {
		return err
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	copy(x, b.values)
	return nil
}
func (b *residentHostBuffer) Close() error {
	if err := b.driver.event("close"); err != nil {
		return err
	}
	b.closed = true
	return nil
}

type residentHostDriver struct {
	events    map[string]int
	fail      string
	at        int
	sentinel  error
	updateNaN bool
	buffers   []*residentHostBuffer
}

func (d *residentHostDriver) event(name string) error {
	if d.events == nil {
		d.events = map[string]int{}
	}
	d.events[name]++
	if d.fail == name && d.events[name] == d.at {
		return d.sentinel
	}
	return nil
}
func (d *residentHostDriver) NewBuffer(ctx context.Context, n int) (ResidentBuffer, error) {
	b := &residentHostBuffer{values: make([]float32, n), driver: d}
	d.buffers = append(d.buffers, b)
	return b, d.event("new")
}
func (d *residentHostDriver) Accumulate(ctx context.Context, dst, a, b ResidentBuffer, n int, scale float32) error {
	if err := d.event("accumulate"); err != nil {
		return err
	}
	x, y, z := a.(*residentHostBuffer).values, b.(*residentHostBuffer).values, dst.(*residentHostBuffer).values
	for i := 0; i < n; i++ {
		z[i] = x[i] + y[i]*scale
	}
	return nil
}
func (d *residentHostDriver) NormStats(ctx context.Context, dst, x ResidentBuffer, n int, positive bool) error {
	if err := d.event("stats"); err != nil {
		return err
	}
	copy(dst.(*residentHostBuffer).values, hostNormStats(x.(*residentHostBuffer).values, positive))
	return nil
}
func hostNormStats(x []float32, positive bool) []float32 {
	count, _ := NormStatsShape(len(x))
	out := make([]float32, count)
	for chunk := 0; chunk < count/3; chunk++ {
		start, end := chunk*NormChunkElements, min(len(x), (chunk+1)*NormChunkElements)
		maximum, invalid := float32(0), float32(0)
		for _, v := range x[start:end] {
			if !residentFinite(v) || (positive && v < 0) {
				invalid = 1
			} else {
				maximum = max(maximum, float32(math.Abs(float64(v))))
			}
		}
		var sum, correction float32
		if maximum > 0 {
			for _, v := range x[start:end] {
				scaled := v / maximum
				term := scaled*scaled - correction
				next := sum + term
				correction = (next - sum) - term
				sum = next
			}
		}
		out[3*chunk], out[3*chunk+1], out[3*chunk+2] = maximum, sum, invalid
	}
	return out
}
func (d *residentHostDriver) ValidateUpdate(ctx context.Context, p, g, m, v, ema ResidentBuffer, n int, step AdamWStep) error {
	return d.event("validate")
}
func (d *residentHostDriver) Update(ctx context.Context, p, g, m, v, ema ResidentBuffer, n int, step AdamWStep) error {
	if err := d.event("update"); err != nil {
		return err
	}
	pv, gv, mv, vv, ev := p.(*residentHostBuffer).values, g.(*residentHostBuffer).values, m.(*residentHostBuffer).values, v.(*residentHostBuffer).values, ema.(*residentHostBuffer).values
	for i := 0; i < n; i++ {
		pv[i], mv[i], vv[i], ev[i] = adamSchedule(step, pv[i], gv[i], mv[i], vv[i], ev[i])
	}
	if d.updateNaN {
		pv[0] = float32(math.NaN())
	}
	return nil
}
func residentFixture() ResidentSnapshot {
	return ResidentSnapshot{Config: ResidentConfig{.01, .9, .999, 1e-8, .01, .9}, Parameters: map[string]ResidentParameter{"a": {Parameter: []float32{.3, -.7}, M: make([]float32, 2), V: make([]float32, 2), EMA: []float32{.3, -.7}}, "b": {Parameter: []float32{.5}, M: make([]float32, 1), V: make([]float32, 1), EMA: []float32{.5}}}}
}
func requireResident(t *testing.T) (*ResidentState, *residentHostDriver) {
	t.Helper()
	d := &residentHostDriver{sentinel: errors.New("native uncertain")}
	s, err := NewResidentState(context.Background(), d, residentFixture(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	return s, d
}
func TestTrainingResidentUpdateResume(t *testing.T) {
	a, d := requireResident(t)
	ctx := context.Background()
	original := residentFixture()
	bytes, err := ResidentBytes(original)
	if err != nil || bytes != 120 {
		t.Fatal("logical byte accounting", bytes, err)
	} // 3*32 + 2*3*4
	if _, err := a.Update(ctx, 1); err == nil {
		t.Fatal("missing gradients")
	}
	if err := a.Accumulate(ctx, map[string][]float32{"a": {3, 0}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Snapshot(ctx); err == nil {
		t.Fatal("dirty snapshot")
	}
	if _, err := a.Update(ctx, 1); err == nil {
		t.Fatal("partial gradient set")
	}
	if err := a.Accumulate(ctx, map[string][]float32{"a": {0, 4}, "b": {0}}, 1); err != nil {
		t.Fatal(err)
	}
	norm, err := a.Update(ctx, 2)
	if err != nil || math.Abs(norm-5) > 1e-6 {
		t.Fatal(norm, err)
	}
	snapshot, err := a.Snapshot(ctx)
	if err != nil || snapshot.Step != 1 {
		t.Fatal(err)
	}
	p, _ := NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, .4, 1)
	for key, state := range snapshot.Parameters {
		g := []float32{3, 4}
		if key == "b" {
			g = []float32{0}
		}
		for i, v := range original.Parameters[key].Parameter {
			want, m, vv, ema := adamSchedule(p, v, g[i], 0, 0, v)
			if math.Abs(float64(state.Parameter[i]-want)) > 1e-7 || math.Abs(float64(state.M[i]-m)) > 1e-7 || math.Abs(float64(state.V[i]-vv)) > 1e-7 || math.Abs(float64(state.EMA[i]-ema)) > 1e-7 {
				t.Fatal("clipped update", key, i, state)
			}
		}
	}
	b, err := NewResidentState(ctx, &residentHostDriver{}, snapshot, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []*ResidentState{a, b} {
		if err := s.Accumulate(ctx, map[string][]float32{"a": {-.2, .8}, "b": {.4}}, .5); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Update(ctx, 1); err != nil {
			t.Fatal(err)
		}
	}
	as, err := a.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	bs, err := b.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for key, p := range as.Parameters {
		for i, x := range p.Parameter {
			if math.Abs(float64(x-bs.Parameters[key].Parameter[i])) > 1e-7 {
				t.Fatal("resume numerical drift")
			}
		}
	}
	as.Parameters["a"].Parameter[0] = 42
	again, err := a.Snapshot(ctx)
	if err != nil || again.Parameters["a"].Parameter[0] == 42 {
		t.Fatal("snapshot borrowed device storage")
	}
	if a.StepCount() != 2 || d.events["new"] != 18 {
		t.Fatal("resident allocations changed", a.StepCount(), d.events)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if a.Close() != nil {
		t.Fatal("double close")
	}
	if _, err := a.Snapshot(ctx); err == nil {
		t.Fatal("closed snapshot")
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	var nilState *ResidentState
	if nilState.Close() != nil || nilState.StepCount() != 0 {
		t.Fatal("nil owner")
	}
}
func TestTrainingResidentAdmission(t *testing.T) {
	ctx := context.Background()
	d := &residentHostDriver{}
	fixture := residentFixture()
	for _, mutate := range []func(*ResidentSnapshot){func(s *ResidentSnapshot) { s.Parameters = nil }, func(s *ResidentSnapshot) { s.Config.LearningRate = 0 }, func(s *ResidentSnapshot) { s.Step = math.MaxUint64 }, func(s *ResidentSnapshot) { s.Parameters[""] = s.Parameters["a"] }, func(s *ResidentSnapshot) { p := s.Parameters["a"]; p.M = nil; s.Parameters["a"] = p }, func(s *ResidentSnapshot) { s.Parameters["a"].Parameter[0] = float32(math.NaN()) }, func(s *ResidentSnapshot) { s.Parameters["a"].V[0] = -1 }} {
		bad := residentFixture()
		mutate(&bad)
		if _, err := ResidentBytes(bad); err == nil {
			t.Fatal("bad resident snapshot")
		}
	}
	if _, err := NewResidentState(nil, d, fixture, 1<<20); err == nil {
		t.Fatal("nil context")
	}
	if _, err := NewResidentState(ctx, nil, fixture, 1<<20); err == nil {
		t.Fatal("nil driver")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := NewResidentState(cancelled, d, fixture, 1<<20); err == nil {
		t.Fatal("cancelled allocation")
	}
	if _, err := NewResidentState(ctx, d, fixture, 1); err == nil {
		t.Fatal("budget")
	}
	if len(d.buffers) != 0 {
		t.Fatal("rejected construction allocated")
	}
	s, _ := requireResident(t)
	for _, branches := range []map[string][]float32{nil, {"bad": {0}}, {"a": {1}}, {"a": {float32(math.Inf(1)), 0}}} {
		if err := s.Accumulate(ctx, branches, 1); err == nil {
			t.Fatal("bad branches")
		}
	}
	if s.Accumulate(ctx, map[string][]float32{"a": {0, 0}}, float32(math.NaN())) == nil {
		t.Fatal("bad scale")
	}
	if s.Accumulate(cancelled, map[string][]float32{"a": {0, 0}}, 1) == nil {
		t.Fatal("cancelled accumulation")
	}
	if _, err := s.Update(ctx, 0); err == nil {
		t.Fatal("bad norm ceiling")
	}
	if _, err := NewResidentState(ctx, d, ResidentSnapshot{}, 1<<20); err == nil {
		t.Fatal("invalid constructor snapshot")
	}
	if (*ResidentState)(nil).Accumulate(ctx, nil, 1) == nil {
		t.Fatal("nil state")
	}
}
func TestTrainingResidentFailureRetention(t *testing.T) {
	ctx := context.Background()
	for _, category := range []string{"new", "upload"} {
		d := &residentHostDriver{fail: category, at: 2, sentinel: errors.New("creation failed")}
		s, err := NewResidentState(ctx, d, residentFixture(), 1<<20)
		if !errors.Is(err, d.sentinel) || s == nil || len(s.owned) == 0 {
			t.Fatal("partial owner lost", category, err)
		}
		if s.Close() == nil {
			t.Fatal("uncertain construction freed")
		}
	}
	for _, category := range []string{"upload", "accumulate", "stats", "download", "update", "validate", "close"} {
		t.Run(category, func(t *testing.T) {
			s, d := requireResident(t)
			before, _ := s.Snapshot(ctx)
			d.fail, d.at = category, d.events[category]+1
			var err error
			switch category {
			case "upload", "accumulate":
				err = s.Accumulate(ctx, map[string][]float32{"a": {1, 2}, "b": {3}}, 1)
			case "close":
				err = s.Close()
			default:
				if e := s.Accumulate(ctx, map[string][]float32{"a": {1, 2}, "b": {3}}, 1); e != nil {
					t.Fatal(e)
				}
				_, err = s.Update(ctx, 1)
			}
			if !errors.Is(err, d.sentinel) {
				t.Fatal("lost error", err)
			}
			if category == "validate" {
				if s.terminal != nil {
					t.Fatal("pure preflight poisoned")
				}
				d.fail = ""
				after := ResidentSnapshot{Parameters: map[string]ResidentParameter{}}
				for _, e := range s.entries {
					after.Parameters[e.name] = ResidentParameter{Parameter: append([]float32(nil), e.parameter.(*residentHostBuffer).values...), M: append([]float32(nil), e.m.(*residentHostBuffer).values...), V: append([]float32(nil), e.v.(*residentHostBuffer).values...), EMA: append([]float32(nil), e.ema.(*residentHostBuffer).values...)}
				}
				if !reflect.DeepEqual(before.Parameters, after.Parameters) {
					t.Fatal("preflight mutated state")
				}
				return
			}
			calls := len(d.buffers)
			if s.terminal == nil || s.Close() == nil {
				t.Fatal("uncertain owner freed")
			}
			if _, err := s.Snapshot(ctx); err == nil {
				t.Fatal("terminal snapshot")
			}
			if _, err := s.Update(ctx, 1); err == nil || len(d.buffers) != calls {
				t.Fatal("uncertain update retried")
			}
			if s.StepCount() != 0 {
				t.Fatal("failed update advanced step")
			}
		})
	}
	s, d := requireResident(t)
	d.updateNaN = true
	if err := s.Accumulate(ctx, map[string][]float32{"a": {1, 2}, "b": {3}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Update(ctx, 1); err == nil || s.StepCount() != 0 || s.terminal == nil {
		t.Fatal("post-update invalid state committed")
	}
}
func TestTrainingResidentNormNumerics(t *testing.T) {
	for _, n := range []int{1, 255, 256, 257, 1025} {
		x := make([]float32, n)
		var square float64
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*.37)) * float32(1e20)
			square += float64(x[i]) * float64(x[i])
		}
		got, err := NormSquareFromStats(hostNormStats(x, false), n)
		if err != nil || math.Abs(got-square) > max(1, square)*3e-7 {
			t.Fatal("norm relative error", n, got, square, err)
		}
	}
	for _, values := range [][]float32{nil, {0, 1, 0}, {1, 0, 0}, {1, 2, 0}, {1, 1, 1}, {-1, 1, 0}, {float32(math.NaN()), 1, 0}, {1, -1, 0}} {
		if _, err := NormSquareFromStats(values, 1); err == nil {
			t.Fatal("bad norm summary")
		}
	}
	if _, err := NormSquareFromStats(nil, 0); err == nil {
		t.Fatal("bad n")
	}
	if _, err := NormStatsShape(1<<26 + 1); err == nil {
		t.Fatal("bad shape")
	}
	x := []float32{math.MaxFloat32, math.MaxFloat32}
	got, err := NormSquareFromStats(hostNormStats(x, false), 2)
	if err != nil || math.IsInf(got, 0) {
		t.Fatal("finite F32 square overflow", got, err)
	}
	for _, x := range [][]float32{{float32(math.NaN())}, {float32(math.Inf(1))}, {-1}} {
		if _, err := NormSquareFromStats(hostNormStats(x, true), 1); err == nil {
			t.Fatal("invalid state summary")
		}
	}
}
func TestTrainingResidentPyTorchUpdate(t *testing.T) {
	data, err := os.ReadFile("../../model/pockettts/testdata/training_step_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Schema                int    `json:"schema"`
		Revision              string `json:"upstream_revision"`
		Parameters, Gradients map[string][]float32
		Adam                  struct {
			LR, B1, B2, Eps, Decay, EMA float32
			Params, Shadows             map[string][]float32
		}
	}
	// Keep fixture tags explicit, independent of ResidentState's layout.
	var raw struct {
		Schema     int                  `json:"schema"`
		Revision   string               `json:"upstream_revision"`
		Parameters map[string][]float32 `json:"parameters"`
		Gradients  map[string][]float32 `json:"gradients"`
		Adam       struct {
			LR      float32              `json:"learning_rate"`
			B1      float32              `json:"beta1"`
			B2      float32              `json:"beta2"`
			Eps     float32              `json:"epsilon"`
			Decay   float32              `json:"weight_decay"`
			EMA     float32              `json:"ema_decay"`
			Params  map[string][]float32 `json:"parameters_after_step"`
			Shadows map[string][]float32 `json:"ema_after_step"`
		} `json:"adamw"`
	}
	if err = json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	oracle.Schema, oracle.Revision, oracle.Parameters, oracle.Gradients = raw.Schema, raw.Revision, raw.Parameters, raw.Gradients
	oracle.Adam.LR, oracle.Adam.B1, oracle.Adam.B2, oracle.Adam.Eps, oracle.Adam.Decay, oracle.Adam.EMA, oracle.Adam.Params, oracle.Adam.Shadows = raw.Adam.LR, raw.Adam.B1, raw.Adam.B2, raw.Adam.Eps, raw.Adam.Decay, raw.Adam.EMA, raw.Adam.Params, raw.Adam.Shadows
	if oracle.Schema != 1 || oracle.Revision != "0acce6b2f390150267557770d2098c5caa9a18ac" {
		t.Fatal("oracle provenance")
	}
	a := oracle.Adam
	initial := ResidentSnapshot{Config: ResidentConfig{a.LR, a.B1, a.B2, a.Eps, a.Decay, a.EMA}, Parameters: map[string]ResidentParameter{}}
	gradients := map[string][]float32{}
	for name := range a.Params {
		x := oracle.Parameters[name]
		initial.Parameters[name] = ResidentParameter{x, make([]float32, len(x)), make([]float32, len(x)), append([]float32(nil), x...)}
		gradients[name] = oracle.Gradients[name]
	}
	s, err := NewResidentState(context.Background(), &residentHostDriver{}, initial, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Accumulate(context.Background(), gradients, 1); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Update(context.Background(), math.MaxFloat32); err != nil {
		t.Fatal(err)
	}
	snapshot, err := s.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	count, maxError, square := 0, float64(0), float64(0)
	for name, p := range snapshot.Parameters {
		for j, got := range [][]float32{p.Parameter, p.EMA} {
			want := a.Params[name]
			if j == 1 {
				want = a.Shadows[name]
			}
			for i, x := range got {
				diff := math.Abs(float64(x) - float64(want[i]))
				if diff > 4e-6 {
					t.Fatal("resident PyTorch drift", name, i, diff)
				}
				maxError = math.Max(maxError, diff)
				square += diff * diff
				count++
			}
		}
	}
	t.Logf("resident host update vs PyTorch values=%d max_abs=%g rms=%g", count, maxError, math.Sqrt(square/float64(count)))
}

type nilResidentDriver struct{ residentHostDriver }

func (d *nilResidentDriver) NewBuffer(context.Context, int) (ResidentBuffer, error) { return nil, nil }
func TestTrainingResidentBoundaryFailures(t *testing.T) {
	ctx := context.Background()
	if owner, err := NewResidentState(ctx, &nilResidentDriver{}, residentFixture(), 1<<20); err == nil || owner == nil || owner.Close() == nil {
		t.Fatal("nil native allocation owner")
	}
	for _, failAt := range []int{2, 3, 5} {
		s, d := requireResident(t)
		if err := s.Accumulate(ctx, map[string][]float32{"a": {1, 2}, "b": {3}}, 1); err != nil {
			t.Fatal(err)
		}
		d.fail, d.at = "stats", failAt
		if _, err := s.Update(ctx, 1); err == nil || s.StepCount() != 0 || s.terminal == nil {
			t.Fatal("post-update summary failure committed", failAt)
		}
	}
	s, d := requireResident(t)
	d.fail, d.at = "download", 1
	if _, err := s.Snapshot(ctx); err == nil || s.terminal == nil {
		t.Fatal("snapshot failure not terminal")
	}
	s, _ = requireResident(t)
	s.entries[0].v.(*residentHostBuffer).values[0] = -1
	if _, err := s.Snapshot(ctx); err == nil || s.terminal == nil {
		t.Fatal("corrupt snapshot")
	}
	s, _ = requireResident(t)
	s.step = uint64(math.MaxInt)
	if _, err := s.Update(ctx, 1); err == nil {
		t.Fatal("step overflow")
	}
	fixture := residentFixture()
	fixture.Step = uint64(math.MaxInt)
	if _, err := ResidentBytes(fixture); err != nil {
		t.Fatal("final completed boundary must snapshot", err)
	}
}
