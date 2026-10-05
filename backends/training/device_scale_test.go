package training

import (
	"context"
	"errors"
	"math"
	"testing"
)

type deviceScaleHostDriver struct {
	globalNormHostDriver
	scaleBuffer ResidentBuffer
}

func (d *deviceScaleHostDriver) ValidateScaledUpdate(ctx context.Context, p, g, m, v, ema, clip ResidentBuffer, n int, step AdamWStep) error {
	if err := step.Validate(); err != nil {
		return err
	}
	if step.GradientScale != 1 {
		return errors.New("host scale injected")
	}
	b, ok := clip.(*residentHostBuffer)
	if !ok || len(b.values) != 4 {
		return errors.New("scale extent")
	}
	d.scaleBuffer = clip
	return d.event("scaled-admit")
}
func (d *deviceScaleHostDriver) UpdateScaled(ctx context.Context, p, g, m, v, ema, clip ResidentBuffer, n int, step AdamWStep) error {
	if err := d.event("scaled-update"); err != nil {
		return err
	}
	if step.GradientScale != 1 || clip != d.scaleBuffer {
		return errors.New("device buffer/constants changed")
	}
	step.GradientScale = clip.(*residentHostBuffer).values[3]
	return d.residentHostDriver.Update(ctx, p, g, m, v, ema, n, step)
}
func TestTrainingDeviceScaleUpdateResume(t *testing.T) {
	ctx := context.Background()
	for _, ema := range []float32{0, .9} {
		t.Run("ema"+map[bool]string{true: "enabled", false: "disabled"}[ema > 0], func(t *testing.T) {
			fixture := residentFixture()
			fixture.Config.EMADecay = ema
			d := &deviceScaleHostDriver{}
			d.sentinel = errors.New("injected failure")
			s, err := NewDeviceScaleResidentState(ctx, d, fixture, 168)
			if err != nil {
				t.Fatal(err)
			}
			legacy, err := NewResidentState(ctx, &residentHostDriver{}, fixture, 120)
			if err != nil {
				t.Fatal(err)
			}
			defer legacy.Close()
			for i, ceiling := range []float32{6.5, 1e20, .01, 1} {
				gradients := map[string][]float32{"a": {3, 4}, "b": {12}}
				if i == 3 {
					gradients = map[string][]float32{"a": {0, 0}, "b": {0}}
				}
				if err = s.Accumulate(ctx, gradients, 1); err != nil {
					t.Fatal(err)
				}
				if err = legacy.Accumulate(ctx, gradients, 1); err != nil {
					t.Fatal(err)
				}
				before := d.events["download"]
				norm, e := s.Update(ctx, ceiling)
				if e != nil {
					t.Fatal(e)
				}
				wantNorm, e := legacy.Update(ctx, ceiling)
				if e != nil {
					t.Fatal(e)
				}
				if math.Abs(norm-wantNorm) > 3e-7 {
					t.Fatal("norm", norm, wantNorm)
				}
				if d.events["download"]-before != 9 {
					t.Fatal("diagnostic+post-update count", d.events)
				}
				got, e := s.Snapshot(ctx)
				if e != nil {
					t.Fatal(e)
				}
				want, e := legacy.Snapshot(ctx)
				if e != nil {
					t.Fatal(e)
				}
				for name, p := range got.Parameters {
					q := want.Parameters[name]
					for j, x := range [][]float32{p.Parameter, p.M, p.V, p.EMA} {
						y := [][]float32{q.Parameter, q.M, q.V, q.EMA}[j]
						for k, v := range x {
							if math.Abs(float64(v-y[k])) > 3e-7 {
								t.Fatal(i, name, j, k, v, y[k])
							}
						}
					}
				}
				if i == 1 {
					if err = s.Close(); err != nil {
						t.Fatal(err)
					}
					d = &deviceScaleHostDriver{}
					d.sentinel = errors.New("injected failure")
					s, err = NewDeviceScaleResidentState(ctx, d, got, 168)
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			if d.events["scaled-update"] != 4 {
				t.Fatal("dispatch coverage", d.events)
			}
			if err = s.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
func TestTrainingDeviceScaleAdmissionRetention(t *testing.T) {
	ctx := context.Background()
	for _, driver := range []ResidentDriver{&residentHostDriver{}, &globalNormHostDriver{}} {
		if s, err := NewDeviceScaleResidentState(ctx, driver, residentFixture(), 168); s != nil || err == nil {
			t.Fatal("missing capability")
		}
	}
	d := &deviceScaleHostDriver{}
	d.sentinel = errors.New("injected failure")
	if s, err := NewDeviceScaleResidentState(ctx, d, residentFixture(), 167); s != nil || err == nil || d.events["new"] != 0 {
		t.Fatal("budget before allocation")
	}
	for _, event := range []string{"scaled-admit", "merge-admit", "clip-admit", "merge", "clip", "download", "stats", "scaled-update"} {
		t.Run(event, func(t *testing.T) {
			d := &deviceScaleHostDriver{}
			d.sentinel = errors.New("injected failure")
			s, err := NewDeviceScaleResidentState(ctx, d, residentFixture(), 168)
			if err != nil {
				t.Fatal(err)
			}
			if err = s.Accumulate(ctx, map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
				t.Fatal(err)
			}
			before := d.events["upload"]
			fail := d.events[event] + 1
			d.fail, d.at = event, fail
			if _, err = s.Update(ctx, 1); err == nil {
				t.Fatal("failure missing")
			}
			if d.events["update"] != 0 || s.StepCount() != 0 {
				t.Fatal("mutation or step after failure", d.events)
			}
			if event == "scaled-admit" || event == "merge-admit" || event == "clip-admit" {
				if s.terminal != nil || d.events["upload"] != before {
					t.Fatal("pure preflight")
				}
				d.fail = ""
				if _, err = s.Update(ctx, 1); err != nil {
					t.Fatal("preflight retry", err)
				}
				if err = s.Close(); err != nil {
					t.Fatal(err)
				}
			} else {
				if s.terminal == nil {
					t.Fatal("uncertain owner discarded")
				}
				snapshotEvents := d.events["download"]
				if _, err = s.Snapshot(ctx); err == nil || d.events["download"] != snapshotEvents {
					t.Fatal("snapshot after uncertainty")
				}
				if _, err = s.Update(ctx, 1); err == nil || s.Close() == nil {
					t.Fatal("uncertain retry/close")
				}
			}
		})
	}
	// Fail after one update: no successful step or retry; mutated owners retained.
	d = &deviceScaleHostDriver{}
	d.sentinel = errors.New("injected failure")
	s, err := NewDeviceScaleResidentState(ctx, d, residentFixture(), 168)
	if err != nil {
		t.Fatal(err)
	}
	s.Accumulate(ctx, map[string][]float32{"a": {3, 4}, "b": {12}}, 1)
	d.fail, d.at = "scaled-update", 2
	if _, err = s.Update(ctx, 1); err == nil || d.events["update"] != 1 || s.StepCount() != 0 || s.terminal == nil || s.Close() == nil {
		t.Fatal("partial mutation retention")
	}
}
func TestTrainingDeviceScaleLegacyIsolation(t *testing.T) {
	d := &deviceScaleHostDriver{}
	d.sentinel = errors.New("injected failure")
	ctx := context.Background()
	s, err := NewGlobalNormResidentState(ctx, d, residentFixture(), 168)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err = s.Accumulate(ctx, map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Update(ctx, 6.5); err != nil {
		t.Fatal(err)
	}
	if d.events["scaled-update"] != 0 || d.events["scaled-admit"] != 0 || d.events["update"] != 2 {
		t.Fatal("legacy routed to device-scale", d.events)
	}
}

func TestTrainingDeviceScaleBoundaryRetention(t *testing.T) {
	ctx := context.Background()
	// Reject the second parameter in complete-pass admission; first must not mutate.
	d := &deviceScaleHostDriver{}
	d.sentinel = errors.New("injected")
	s, err := NewDeviceScaleResidentState(ctx, d, residentFixture(), 168)
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Accumulate(ctx, map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
		t.Fatal(err)
	}
	before := d.events["upload"]
	d.fail, d.at = "scaled-admit", 2
	if _, err = s.Update(ctx, 1); err == nil || s.terminal != nil || d.events["merge"] != 0 || d.events["scaled-update"] != 0 || d.events["upload"] != before {
		t.Fatal("second preflight changed state")
	}
	d.fail = ""
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	// Invalid gradient becomes invalid norm diagnostic, never an update.
	d = &deviceScaleHostDriver{}
	s, err = NewDeviceScaleResidentState(ctx, d, residentFixture(), 168)
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Accumulate(ctx, map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
		t.Fatal(err)
	}
	s.entries[0].gradient.(*residentHostBuffer).values[0] = float32(math.NaN())
	if _, err = s.Update(ctx, 1); err == nil || s.terminal == nil || d.events["scaled-update"] != 0 || s.Close() == nil {
		t.Fatal("invalid diagnostic mutated state")
	}
	// Failure in first post-update statistics retains both mutated parameters.
	d = &deviceScaleHostDriver{}
	d.sentinel = errors.New("injected post-update")
	s, err = NewDeviceScaleResidentState(ctx, d, residentFixture(), 168)
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Accumulate(ctx, map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
		t.Fatal(err)
	}
	d.fail, d.at = "stats", 3
	if _, err = s.Update(ctx, 1); err == nil || s.terminal == nil || s.StepCount() != 0 || d.events["scaled-update"] != 2 || s.Close() == nil {
		t.Fatal("post-update failure committed/discarded")
	}
}
