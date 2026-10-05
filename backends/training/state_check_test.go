package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestTrainingStateCheckDiagnostic(t *testing.T) {
	for _, maximum := range []float32{0, math.SmallestNonzeroFloat32, 1, math.MaxFloat32} {
		if err := DecodeStateCheck([]float32{maximum, 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	for _, values := range [][]float32{nil, {0, 0, 0}, {-1, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}, {float32(math.NaN()), 0, 0, 0}, {1, 0, float32(math.Inf(1)), 0}} {
		if DecodeStateCheck(values) == nil {
			t.Fatal("invalid diagnostic", values)
		}
	}
}
func TestTrainingStateCheckUpdateParity(t *testing.T) {
	for _, ema := range []float32{0, .9} {
		t.Run(map[bool]string{true: "EMA", false: "noEMA"}[ema > 0], func(t *testing.T) {
			ctx := context.Background()
			fixture := residentFixture()
			fixture.Config.EMADecay = ema
			d := &deviceScaleHostDriver{}
			s, err := NewStateCheckResidentState(ctx, d, fixture, 168)
			if err != nil {
				t.Fatal(err)
			}
			legacyDriver := &deviceScaleHostDriver{}
			legacy, err := NewDeviceScaleResidentState(ctx, legacyDriver, fixture, 168)
			if err != nil {
				t.Fatal(err)
			}
			defer legacy.Close()
			for i, limit := range []float32{6.5, .01, 1e20} {
				grad := map[string][]float32{"a": {3, 4}, "b": {12}}
				if err = s.Accumulate(ctx, grad, 1); err != nil {
					t.Fatal(err)
				}
				if err = legacy.Accumulate(ctx, grad, 1); err != nil {
					t.Fatal(err)
				}
				downloads := d.events["download"]
				if _, err = s.Update(ctx, limit); err != nil {
					t.Fatal(err)
				}
				if d.events["download"]-downloads != 2 {
					t.Fatal("expected norm+one state-check diagnostic", d.events)
				}
				if _, err = legacy.Update(ctx, limit); err != nil {
					t.Fatal(err)
				}
				got, e := s.Snapshot(ctx)
				if e != nil {
					t.Fatal(e)
				}
				want, e := legacy.Snapshot(ctx)
				if e != nil {
					t.Fatal(e)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatal("unchanged optimizer arithmetic", i)
				}
				if i == 1 {
					if err = s.Close(); err != nil {
						t.Fatal(err)
					}
					d = &deviceScaleHostDriver{}
					s, err = NewStateCheckResidentState(ctx, d, got, 168)
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			if err = s.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
func TestTrainingStateCheckFailureRetention(t *testing.T) {
	ctx := context.Background()
	// With two parameters gradient norm uses4 merges/2statistics;mutable checks
	// use8 additional merges/statistics and a second zero-summary upload/download.
	cases := []struct {
		name, event string
		at          int
		preflight   bool
	}{
		{"preflight-first", "merge-admit", 1, true}, {"preflight-last", "merge-admit", 8, true},
		{"reset", "upload", 1, false}, {"post-reset", "upload", 2, false},
		{"post-stats", "stats", 3, false}, {"last-stats", "stats", 10, false},
		{"post-merge", "merge", 5, false}, {"last-merge", "merge", 12, false}, {"post-download", "download", 2, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := &deviceScaleHostDriver{}
			d.sentinel = errors.New("injected")
			s, err := NewStateCheckResidentState(ctx, d, residentFixture(), 168)
			if err != nil {
				t.Fatal(err)
			}
			if err = s.Accumulate(ctx, map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
				t.Fatal(err)
			}
			d.fail, d.at = c.event, d.events[c.event]+c.at
			uploads := d.events["upload"]
			if _, err = s.Update(ctx, 1); err == nil || s.StepCount() != 0 {
				t.Fatal("failure/step")
			}
			if c.preflight {
				if s.terminal != nil || d.events["update"] != 0 || d.events["upload"] != uploads {
					t.Fatal("preflight mutated state")
				}
				d.fail = ""
				if _, err = s.Update(ctx, 1); err != nil {
					t.Fatal(err)
				}
				if err = s.Close(); err != nil {
					t.Fatal(err)
				}
			} else {
				if s.terminal == nil || s.Close() == nil {
					t.Fatal("owner not retained")
				}
				before := d.events["download"]
				if _, err = s.Snapshot(ctx); err == nil || d.events["download"] != before {
					t.Fatal("snapshot after uncertainty")
				}
				if _, err = s.Update(ctx, 1); err == nil {
					t.Fatal("retry")
				}
				if c.name != "reset" && d.events["update"] != 2 {
					t.Fatal("failure phase not post-update", d.events)
				}
			}
		})
	}
}

type corruptStateCheckDriver struct {
	deviceScaleHostDriver
	which    int
	negative bool
}

func (d *corruptStateCheckDriver) UpdateScaled(ctx context.Context, p, g, m, v, ema, clip ResidentBuffer, n int, step AdamWStep) error {
	if err := d.deviceScaleHostDriver.UpdateScaled(ctx, p, g, m, v, ema, clip, n, step); err != nil {
		return err
	}
	target := []ResidentBuffer{p, m, v, ema}[d.which].(*residentHostBuffer)
	if d.negative {
		target.values[0] = -1
	} else {
		target.values[0] = float32(math.NaN())
	}
	return nil
}
func TestTrainingStateCheckInvalidMutableState(t *testing.T) {
	for i, name := range []string{"parameter", "M", "V", "EMA"} {
		t.Run(name, func(t *testing.T) {
			for _, ema := range []float32{0, .9} {
				d := &corruptStateCheckDriver{which: i}
				fixture := residentFixture()
				fixture.Config.EMADecay = ema
				s, err := NewStateCheckResidentState(context.Background(), d, fixture, 168)
				if err != nil {
					t.Fatal(err)
				}
				if err = s.Accumulate(context.Background(), map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
					t.Fatal(err)
				}
				if _, err = s.Update(context.Background(), 1); err == nil || s.StepCount() != 0 || s.terminal == nil || s.Close() == nil {
					t.Fatal("invalid mutable state committed", name, ema)
				}
			}
		})
	}
	d := &corruptStateCheckDriver{which: 2, negative: true}
	s, err := NewStateCheckResidentState(context.Background(), d, residentFixture(), 168)
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Accumulate(context.Background(), map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Update(context.Background(), 1); err == nil || s.terminal == nil || s.StepCount() != 0 {
		t.Fatal("negative V committed")
	}
}
func TestTrainingStateCheckAdmission(t *testing.T) {
	for _, d := range []ResidentDriver{&residentHostDriver{}, &globalNormHostDriver{}} {
		if s, err := NewStateCheckResidentState(context.Background(), d, residentFixture(), 168); s != nil || err == nil {
			t.Fatal("capability")
		}
	}
	d := &deviceScaleHostDriver{}
	if s, err := NewStateCheckResidentState(context.Background(), d, residentFixture(), 167); s != nil || err == nil || d.events["new"] != 0 {
		t.Fatal("budget before allocation")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewStateCheckResidentState(ctx, d, residentFixture(), 168); err == nil {
		t.Fatal("cancel")
	}
}

func TestTrainingStateCheckChunkBoundaries(t *testing.T) {
	for _, n := range []int{1, 255, 256, 257, 4097} {
		values := make([]float32, n)
		for i := range values {
			switch i % 4 {
			case 0:
				values[i] = math.MaxFloat32
			case 1:
				values[i] = math.SmallestNonzeroFloat32
			case 2:
				values[i] = -1e30
			}
		}
		stats := hostNormStats(values, false)
		summary := mockMergeNorm([]float32{0, 0, 0, 0}, stats, n, 0, 0)
		if err := DecodeStateCheck(summary); err != nil {
			t.Fatal("finite independent values rejected", n, err)
		}
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			values[n-1] = bad
			stats = hostNormStats(values, false)
			summary = mockMergeNorm([]float32{0, 0, 0, 0}, stats, n, 0, 0)
			if DecodeStateCheck(summary) == nil {
				t.Fatal("invalid last chunk ignored", n)
			}
		}
	}
	// Invalid summary carry must survive subsequent valid tensor merges.
	prior := []float32{1, 0, 1, 0}
	next := mockMergeNorm(prior, hostNormStats([]float32{2, 3}, false), 2, 1, 0)
	if DecodeStateCheck(next) == nil {
		t.Fatal("invalid flag lost across tensor")
	}
}
