package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

// Mock the specified device F32 reduction order. Expected norms/clipped updates
// below use independent F64 hypot/direct clipping, not these helpers as goldens.
func mockMergeNorm(a, b []float32, n int, previous int64, mode uint32) []float32 {
	maximum, sum, invalid, correction := a[0], a[1], a[2], a[3]
	if !finiteF32Compat(a) || invalid != 0 || maximum < 0 || sum < 0 || sum > float32(previous)*(1+2e-6) || (maximum == 0 && sum != 0) {
		invalid = 1
	}
	for i := 0; i < len(b); i += 3 {
		m, s, flag := b[i], b[i+1], b[i+2]
		chunk := min(256, n-(i/3)*256)
		if !finiteF32Compat(b[i:i+3]) || flag != 0 || m < 0 || s < 0 || s > float32(chunk) || (m == 0 && s != 0) || (m > 0 && s < 1) {
			invalid = 1
		}
		if mode == 0 {
			maximum = max(maximum, m)
		} else if m > maximum {
			invalid = 1
		}
	}
	total, comp := sum, correction
	if mode == 0 {
		total, comp = 0, 0
	} else if invalid == 0 && maximum > 0 {
		for i := 0; i < len(b); i += 3 {
			r := b[i] / maximum
			term := (r*r)*b[i+1] - comp
			next := total + term
			comp = (next - total) - term
			total = next
		}
	}
	return []float32{maximum, total, invalid, comp}
}
func finiteF32Compat(a []float32) bool {
	for _, x := range a {
		if !residentFinite(x) {
			return false
		}
	}
	return true
}
func mockGlobalNorm(vectors [][]float32) ([]float32, int64) {
	a := []float32{0, 0, 0, 0}
	var elements int64
	for _, mode := range []uint32{0, 2} {
		elements = 0
		for _, v := range vectors {
			a = mockMergeNorm(a, hostNormStats(v, false), len(v), elements, mode)
			elements += int64(len(v))
		}
	}
	return a, elements
}
func mockClipNorm(a []float32, elements int64, maxNorm float32) []float32 {
	invalid := a[2]
	scale := float32(1)
	if !residentFinite(a[3]) || !residentFinite(a[0]) || !residentFinite(a[1]) || !residentFinite(invalid) || invalid != 0 || a[0] < 0 || a[1] < 0 || a[1] > float32(elements)*(1+2e-6) || (a[0] == 0 && a[1] != 0) || (a[0] > 0 && a[1] < 1) {
		invalid = 1
	}
	if invalid == 0 && a[0] > 0 {
		root := float32(math.Sqrt(float64(a[1])))
		ratio := maxNorm / a[0]
		if ratio < root {
			scale = ratio / root
		}
	}
	if invalid != 0 {
		scale = 0
	}
	return []float32{a[0], a[1], invalid, scale}
}

type globalNormHostDriver struct{ residentHostDriver }

func (d *globalNormHostDriver) ValidateNormMerge(ctx context.Context, dst, a, b ResidentBuffer, n int, previous int64, mode uint32) error {
	if _, err := NormMergeShape(n, previous); err != nil {
		return err
	}
	return d.event("merge-admit")
}
func (d *globalNormHostDriver) ValidateNormClip(ctx context.Context, dst, a ResidentBuffer, elements int64, maxNorm float32) error {
	if err := NormClipShape(elements, maxNorm); err != nil {
		return err
	}
	return d.event("clip-admit")
}
func (d *globalNormHostDriver) NormMerge(ctx context.Context, dst, a, b ResidentBuffer, n int, previous int64, mode uint32) error {
	if err := d.event("merge"); err != nil {
		return err
	}
	copy(dst.(*residentHostBuffer).values, mockMergeNorm(a.(*residentHostBuffer).values, b.(*residentHostBuffer).values, n, previous, mode))
	return ctx.Err()
}
func (d *globalNormHostDriver) NormClip(ctx context.Context, dst, a ResidentBuffer, elements int64, maxNorm float32) error {
	if err := d.event("clip"); err != nil {
		return err
	}
	copy(dst.(*residentHostBuffer).values, mockClipNorm(a.(*residentHostBuffer).values, elements, maxNorm))
	return ctx.Err()
}
func TestTrainingGlobalNormReference(t *testing.T) {
	var maxNormRelative, maxScaleError float64
	for _, magnitude := range []float32{0, 1e-30, 1e-10, 1, 1e10, 1e30} {
		var vectors [][]float32
		var previous int64
		var reference float64
		for _, n := range []int{1, 257, 4097} {
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(i%17-8) * magnitude
				reference = math.Hypot(reference, float64(x[i]))
			}
			vectors = append(vectors, x)
			previous += int64(n)
		}
		accumulator, _ := mockGlobalNorm(vectors)
		for _, ceiling := range []float32{1e-20, .01, 1, 1e20} {
			values := mockClipNorm(accumulator, previous, ceiling)
			norm, scale, err := DecodeGlobalNorm(values, previous)
			if err != nil {
				t.Fatal(err)
			}
			if reference > 0 {
				relative := math.Abs(norm-reference) / reference
				maxNormRelative = math.Max(maxNormRelative, relative)
				if relative > 3e-7 {
					t.Fatal("global F64 norm", magnitude, norm, reference, relative)
				}
			}
			want := 1.0
			if reference > float64(ceiling) {
				want = float64(ceiling) / reference
			}
			e := math.Abs(float64(scale) - want)
			maxScaleError = math.Max(maxScaleError, e)
			if e > 1e-7+3e-7*want {
				t.Fatal("global F64 scale", magnitude, ceiling, scale, want)
			}
		}
	}
	t.Logf("global norm F32/F64 max relative=%g scale absolute=%g; six magnitudes and four ceilings", maxNormRelative, maxScaleError)
	// Max F32 input does not square-overflow; F64 diagnostic can exceed F32.
	x := []float32{math.MaxFloat32, math.MaxFloat32}
	a, _ := mockGlobalNorm([][]float32{x})
	values := mockClipNorm(a, 2, 1)
	norm, scale, err := DecodeGlobalNorm(values, 2)
	if err != nil || norm <= math.MaxFloat32 || scale <= 0 {
		t.Fatal("scaled overflow-safe norm", norm, scale, err)
	}
	for _, bad := range [][]float32{{-1, 1, 0}, {1, 0, 0}, {0, 1, 0}, {1, 257, 0}, {1, 1, 1}, {float32(math.NaN()), 1, 0}} {
		a := mockMergeNorm([]float32{0, 0, 0, 0}, bad, 1, 0, 0)
		if _, _, err := DecodeGlobalNorm(mockClipNorm(a, 1, 1), 1); err == nil {
			t.Fatal("malformed summary admitted", bad)
		}
	}
}
func TestTrainingGlobalNormResidentUpdate(t *testing.T) {
	snapshot := residentFixture()
	d := &globalNormHostDriver{}
	s, err := NewGlobalNormResidentState(context.Background(), d, snapshot, 168)
	if err != nil {
		t.Fatal(err)
	}
	before := d.events["download"]
	grad := map[string][]float32{"a": {3, 4}, "b": {12}}
	if err := s.Accumulate(context.Background(), grad, 1); err != nil {
		t.Fatal(err)
	}
	norm, err := s.Update(context.Background(), 6.5)
	if err != nil || math.Abs(norm-13) > 2e-6 {
		t.Fatal(norm, err)
	}
	// One gradient-boundary transfer instead of two per-parameter transfers;
	// four post-update mutable-state summaries per parameter still download.
	if d.events["download"]-before != 1+4*len(snapshot.Parameters) {
		t.Fatal("global gradient summary downloads", d.events)
	}
	legacy, ld := requireResident(t)
	if err := legacy.Accumulate(context.Background(), grad, .5); err != nil {
		t.Fatal(err)
	}
	if _, err := legacy.Update(context.Background(), 1e20); err != nil {
		t.Fatal(err)
	}
	got, err := s.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want, err := legacy.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for name, p := range got.Parameters {
		w := want.Parameters[name]
		for j, x := range [][]float32{p.Parameter, p.M, p.V, p.EMA} {
			y := [][]float32{w.Parameter, w.M, w.V, w.EMA}[j]
			for i, v := range x {
				if math.Abs(float64(v-y[i])) > 3e-7 {
					t.Fatal("clipped update", name, j, i, v, y[i])
				}
			}
		}
	}
	if d.events["update"] != ld.events["update"] {
		t.Fatal("update coverage")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}
func TestTrainingGlobalNormAdmissionRetention(t *testing.T) {
	if _, err := NewGlobalNormResidentState(context.Background(), &residentHostDriver{}, residentFixture(), 1<<20); err == nil {
		t.Fatal("missing driver")
	}
	if _, err := NewGlobalNormResidentState(context.Background(), &globalNormHostDriver{}, residentFixture(), 167); err == nil {
		t.Fatal("budget")
	}
	for _, n := range []int64{0, 1<<38 + 1} {
		if NormClipShape(n, 1) == nil {
			t.Fatal("element bound")
		}
	}
	if _, err := NormMergeShape(1, 1<<38); err == nil {
		t.Fatal("merge count")
	}
	for _, v := range [][]float32{nil, {1, 1, 1, 1}, {0, 1, 0, 1}, {1, 1, 0, 2}, {1, 1, 0, -1}, {1, 1, 0, float32(math.NaN())}} {
		if _, _, err := DecodeGlobalNorm(v, 1); err == nil {
			t.Fatal("boundary")
		}
	}
	for _, category := range []string{"merge-admit", "clip-admit", "merge", "clip", "download", "stats", "update"} {
		t.Run(category, func(t *testing.T) {
			d := &globalNormHostDriver{}
			s, err := NewGlobalNormResidentState(context.Background(), d, residentFixture(), 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			before, _ := s.Snapshot(context.Background())
			if err := s.Accumulate(context.Background(), map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
				t.Fatal(err)
			}
			d.fail, d.at, d.sentinel = category, d.events[category]+1, errors.New("fail")
			if _, err := s.Update(context.Background(), 1); err == nil {
				t.Fatal("failure admitted")
			}
			if s.step != 0 {
				t.Fatal("committed failed step")
			}
			if category == "merge-admit" || category == "clip-admit" {
				if s.terminal != nil || d.events["update"] != 0 {
					t.Fatal("preflight mutated")
				}
				return
			}
			if s.terminal == nil || s.Close() == nil {
				t.Fatal("uncertain owner freed")
			}
			for i, e := range s.entries {
				if category != "update" {
					if !reflect.DeepEqual(e.parameter.(*residentHostBuffer).values, before.Parameters[e.name].Parameter) {
						t.Fatal("pre-update failure mutated", i)
					}
				}
			}
		})
	}
}
func TestTrainingGlobalNormInvalidGradient(t *testing.T) {
	d := &globalNormHostDriver{}
	s, err := NewGlobalNormResidentState(context.Background(), d, residentFixture(), 168)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Accumulate(context.Background(), map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
		t.Fatal(err)
	}
	s.entries[0].gradient.(*residentHostBuffer).values[0] = float32(math.Inf(1))
	if _, err := s.Update(context.Background(), 1); err == nil || s.terminal == nil || d.events["update"] != 0 || s.step != 0 {
		t.Fatal("invalid gradient passed update")
	}
}
func TestTrainingGlobalNormAllocationRetention(t *testing.T) {
	// Last three allocations own global summaries/scale; partial owners retained.
	for _, at := range []int{19, 20, 21} {
		d := &globalNormHostDriver{}
		d.fail, d.at, d.sentinel = "new", at, errors.New("allocation")
		owner, err := NewGlobalNormResidentState(context.Background(), d, residentFixture(), 168)
		if err == nil || owner == nil || owner.terminal == nil || owner.Close() == nil {
			t.Fatal("allocation owner lost", at, owner, err)
		}
	}
}
func TestTrainingGlobalNormManyParameters(t *testing.T) {
	// Exercise the 4096-parameter inventory limit with changing maxima. Each
	// independent reference uses direct F64 hypot over gradients, not summaries.
	for _, pattern := range []string{"flat", "increasing", "mixed"} {
		var vectors [][]float32
		var elements int64
		var want float64
		for parameter := 0; parameter < 4096; parameter++ {
			n := 1 + parameter%31
			x := make([]float32, n)
			for i := range x {
				switch pattern {
				case "flat":
					x[i] = 1
				case "increasing":
					x[i] = 1 + float32(parameter)*.00013 + float32(i)*.00007
				case "mixed":
					x[i] = float32(math.Ldexp(float64((parameter+i)%17-8), parameter%81-40))
				}
				want = math.Hypot(want, float64(x[i]))
			}
			vectors = append(vectors, x)
			elements += int64(n)
		}
		a, _ := mockGlobalNorm(vectors)
		ceiling := float32(want * .375)
		values := mockClipNorm(a, elements, ceiling)
		norm, scale, err := DecodeGlobalNorm(values, elements)
		if err != nil {
			t.Fatal(pattern, err, a)
		}
		relative := math.Abs(norm-want) / want
		scaleError := math.Abs(float64(scale) - float64(ceiling)/want)
		if relative > 3e-7 || scaleError > 1e-7+3e-7*float64(ceiling)/want {
			t.Fatal("many parameter rounding", pattern, relative, scaleError)
		}
		t.Logf("4096-parameter %s: elements=%d norm relative=%g clipping scale absolute=%g", pattern, elements, relative, scaleError)
	}
}
func TestTrainingGlobalNormSecondPassRetention(t *testing.T) {
	for _, category := range []string{"merge-admit", "merge"} {
		d := &globalNormHostDriver{}
		s, err := NewGlobalNormResidentState(context.Background(), d, residentFixture(), 168)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Accumulate(context.Background(), map[string][]float32{"a": {3, 4}, "b": {12}}, 1); err != nil {
			t.Fatal(err)
		}
		d.fail, d.at, d.sentinel = category, len(s.entries)+1, errors.New("sum-pass")
		if _, err := s.Update(context.Background(), 1); err == nil || s.step != 0 || d.events["update"] != 0 {
			t.Fatal("sum pass failure admitted", category)
		}
		if category == "merge-admit" {
			if s.terminal != nil || d.events["merge"] != 0 {
				t.Fatal("sum preflight dispatched")
			}
		} else if s.terminal == nil || s.Close() == nil {
			t.Fatal("sum failure owner freed")
		}
	}
}
