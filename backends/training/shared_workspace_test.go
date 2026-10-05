package training

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"testing"
)

type sharedHostDriver struct {
	deviceScaleHostDriver
	views []ResidentBuffer
	hook  func(string)
}

type sharedHostPrefix struct {
	driver *sharedHostDriver
	parent *residentHostBuffer
	values []float32
	access *SharedResidentAccess
}

func (b *sharedHostPrefix) Upload(ctx context.Context, x []float32) error {
	if err := b.access.Validate(ctx, b.driver, true); err != nil {
		return err
	}
	if b.parent.closed || len(x) != len(b.values) {
		return errors.New("prefix owner/extent")
	}
	if err := b.driver.event("upload"); err != nil {
		return err
	}
	copy(b.values, x)
	return ctx.Err()
}
func (b *sharedHostPrefix) Download(ctx context.Context, x []float32) error {
	if err := b.access.Validate(ctx, b.driver, false); err != nil {
		return err
	}
	if b.parent.closed || len(x) != len(b.values) {
		return errors.New("prefix owner/extent")
	}
	if err := b.driver.event("download"); err != nil {
		return err
	}
	copy(x, b.values)
	return ctx.Err()
}
func (b *sharedHostPrefix) Close() error { return errors.New("prefix cannot close allocation") }
func (d *sharedHostDriver) Prefix(ctx context.Context, parent ResidentBuffer, n int, access *SharedResidentAccess) (ResidentBuffer, error) {
	if err := access.Validate(ctx, d, false); err != nil {
		return nil, err
	}
	if err := d.event("prefix"); err != nil {
		return nil, err
	}
	b, ok := parent.(*residentHostBuffer)
	if !ok || b.driver != &d.residentHostDriver || b.closed || n <= 0 || n > len(b.values) {
		return nil, errors.New("prefix extent/owner")
	}
	return &sharedHostPrefix{d, b, b.values[:n:n], access}, nil
}
func (d *sharedHostDriver) raw(ctx context.Context, b ResidentBuffer, write bool) (*residentHostBuffer, error) {
	if ctx == nil {
		return nil, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if prefix, ok := b.(*sharedHostPrefix); ok {
		if err := prefix.access.Validate(ctx, d, write); err != nil {
			return nil, err
		}
		if prefix.parent.closed {
			return nil, errors.New("expired prefix parent")
		}
		d.views = append(d.views, prefix)
		return &residentHostBuffer{values: prefix.values, driver: &d.residentHostDriver}, nil
	}
	x, ok := b.(*residentHostBuffer)
	if !ok || x == nil || x.closed || x.driver != &d.residentHostDriver {
		return nil, errors.New("foreign/expired mock allocation")
	}
	return x, nil
}
func sharedHostOverlap(a, b *residentHostBuffer) bool {
	// Same-parent prefix wrappers can differ; compare actual element addresses.
	for i := range a.values {
		for j := range b.values {
			if &a.values[i] == &b.values[j] {
				return true
			}
		}
	}
	return false
}
func (d *sharedHostDriver) triple(ctx context.Context, out, a, b ResidentBuffer, n int) (*residentHostBuffer, *residentHostBuffer, *residentHostBuffer, error) {
	x, err := d.raw(ctx, out, true)
	if err != nil {
		return nil, nil, nil, err
	}
	y, err := d.raw(ctx, a, false)
	if err != nil {
		return nil, nil, nil, err
	}
	z, err := d.raw(ctx, b, false)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(x.values) != n || len(y.values) != n || len(z.values) != n || sharedHostOverlap(x, y) || sharedHostOverlap(x, z) || sharedHostOverlap(y, z) {
		return nil, nil, nil, errors.New("mock range alias/extent")
	}
	return x, y, z, nil
}
func (d *sharedHostDriver) ValidateSharedAccumulate(ctx context.Context, o, a, b ResidentBuffer, n int, scale float32) error {
	if !residentFinite(scale) {
		return errors.New("scale")
	}
	if _, _, _, err := d.triple(ctx, o, a, b, n); err != nil {
		return err
	}
	return d.event("shared-admit")
}
func (d *sharedHostDriver) Accumulate(ctx context.Context, o, a, b ResidentBuffer, n int, scale float32) error {
	x, y, z, err := d.triple(ctx, o, a, b, n)
	if err != nil {
		return err
	}
	if d.hook != nil {
		d.hook("accumulate")
	}
	if err := d.event("accumulate"); err != nil {
		return err
	}
	for i := range x.values {
		x.values[i] = y.values[i] + z.values[i]*scale
	}
	return ctx.Err()
}
func (d *sharedHostDriver) ValidateCopy(ctx context.Context, o, a ResidentBuffer, n int) error {
	x, err := d.raw(ctx, o, true)
	if err != nil {
		return err
	}
	y, err := d.raw(ctx, a, false)
	if err != nil {
		return err
	}
	if len(x.values) != n || len(y.values) != n || sharedHostOverlap(x, y) {
		return errors.New("copy extent/alias")
	}
	return d.event("copy-admit")
}
func (d *sharedHostDriver) Copy(ctx context.Context, o, a ResidentBuffer, n int) error {
	if err := d.ValidateCopy(ctx, o, a, n); err != nil {
		return err
	}
	x, _ := d.raw(ctx, o, true)
	y, _ := d.raw(ctx, a, false)
	if d.hook != nil {
		d.hook("copy")
	}
	if err := d.event("copy"); err != nil {
		if n > 0 {
			x.values[0] = y.values[0]
		}
		return err
	}
	copy(x.values, y.values)
	return ctx.Err()
}
func (d *sharedHostDriver) ValidateNormStats(ctx context.Context, o, a ResidentBuffer, n int, positive bool) error {
	x, err := d.raw(ctx, o, true)
	if err != nil {
		return err
	}
	y, err := d.raw(ctx, a, false)
	if err != nil {
		return err
	}
	count, err := NormStatsShape(n)
	if err != nil || len(x.values) != count || len(y.values) != n || sharedHostOverlap(x, y) {
		return errors.New("stats range")
	}
	return d.event("stats-admit")
}
func (d *sharedHostDriver) NormStats(ctx context.Context, o, a ResidentBuffer, n int, positive bool) error {
	if err := d.ValidateNormStats(ctx, o, a, n, positive); err != nil {
		return err
	}
	x, _ := d.raw(ctx, o, true)
	y, _ := d.raw(ctx, a, false)
	if d.hook != nil {
		d.hook("stats")
	}
	return d.residentHostDriver.NormStats(ctx, x, y, n, positive)
}
func (d *sharedHostDriver) ValidateNormMerge(ctx context.Context, o, a, stats ResidentBuffer, n int, previous int64, mode uint32) error {
	x, err := d.raw(ctx, o, true)
	if err != nil {
		return err
	}
	y, err := d.raw(ctx, a, false)
	if err != nil {
		return err
	}
	z, err := d.raw(ctx, stats, false)
	if err != nil {
		return err
	}
	return d.globalNormHostDriver.ValidateNormMerge(ctx, x, y, z, n, previous, mode)
}
func (d *sharedHostDriver) NormMerge(ctx context.Context, o, a, stats ResidentBuffer, n int, previous int64, mode uint32) error {
	x, err := d.raw(ctx, o, true)
	if err != nil {
		return err
	}
	y, err := d.raw(ctx, a, false)
	if err != nil {
		return err
	}
	z, err := d.raw(ctx, stats, false)
	if err != nil {
		return err
	}
	return d.globalNormHostDriver.NormMerge(ctx, x, y, z, n, previous, mode)
}
func sharedFixture(count int) ResidentSnapshot {
	s := residentFixture()
	s.Parameters = map[string]ResidentParameter{}
	for i := 0; i < count; i++ {
		n := []int{513, 257, 3, 1}[i%4]
		p := make([]float32, n)
		for j := range p {
			p[j] = float32(j%7) * .01
		}
		s.Parameters[fmt.Sprint(i)] = ResidentParameter{Parameter: p, M: make([]float32, n), V: make([]float32, n), EMA: append([]float32(nil), p...)}
	}
	return s
}
func requireShared(t *testing.T, fixture ResidentSnapshot) (*ResidentState, *sharedHostDriver) {
	t.Helper()
	d := &sharedHostDriver{}
	d.sentinel = errors.New("shared uncertain")
	p, err := SharedResidentBytes(fixture)
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewSharedWorkspaceResidentState(context.Background(), d, fixture, p.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	return s, d
}
func TestTrainingSharedWorkspacePlan(t *testing.T) {
	f := residentFixture()
	p, err := SharedResidentBytes(f)
	if err != nil || p.Bytes != 144 || p.Elements != 3 || p.MaxElements != 2 || p.MaxStats != 3 {
		t.Fatal("plan", p, err)
	}
	d := &sharedHostDriver{}
	if s, err := NewSharedWorkspaceResidentState(context.Background(), d, f, p.Bytes-1); err == nil || s != nil || len(d.buffers) != 0 {
		t.Fatal("underbudget")
	}
	s, d := requireShared(t, f)
	if len(s.owned) != 5*len(f.Parameters)+7 {
		t.Fatal("hidden allocation")
	}
	var bytes int64
	for _, b := range d.buffers {
		bytes += 4 * int64(len(b.values))
	}
	if bytes != p.Bytes {
		t.Fatal("allocation inventory", bytes, p.Bytes)
	}
	if s.entries[0].scratch != nil || s.entries[0].stats != nil {
		t.Fatal("perparam workspace allocated")
	}
	if s.Close() != nil {
		t.Fatal("close")
	}
	if s, err := NewSharedWorkspaceResidentState(context.Background(), &deviceScaleHostDriver{}, f, p.Bytes); err == nil || s != nil {
		t.Fatal("unsupported allocated")
	}
	bad := residentFixture()
	bad.Parameters["b"] = ResidentParameter{Parameter: []float32{1}, M: []float32{0}, V: []float32{-1}, EMA: []float32{1}}
	if _, err := SharedResidentBytes(bad); err == nil {
		t.Fatal("invalid snapshot")
	}
}
func TestTrainingSharedWorkspaceNumericalUpdateResume(t *testing.T) {
	ctx := context.Background()
	var comparisons, sum, square, maximum float64
	for _, count := range []int{1, 2, 3, 4} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			fixture := sharedFixture(count)
			s, d := requireShared(t, fixture)
			legacyDriver := &deviceScaleHostDriver{}
			legacy, err := NewStateCheckResidentState(ctx, legacyDriver, fixture, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			stable := map[string]ResidentBuffer{}
			for _, e := range s.entries {
				stable[e.name] = e.gradient
			}
			for step := 0; step < 3; step++ {
				branches := map[string][]float32{}
				for i, e := range s.entries {
					x := make([]float32, e.n)
					for j := range x {
						x[j] = float32(i+1) * float32(j%13-6) * float32(step+1) * .07
					}
					branches[e.name] = x
				}
				for _, scale := range []float32{0, .25, 1} {
					if err = s.Accumulate(ctx, branches, scale); err != nil {
						t.Fatal(err)
					}
					if err = legacy.Accumulate(ctx, branches, scale); err != nil {
						t.Fatal(err)
					}
				}
				for _, e := range s.entries {
					if e.gradient != stable[e.name] {
						t.Fatal("gradient identity swapped")
					}
				}
				norm, err := s.Update(ctx, []float32{.01, .1, 1e20}[step])
				if err != nil {
					t.Fatal(err)
				}
				wantNorm, err := legacy.Update(ctx, []float32{.01, .1, 1e20}[step])
				if err != nil {
					t.Fatal(err)
				}
				if math.Abs(norm-wantNorm) > 3e-7*max(1., wantNorm) {
					t.Fatal("norm order", norm, wantNorm)
				}
				got, err := s.Snapshot(ctx)
				if err != nil {
					t.Fatal(err)
				}
				want, err := legacy.Snapshot(ctx)
				if err != nil {
					t.Fatal(err)
				}
				for name, p := range got.Parameters {
					q := want.Parameters[name]
					for i, a := range [][]float32{p.Parameter, p.M, p.V, p.EMA} {
						b := [][]float32{q.Parameter, q.M, q.V, q.EMA}[i]
						for j, v := range a {
							delta := math.Abs(float64(v) - float64(b[j]))
							if delta > 2e-6 {
								t.Fatal("state difference", delta)
							}
							comparisons++
							sum += delta
							square += delta * delta
							maximum = math.Max(maximum, delta)
						}
					}
				}
				if step == 1 {
					if s.Close() != nil {
						t.Fatal("close")
					}
					s, d = requireShared(t, got)
					stable = map[string]ResidentBuffer{}
					for _, e := range s.entries {
						stable[e.name] = e.gradient
					}
				}
			}
			if d.events["stats-admit"] == 0 || s.Close() != nil || legacy.Close() != nil {
				t.Fatal("validation/close")
			}
		})
	}
	t.Logf("shared versus established host schedule comparisons%.0f mean_abs%.9g RMS%.9g max_abs%.9g unchanged2e-6", comparisons, sum/comparisons, math.Sqrt(square/comparisons), maximum)
}
func TestTrainingSharedWorkspaceViews(t *testing.T) {
	ctx := context.Background()
	s, d := requireShared(t, residentFixture())
	w := s.shared
	if err := w.begin(ctx); err != nil {
		t.Fatal(err)
	}
	opctx := w.ctx
	v, err := w.view(w.scratch, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	z, err := w.view(w.zero, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if z.Upload(opctx, []float32{1}) == nil || v.Close() == nil {
		t.Fatal("view permission")
	}
	if _, err := (&sharedHostDriver{}).raw(opctx, v, false); err == nil {
		t.Fatal("foreign driver")
	}
	if _, err := d.raw(context.WithValue(ctx, 0, 0), v, false); err == nil {
		t.Fatal("foreign context")
	}
	if w.begin(ctx) == nil || s.Close() == nil || s.Accumulate(ctx, map[string][]float32{"a": {1, 2}}, 1) == nil {
		t.Fatal("workspace reentry")
	}
	if err := v.Upload(opctx, []float32{7}); err != nil {
		t.Fatal(err)
	}
	if s.shared.scratch.(*residentHostBuffer).values[1] != 0 {
		t.Fatal("prefix tail changed")
	}
	alias, _ := w.view(w.scratch, 1, true)
	if d.ValidateCopy(opctx, alias, v, 1) == nil {
		t.Fatal("same allocation wrappers accepted")
	}
	w.end()
	if v.Upload(opctx, []float32{0}) == nil || v.Download(opctx, make([]float32, 1)) == nil {
		t.Fatal("escaped generation")
	}
	if w.begin(ctx) != nil {
		t.Fatal("next generation")
	}
	if _, err := d.raw(opctx, v, true); err == nil {
		t.Fatal("stale sameowner generation")
	}
	w.end()
	if s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingSharedWorkspaceFailures(t *testing.T) {
	for _, event := range []string{"new", "upload"} {
		d := &sharedHostDriver{}
		d.fail = event
		d.at = 2
		d.sentinel = errors.New("partial")
		s, err := NewSharedWorkspaceResidentState(context.Background(), d, residentFixture(), 1024)
		if err == nil || s == nil || s.Close() == nil {
			t.Fatal("constructionretain", event)
		}
	}
	for _, event := range []string{"shared-admit", "copy-admit", "upload", "accumulate", "copy", "stats-admit", "merge-admit", "clip-admit", "scaled-admit", "stats", "merge", "clip", "scaled-update", "update", "download"} {
		t.Run(event, func(t *testing.T) {
			ctx := context.Background()
			s, d := requireShared(t, residentFixture())
			d.fail = event
			d.at = d.events[event] + 1
			branches := map[string][]float32{"a": {1, 2}, "b": {3}}
			err := s.Accumulate(ctx, branches, 1)
			if err == nil {
				_, err = s.Update(ctx, .1)
			}
			if err == nil {
				t.Fatal("injected accepted")
			}
			pure := event == "shared-admit" || event == "copy-admit" || event == "stats-admit" || event == "merge-admit" || event == "clip-admit" || event == "scaled-admit"
			if pure {
				if s.terminal != nil || d.events["update"] != 0 {
					t.Fatal("purepreflight mutated")
				}
				d.fail = ""
				return
			}
			if s.terminal == nil {
				t.Fatal("execution not terminal")
			}
			before := d.events["copy"]
			if s.Accumulate(ctx, branches, 1) == nil {
				t.Fatal("retry")
			}
			if _, e := s.Update(ctx, .1); e == nil {
				t.Fatal("update")
			}
			if _, e := s.Snapshot(ctx); e == nil {
				t.Fatal("snapshot")
			}
			if s.Close() == nil || d.events["copy"] != before {
				t.Fatal("release/retry")
			}
		})
	}
}
func TestTrainingSharedWorkspaceCompletePreflight(t *testing.T) {
	ctx := context.Background()
	s, d := requireShared(t, sharedFixture(3))
	branches := map[string][]float32{}
	for _, e := range s.entries {
		branches[e.name] = make([]float32, e.n)
	}
	d.fail = "copy-admit"
	d.at = 3
	before := d.events["upload"]
	if s.Accumulate(ctx, branches, 1) == nil || d.events["upload"] != before || d.events["accumulate"] != 0 {
		t.Fatal("late accumulation preflight wrote")
	}
	d.fail = ""
	if err := s.Accumulate(ctx, branches, 1); err != nil {
		t.Fatal(err)
	}
	// Four P/M/V/EMA checks per entry admitted before BOTH gradient-stat passes.
	d.fail = "stats-admit"
	d.at = d.events["stats-admit"] + 4*len(s.entries)
	if _, err := s.Update(ctx, .1); err == nil || d.events["update"] != 0 || d.events["stats"] != 0 {
		t.Fatal("statecheck late preflight mutated")
	}
}

// Independent F64 reference to global norm, including odd/even max ping-pong.
func TestTrainingSharedWorkspaceGlobalNormF64(t *testing.T) {
	for _, count := range []int{1, 2, 3, 4} {
		s, d := requireShared(t, sharedFixture(count))
		branches := map[string][]float32{}
		var reference float64
		for i, e := range s.entries {
			x := make([]float32, e.n)
			for j := range x {
				x[j] = float32(i+1) * float32(j%11+1) * .023
				reference = math.Hypot(reference, float64(x[j]))
			}
			branches[e.name] = x
		}
		if err := s.Accumulate(context.Background(), branches, 1); err != nil {
			t.Fatal(err)
		}
		got, err := s.Update(context.Background(), .1)
		if err != nil {
			t.Fatal(err)
		}
		rel := math.Abs(got-reference) / reference
		if rel > 3e-7 {
			t.Fatal("F64norm", rel)
		}
		if d.events["stats"] != 6*count {
			t.Fatal("statsrecompute bothpasses/statechecks", d.events["stats"], count)
		}
		t.Logf("params%d independentF64norm%.9g observed%.9g relative_error%.9g unchanged3e-7", count, reference, got, rel)
	}
}

func (d *sharedHostDriver) ValidateSharedInventory(ctx context.Context, buffers []ResidentBuffer, extents []int) error {
	if len(buffers) != len(extents) || len(buffers) == 0 {
		return errors.New("inventorycount")
	}
	for i, b := range buffers {
		x, err := d.raw(ctx, b, true)
		if err != nil || len(x.values) != extents[i] {
			return errors.New("inventoryextent")
		}
		for _, previous := range buffers[:i] {
			y, _ := d.raw(ctx, previous, true)
			if sharedHostOverlap(x, y) {
				return errors.New("inventoryphysicalalias")
			}
		}
	}
	return d.event("inventory-admit")
}

func TestTrainingSharedWorkspaceInventoryPreflight(t *testing.T) {
	d := &sharedHostDriver{}
	d.fail = "inventory-admit"
	d.at = 1
	d.sentinel = errors.New("inventory")
	s, err := NewSharedWorkspaceResidentState(context.Background(), d, residentFixture(), 1024)
	if err == nil || s == nil || d.events["upload"] != 0 || s.Close() == nil {
		t.Fatal("inventory preflight upload/release")
	}
	// Same allocation under distinct wrappers must reject before uploads.
	d = &sharedHostDriver{}
	b, _ := d.NewBuffer(context.Background(), 2)
	x := b.(*residentHostBuffer)
	alias := &residentHostBuffer{values: x.values, driver: &d.residentHostDriver}
	if d.ValidateSharedInventory(context.Background(), []ResidentBuffer{x, alias}, []int{2, 2}) == nil {
		t.Fatal("physical alias inventory")
	}
}
func TestTrainingSharedWorkspaceMissingGradientAndSnapshot(t *testing.T) {
	ctx := context.Background()
	s, _ := requireShared(t, residentFixture())
	if err := s.Accumulate(ctx, map[string][]float32{"a": {1, 2}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Update(ctx, .1); err == nil {
		t.Fatal("unused parameter silentlyready")
	}
	if _, err := s.Snapshot(ctx); err == nil {
		t.Fatal("dirtysnapshot")
	}
	if err := s.Accumulate(ctx, map[string][]float32{"b": {0}}, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Update(ctx, .1); err != nil {
		t.Fatal(err)
	}
	snap, err := s.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	s.entries[0].parameter.(*residentHostBuffer).values[0] = 99
	if snap.Parameters["a"].Parameter[0] == 99 {
		t.Fatal("snapshotborrowed")
	}
}

func TestTrainingSharedWorkspaceGuardedPrefixCannotEscape(t *testing.T) {
	ctx := context.Background()
	s, d := requireShared(t, residentFixture())
	w := s.shared
	if err := w.begin(ctx); err != nil {
		t.Fatal(err)
	}
	opctx := w.ctx
	zero, err := w.view(w.zero, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch, err := w.view(w.scratch, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	escaped := zero // Returned interface itself remains guarded; no raw unwrap API.
	if _, err := d.raw(opctx, escaped, false); err != nil {
		t.Fatal(err)
	}
	if escaped.Upload(opctx, []float32{7}) == nil {
		t.Fatal("readonlyafterread")
	}
	v := make([]float32, 1)
	if zero.Download(opctx, v) != nil || v[0] != 0 {
		t.Fatal("zero changed")
	}
	w.end()
	if escaped.Download(opctx, v) == nil || scratch.Upload(opctx, []float32{8}) == nil {
		t.Fatal("retainedinterface stale")
	}
	if err = w.begin(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err = d.raw(opctx, scratch, true); err == nil {
		t.Fatal("raw driver staleprefix")
	}
	w.end()
	if s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingSharedWorkspaceFullPreflightStatsBothPasses(t *testing.T) {
	ctx := context.Background()
	s, d := requireShared(t, sharedFixture(3))
	branches := map[string][]float32{}
	for _, e := range s.entries {
		branches[e.name] = make([]float32, e.n)
	}
	if err := s.Accumulate(ctx, branches, 1); err != nil {
		t.Fatal(err)
	}
	// Four state-check slots per entry first, then max pass, then fixed-max pass.
	d.fail = "stats-admit"
	d.at = d.events["stats-admit"] + 6*len(s.entries)
	if _, err := s.Update(ctx, .1); err == nil || d.events["update"] != 0 || d.events["stats"] != 0 {
		t.Fatal("second gradient-stat pass preflight after mutation")
	}
}

func TestTrainingSharedWorkspaceExactOperationContext(t *testing.T) {
	ctx := context.Background()
	s, d := requireShared(t, residentFixture())
	w := s.shared
	if err := w.begin(ctx); err != nil {
		t.Fatal(err)
	}
	v, err := w.view(w.scratch, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	original := w.ctx.(*sharedOperationContext)
	copied := *original
	if v.Upload(ctx, []float32{1}) == nil || v.Upload(&copied, []float32{1}) == nil {
		t.Fatal("equal copied context accepted")
	}
	if err = v.Upload(original, []float32{1}); err != nil {
		t.Fatal(err)
	}
	if _, err = d.raw(&copied, v, false); err == nil {
		t.Fatal("driver copied ctx accepted")
	}
	w.end()
}

func TestTrainingSharedWorkspacePyTorchUpdate(t *testing.T) {
	data, err := os.ReadFile("../../model/pockettts/testdata/training_step_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Schema     int                  `json:"schema"`
		Revision   string               `json:"upstream_revision"`
		Parameters map[string][]float32 `json:"parameters"`
		Gradients  map[string][]float32 `json:"gradients"`
		Adam       struct {
			LR       float32              `json:"learning_rate"`
			B1       float32              `json:"beta1"`
			B2       float32              `json:"beta2"`
			Eps      float32              `json:"epsilon"`
			WD       float32              `json:"weight_decay"`
			EMA      float32              `json:"ema_decay"`
			After    map[string][]float32 `json:"parameters_after_step"`
			EMAAfter map[string][]float32 `json:"ema_after_step"`
		} `json:"adamw"`
	}
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.Revision != "0acce6b2f390150267557770d2098c5caa9a18ac" {
		t.Fatal("oracle provenance")
	}
	a := oracle.Adam
	snapshot := ResidentSnapshot{Config: ResidentConfig{a.LR, a.B1, a.B2, a.Eps, a.WD, a.EMA}, Parameters: map[string]ResidentParameter{}}
	for name, p := range oracle.Parameters {
		if _, ok := oracle.Gradients[name]; !ok {
			continue
		}
		snapshot.Parameters[name] = ResidentParameter{Parameter: p, M: make([]float32, len(p)), V: make([]float32, len(p)), EMA: append([]float32(nil), p...)}
	}
	s, _ := requireShared(t, snapshot)
	ctx := context.Background()
	if err = s.Accumulate(ctx, oracle.Gradients, 1); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Update(ctx, math.MaxFloat32); err != nil {
		t.Fatal(err)
	}
	got, err := s.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var n, sum, squares, maximum float64
	for name, p := range got.Parameters {
		for i, v := range p.Parameter {
			e := math.Abs(float64(v) - float64(a.After[name][i]))
			if e > 4e-6 {
				t.Fatal("PyTorch parameter", name, e)
			}
			n++
			sum += e
			squares += e * e
			maximum = math.Max(maximum, e)
		}
		for i, v := range p.EMA {
			e := math.Abs(float64(v) - float64(a.EMAAfter[name][i]))
			if e > 4e-6 {
				t.Fatal("PyTorchEMA", name, e)
			}
			n++
			sum += e
			squares += e * e
			maximum = math.Max(maximum, e)
		}
	}
	if s.Close() != nil {
		t.Fatal("close")
	}
	t.Logf("independent pinned PyTorch parameter/EMA comparisons%.0f mean_abs%.9g RMS%.9g max_abs%.9g unchanged4e-6", n, sum/n, math.Sqrt(squares/n), maximum)
}
