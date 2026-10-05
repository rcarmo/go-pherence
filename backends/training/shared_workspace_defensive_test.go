package training

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"
)

// Restrict the method set without inventing a native implementation.
type sharedNoScale struct{ ResidentSharedWorkspaceDriver }
type sharedNoGlobal struct {
	ResidentSharedWorkspaceDriver
	ResidentDeviceScaleDriver
}
type sharedNilResult struct {
	ResidentSharedWorkspaceDriver
	ResidentDeviceScaleDriver
	ResidentGlobalNormDriver
	allocations   int
	nilAllocation int
	nilPrefix     bool
}

func (d *sharedNilResult) NewBuffer(ctx context.Context, n int) (ResidentBuffer, error) {
	d.allocations++
	if d.allocations == d.nilAllocation {
		return nil, nil
	}
	return d.ResidentSharedWorkspaceDriver.NewBuffer(ctx, n)
}
func (d *sharedNilResult) Prefix(context.Context, ResidentBuffer, int, *SharedResidentAccess) (ResidentBuffer, error) {
	if d.nilPrefix {
		return nil, nil
	}
	return nil, errors.New("unexpected prefix")
}

func TestTrainingSharedWorkspaceDefensiveConstructor(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []string{"nil-context", "nil-driver", "typed-nil", "no-scale", "no-global", "cancel", "zero-ceiling", "invalid-snapshot"} {
		t.Run(kind, func(t *testing.T) {
			d := &sharedHostDriver{}
			var driver ResidentDriver = d
			c := ctx
			f := residentFixture()
			limit := int64(1024)
			switch kind {
			case "nil-context":
				c = nil
			case "nil-driver":
				driver = nil
			case "typed-nil":
				driver = (*sharedHostDriver)(nil)
			case "no-scale":
				driver = &sharedNoScale{d}
			case "no-global":
				driver = &sharedNoGlobal{d, d}
			case "cancel":
				x, cancel := context.WithCancel(ctx)
				cancel()
				c = x
			case "zero-ceiling":
				limit = 0
			case "invalid-snapshot":
				f.Parameters["a"] = ResidentParameter{}
			}
			s, err := NewSharedWorkspaceResidentState(c, driver, f, limit)
			if err == nil || s != nil || len(d.buffers) != 0 {
				t.Fatal("preallocation rejection", s, err, len(d.buffers))
			}
		})
	}
	for _, at := range []int{1, 7, 17} {
		t.Run(fmt.Sprintf("nil-buffer-%d", at), func(t *testing.T) {
			host := &sharedHostDriver{}
			d := &sharedNilResult{ResidentSharedWorkspaceDriver: host, ResidentDeviceScaleDriver: host, ResidentGlobalNormDriver: host, nilAllocation: at}
			s, err := NewSharedWorkspaceResidentState(ctx, d, residentFixture(), 1024)
			if err == nil || s == nil || len(s.owned) != at-1 || host.events["upload"] != 0 || s.terminal == nil || s.Close() == nil {
				t.Fatal("nil allocation retention", at, err)
			}
		})
	}
}
func TestTrainingSharedWorkspaceDefensivePrefixAndAccess(t *testing.T) {
	ctx := context.Background()
	s, d := requireShared(t, residentFixture())
	w := s.shared
	if _, err := w.view(w.scratch, 1, true); err == nil {
		t.Fatal("inactive prefix")
	}
	c, cancel := context.WithCancel(ctx)
	cancel()
	if w.begin(c) == nil {
		t.Fatal("cancel begin")
	}
	w.generation = math.MaxUint64
	if w.begin(ctx) == nil || w.active {
		t.Fatal("generation wrapped")
	}
	w.generation = 0
	if err := w.begin(ctx); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		b     ResidentBuffer
		n     int
		write bool
	}{{w.scratch, 0, true}, {w.zero, 1, true}, {w.scratch, 3, true}, {nil, 1, false}} {
		if _, err := w.view(tc.b, tc.n, tc.write); err == nil {
			t.Fatal("prefix admitted", tc)
		}
	}
	token := &SharedResidentAccess{workspace: w, generation: w.generation, writable: true}
	for _, a := range []*SharedResidentAccess{nil, {}} {
		if a.Validate(w.ctx, d, false) == nil {
			t.Fatal("nil access")
		}
	}
	if token.Validate(nil, d, false) == nil || token.Validate(w.ctx, nil, false) == nil || token.Validate(w.ctx, (*sharedHostDriver)(nil), false) == nil {
		t.Fatal("nil identity")
	}
	original := w.driver
	w.driver = &sharedNilResult{nilPrefix: true}
	if _, err := w.view(w.scratch, 1, true); err == nil {
		t.Fatal("nil prefix")
	}
	w.driver = original
	v, err := w.view(w.scratch, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	parent := w.scratch.(*residentHostBuffer)
	parent.closed = true
	if v.Upload(w.ctx, []float32{1}) == nil || v.Download(w.ctx, make([]float32, 1)) == nil {
		t.Fatal("dead parent")
	}
	parent.closed = false
	s.terminal = errors.New("terminal")
	if token.Validate(w.ctx, d, false) == nil {
		t.Fatal("terminal access")
	}
	s.terminal = nil
	s.closed = true
	if token.Validate(w.ctx, d, false) == nil {
		t.Fatal("closed access")
	}
	s.closed = false
	w.end()
	c, cancel = context.WithCancel(ctx)
	if w.begin(c) != nil {
		t.Fatal("begin")
	}
	v, err = w.view(w.branch, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	opctx := w.ctx
	cancel()
	if v.Upload(opctx, []float32{1}) == nil || v.Download(opctx, make([]float32, 1)) == nil {
		t.Fatal("cancel access")
	}
	w.end()
	if s.Close() != nil {
		t.Fatal("close")
	}
}
func assertSharedTerminalFences(t *testing.T, s *ResidentState, d *sharedHostDriver) {
	t.Helper()
	ctx := context.Background()
	copies, updates, downloads := d.events["copy"], d.events["update"], d.events["download"]
	if s.terminal == nil || s.Accumulate(ctx, map[string][]float32{"a": {1, 2}}, 1) == nil {
		t.Fatal("terminal/retry")
	}
	if _, err := s.Update(ctx, 1); err == nil {
		t.Fatal("terminal update")
	}
	if _, err := s.Snapshot(ctx); err == nil {
		t.Fatal("terminal snapshot")
	}
	if s.Close() == nil {
		t.Fatal("terminal close")
	}
	if copies != d.events["copy"] || updates != d.events["update"] || downloads != d.events["download"] {
		t.Fatal("terminal dispatch")
	}
	for _, b := range d.buffers {
		if b.closed {
			t.Fatal("owner freed")
		}
	}
}
func sharedGradientContentsError(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("length %d want %d", len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("nonfinite[%d]=%v", i, v)
		}
		if math.Abs(float64(v)-float64(want[i])) > 2e-6 {
			return fmt.Errorf("value[%d]=%v want %v", i, v, want[i])
		}
	}
	return nil
}
func TestTrainingSharedWorkspaceDefensiveLaterCopy(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  []float32
		want []float32
	}{{"length", []float32{1}, []float32{1, 2}}, {"nonfinite", []float32{float32(math.NaN())}, []float32{0}}} {
		if err := sharedGradientContentsError(tc.got, tc.want); err == nil {
			t.Fatal("later copy predicate admitted invalid gradient", tc.name)
		}
	}
	for _, tapeMode := range []bool{false, true} {
		t.Run(map[bool]string{false: "host", true: "compact-tape"}[tapeMode], func(t *testing.T) {
			ctx := context.Background()
			f := residentFixture()
			if tapeMode {
				f = sharedCompactFixture()
			}
			s, d := requireShared(t, f)
			d.fail = "copy"
			d.at = 2
			stable := []ResidentBuffer{s.entries[0].gradient, s.entries[1].gradient}
			if tapeMode {
				s.entries[1].gradient.(*residentHostBuffer).values[1] = -13
			}
			var err error
			if tapeMode {
				tape, b, c := compactBindingFixture()
				_, err = s.RunCompactZeroPrunedTape(ctx, tape, b, c, []int{6}, true, 1, 4096)
			} else {
				err = s.Accumulate(ctx, map[string][]float32{"a": {2, 4}, "b": {6}}, 1)
			}
			if err == nil || !s.entries[0].ready || s.entries[1].ready || !s.dirty || d.events["copy"] != 2 {
				t.Fatal("later copy boundary", err)
			}
			for i, e := range s.entries {
				if e.gradient != stable[i] {
					t.Fatal("G swapped")
				}
			}
			// Independent fixture arithmetic: compact token 0 sums row1=(3,4),
			// token2 sums rows0+2=(6,8); dense b=(2,3)+(1,2)=(3,5).
			// Fault copy writes only its first element, leaving the second sentinel.
			want := [][]float32{{2, 4}, {6}}
			if tapeMode {
				want = [][]float32{{3, 4, 0, 0, 6, 8, 0, 0}, {3, -13}}
			}
			for i, e := range s.entries {
				values := e.gradient.(*residentHostBuffer).values
				if err := sharedGradientContentsError(values, want[i]); err != nil {
					t.Fatal("committed/partial gradient contents", i, err)
				}
			}
			if tapeMode && len(s.retained) == 0 {
				t.Fatal("source unretained")
			}
			assertSharedTerminalFences(t, s, d)
		})
	}
}
func TestTrainingSharedWorkspaceDefensiveLatePreflight(t *testing.T) {
	for _, kind := range []string{"host-dense", "tape-dense", "tape-compact"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			f := residentFixture()
			if kind != "host-dense" {
				f = sharedCompactFixture()
			}
			if kind == "tape-compact" {
				f.Parameters["z"] = f.Parameters["a"]
				delete(f.Parameters, "a")
			}
			s, d := requireShared(t, f)
			uploads := d.events["upload"]
			d.fail = "shared-admit"
			d.at = 2
			var err error
			if kind == "host-dense" {
				err = s.Accumulate(ctx, map[string][]float32{"a": {1, 2}, "b": {3}}, 1)
			} else {
				tape, b, c := compactBindingFixture()
				if kind == "tape-compact" {
					b["z"] = b["a"]
					delete(b, "a")
					c["z"] = c["a"]
					delete(c, "a")
					d.fail = "compact-admit"
					d.at = 1
				} else {
					d.at = 1
				}
				_, err = s.RunCompactZeroPrunedTape(ctx, tape, b, c, []int{6}, true, 1, 4096)
			}
			if err == nil || d.events["upload"] != uploads || d.events["accumulate"] != 0 || d.events["compact"] != 0 || d.events["copy"] != 0 || s.dirty {
				t.Fatal("late preflight mutated", kind, err)
			}
			for _, e := range s.entries {
				if e.ready {
					t.Fatal("preflight readiness")
				}
			}
			if kind == "host-dense" {
				if s.terminal != nil {
					t.Fatal("pure rejection poisoned")
				}
				d.fail = ""
				if s.Close() != nil {
					t.Fatal("close")
				}
			} else {
				if d.events["tape-run"] != 1 || len(s.retained) == 0 {
					t.Fatal("executed source retention")
				}
				assertSharedTerminalFences(t, s, d)
			}
		})
	}
}

type sharedCallbackBuffer struct {
	ResidentBuffer
	callback func(string)
}

func (b *sharedCallbackBuffer) Download(ctx context.Context, v []float32) error {
	b.callback("download")
	return b.ResidentBuffer.Download(ctx, v)
}
func (b *sharedCallbackBuffer) Close() error { b.callback("close"); return b.ResidentBuffer.Close() }
func TestTrainingSharedWorkspaceDefensiveSnapshotCloseReentry(t *testing.T) {
	for _, kind := range []string{"snapshot-reentry", "close-reentry", "snapshot-failure", "close-failure"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d := requireShared(t, residentFixture())
			reentries := 0
			cb := func(string) {
				reentries++
				if s.Accumulate(ctx, map[string][]float32{"a": {1, 2}}, 1) == nil || s.Close() == nil {
					t.Fatal("reentry mutation")
				}
				if _, err := s.Update(ctx, 1); err == nil {
					t.Fatal("reentry update")
				}
				if _, err := s.Snapshot(ctx); err == nil {
					t.Fatal("reentry snapshot")
				}
			}
			switch kind {
			case "snapshot-reentry":
				s.entries[0].parameter = &sharedCallbackBuffer{s.entries[0].parameter, cb}
				_, err := s.Snapshot(ctx)
				if err != nil || reentries != 1 {
					t.Fatal("snapshot", err, reentries)
				}
				s.entries[0].parameter = s.owned[0]
				if s.Close() != nil {
					t.Fatal("close")
				}
			case "close-reentry":
				s.owned[0] = &sharedCallbackBuffer{s.owned[0], cb}
				if s.Close() != nil || reentries != 1 || !s.closed {
					t.Fatal("close")
				}
			case "snapshot-failure":
				d.fail = "download"
				d.at = 2
				_, err := s.Snapshot(ctx)
				if err == nil {
					t.Fatal("snapshot failure")
				}
				assertSharedTerminalFences(t, s, d)
			case "close-failure":
				d.fail = "close"
				d.at = 2
				if s.Close() == nil || s.terminal == nil || s.closed {
					t.Fatal("partial close boundary")
				}
				closes := d.events["close"]
				if s.Close() == nil || d.events["close"] != closes {
					t.Fatal("close retried")
				}
			}
		})
	}
}
