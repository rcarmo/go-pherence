package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func BenchmarkTrainingSharedSnapshotStorage(b *testing.B) {
	// The test driver's exhaustive pairwise element alias scan is quadratic;
	// keep constructor size bounded instead of bypassing its validation.
	const n = 1024
	fixture := residentFixture()
	fixture.Parameters = map[string]ResidentParameter{"p": {Parameter: make([]float32, n), M: make([]float32, n), V: make([]float32, n), EMA: make([]float32, n)}}
	for _, mode := range []string{"public-owned", "private-reuse"} {
		b.Run(mode, func(b *testing.B) {
			d := &sharedHostDriver{}
			state, err := NewSharedWorkspaceResidentState(context.Background(), d, fixture, 2<<20)
			if err != nil {
				b.Fatal(err)
			}
			defer func() {
				if err := state.Close(); err != nil {
					b.Fatal(err)
				}
			}()
			dst, err := state.Snapshot(context.Background())
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if mode == "public-owned" {
					dst, err = state.Snapshot(context.Background())
				} else {
					err = state.SnapshotInto(context.Background(), &dst)
				}
				if err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			if dst.Step != state.StepCount() {
				b.Fatal("snapshot metadata")
			}
		})
	}
}

func TestTrainingSharedSnapshotIntoOwnership(t *testing.T) {
	s, d := requireShared(t, residentFixture())
	ctx := context.Background()
	dst, err := s.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	pointers := dst.Parameters
	for step := 0; step < 3; step++ {
		if err := s.Accumulate(ctx, map[string][]float32{"a": {1, -.2}, "b": {.1}}, 1); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Update(ctx, .5); err != nil {
			t.Fatal(err)
		}
		public, err := s.Snapshot(ctx)
		if err != nil {
			t.Fatal(err)
		}
		allocations := len(d.buffers)
		if err := s.SnapshotInto(ctx, &dst); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(public, dst) || len(d.buffers) != allocations || s.borrowed {
			t.Fatal("snapshot parity/allocation/borrow")
		}
		for name, p := range dst.Parameters {
			old := pointers[name]
			if &old.Parameter[0] != &p.Parameter[0] || &old.M[0] != &p.M[0] || &old.V[0] != &p.V[0] || &old.EMA[0] != &p.EMA[0] {
				t.Fatal("destination storage replaced")
			}
		}
		public.Parameters["a"].Parameter[0] += 1
		fresh, err := s.Snapshot(ctx)
		if err != nil || !reflect.DeepEqual(fresh, dst) {
			t.Fatal("public or destination aliases native state", err)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTrainingSharedSnapshotIntoAllocations(t *testing.T) {
	fixture := sharedFixture(2)
	s, _ := requireShared(t, fixture)
	dst, err := s.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	reused := testing.AllocsPerRun(10, func() {
		if err := s.SnapshotInto(context.Background(), &dst); err != nil {
			panic(err)
		}
	})
	public := testing.AllocsPerRun(10, func() {
		if _, err := s.Snapshot(context.Background()); err != nil {
			panic(err)
		}
	})
	if reused > 5 || reused >= public {
		t.Fatal("snapshot metadata allocation regression", reused, public)
	}
	t.Logf("two tensor snapshot allocations: fresh=%g reused=%g; no parameter-sized reused arrays", public, reused)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTrainingSharedSnapshotIntoAdmission(t *testing.T) {
	for _, kind := range []string{"nil-dst", "config", "step", "name", "count", "extent", "alias-same", "alias-cross", "alias-offset", "nonfinite", "negativeV", "dirty", "borrowed", "lease", "workspace", "closed", "terminal", "cancelled", "nil-context"} {
		t.Run(kind, func(t *testing.T) {
			s, d := requireShared(t, residentFixture())
			ctx := context.Background()
			dst, err := s.Snapshot(ctx)
			if err != nil {
				t.Fatal(err)
			}
			arg := &dst
			switch kind {
			case "nil-dst":
				arg = nil
			case "config":
				dst.Config.Epsilon *= 2
			case "step":
				dst.Step++
			case "name":
				dst.Parameters["foreign"] = dst.Parameters["a"]
				delete(dst.Parameters, "a")
			case "count":
				delete(dst.Parameters, "a")
			case "extent":
				dst.Parameters["a"] = dst.Parameters["b"]
			case "alias-same":
				p := dst.Parameters["a"]
				p.M = p.Parameter
				dst.Parameters["a"] = p
			case "alias-cross":
				p := dst.Parameters["b"]
				p.M = dst.Parameters["a"].Parameter[:1]
				dst.Parameters["b"] = p
			case "alias-offset":
				p := dst.Parameters["a"]
				base := []float32{0, 0, 0}
				p.M, p.V = base[:2], base[1:]
				dst.Parameters["a"] = p
			case "nonfinite":
				dst.Parameters["a"].M[0] = float32(math.NaN())
			case "negativeV":
				dst.Parameters["a"].V[0] = -1
			case "dirty":
				s.dirty = true
			case "borrowed":
				s.borrowed = true
			case "lease":
				s.lease = &ResidentTapeLease{}
			case "workspace":
				s.shared.active = true
			case "closed":
				s.closed = true
			case "terminal":
				s.terminal = errors.New("fixture terminal")
			case "cancelled":
				cancelCtx, cancel := context.WithCancel(ctx)
				cancel()
				ctx = cancelCtx
			case "nil-context":
				ctx = nil
			}
			downloads := d.events["download"]
			before := map[string][4][]uint32{}
			for name, p := range dst.Parameters {
				var slots [4][]uint32
				for slot, x := range [][]float32{p.Parameter, p.M, p.V, p.EMA} {
					for _, v := range x {
						slots[slot] = append(slots[slot], math.Float32bits(v))
					}
				}
				before[name] = slots
			}
			if err := s.SnapshotInto(ctx, arg); err == nil || d.events["download"] != downloads {
				t.Fatal("invalid destination/state performed transfers")
			}
			for name, p := range dst.Parameters {
				for slot, x := range [][]float32{p.Parameter, p.M, p.V, p.EMA} {
					for i, v := range x {
						if math.Float32bits(v) != before[name][slot][i] {
							t.Fatal("admission mutated destination")
						}
					}
				}
			}
		})
	}
	if err := (*ResidentState)(nil).SnapshotInto(context.Background(), nil); err == nil {
		t.Fatal("nil state")
	}
	legacy, _ := requireResident(t)
	dst, _ := legacy.Snapshot(context.Background())
	if legacy.SnapshotInto(context.Background(), &dst) == nil {
		t.Fatal("legacy route changed")
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}
}

type snapshotCancelBuffer struct {
	ResidentBuffer
	cancel context.CancelFunc
}

func (b *snapshotCancelBuffer) Download(ctx context.Context, dst []float32) error {
	if err := b.ResidentBuffer.Download(ctx, dst); err != nil {
		return err
	}
	b.cancel()
	return nil // Simulate cancellation observed just after completed copy.
}

func TestTrainingSharedSnapshotIntoFailureRetention(t *testing.T) {
	for _, kind := range []string{"download", "negativeV", "nonfinite", "late-cancel"} {
		t.Run(kind, func(t *testing.T) {
			s, d := requireShared(t, residentFixture())
			ctx := context.Background()
			dst, err := s.Snapshot(ctx)
			if err != nil {
				t.Fatal(err)
			}
			s.step++ // Completed update stand-in, no numerical qualification.
			s.entries[0].parameter.(*residentHostBuffer).values[0] += .1
			switch kind {
			case "download":
				d.fail, d.at = "download", d.events["download"]+2
			case "negativeV":
				s.entries[0].v.(*residentHostBuffer).values[0] = -1
			case "nonfinite":
				s.entries[0].m.(*residentHostBuffer).values[0] = float32(math.NaN())
			case "late-cancel":
				cancelCtx, cancel := context.WithCancel(ctx)
				defer cancel()
				ctx = cancelCtx
				s.entries[0].m = &snapshotCancelBuffer{s.entries[0].m, cancel}
			}
			old := dst.Parameters[s.entries[0].name].Parameter[0]
			if err := s.SnapshotInto(ctx, &dst); err == nil || s.terminal == nil || s.borrowed || dst.Step != 0 || dst.Parameters[s.entries[0].name].Parameter[0] == old {
				t.Fatal("partial snapshot was not terminal or advanced metadata", err)
			}
			calls := d.events["download"]
			if s.SnapshotInto(ctx, &dst) == nil || s.Close() == nil {
				t.Fatal("terminal owner retried/freed")
			}
			if _, err := s.Snapshot(ctx); err == nil || d.events["download"] != calls {
				t.Fatal("terminal owner exported")
			}
		})
	}
}
