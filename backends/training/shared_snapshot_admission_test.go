package training

import (
	"context"
	"reflect"
	"testing"
)

// Cancellation can arrive after all transfers while destination validation
// scans the full state. Count context checks to isolate that final boundary;
// no global hook or production cancellation seam is added.
type snapshotFinalCancelContext struct {
	context.Context
	calls, at int
}

func (c *snapshotFinalCancelContext) Err() error {
	c.calls++
	if c.calls >= c.at {
		return context.Canceled
	}
	return nil
}

func TestTrainingSharedSnapshotIntoFinalCancellation(t *testing.T) {
	s, d := requireShared(t, residentFixture())
	dst, err := s.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	s.step++
	// s.admit once, then each mock Download and following explicit check,
	// finally the check after full destination numerical validation.
	ctx := &snapshotFinalCancelContext{Context: context.Background(), at: 2 + 8*len(s.entries)}
	before := d.events["download"]
	if err := s.SnapshotInto(ctx, &dst); err != context.Canceled || s.terminal == nil || s.borrowed || dst.Step != 0 {
		t.Fatal("late validation cancellation committed", err, ctx.calls)
	}
	if d.events["download"]-before != 4*len(s.entries) {
		t.Fatal("not cancelled at final boundary")
	}
	if s.Close() == nil {
		t.Fatal("terminal owner freed")
	}
}

func TestTrainingSharedSnapshotIntoUnequalDestinationInventory(t *testing.T) {
	for _, kind := range []string{"renamed", "consistent-short"} {
		t.Run(kind, func(t *testing.T) {
			s, d := requireShared(t, residentFixture())
			dst, err := s.Snapshot(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if kind == "renamed" {
				p := dst.Parameters["a"]
				delete(dst.Parameters, "a")
				dst.Parameters["other"] = p
			} else {
				p := dst.Parameters["a"]
				p.Parameter = p.Parameter[:1]
				p.M = p.M[:1]
				p.V = p.V[:1]
				p.EMA = p.EMA[:1]
				dst.Parameters["a"] = p
			}
			if _, err := ResidentBytes(dst); err != nil {
				t.Fatal("fixture must reach exact destination inventory", err)
			}
			before := d.events["download"]
			copyDst := dst
			if err := s.SnapshotInto(context.Background(), &dst); err == nil || d.events["download"] != before || s.terminal != nil {
				t.Fatal("bad inventory transferred/poisoned")
			}
			if !reflect.DeepEqual(copyDst, dst) {
				t.Fatal("admission changed metadata")
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
