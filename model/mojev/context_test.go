package mojev

import (
	"context"
	"errors"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
)

func TestLaunchGroupsCancellationDrains(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	launched, drained := 0, 0
	err := runLaunchGroups(ctx, 40, func(start, end int) error {
		launched++
		if start != 0 || end != 16 {
			t.Fatal(start, end)
		}
		cancel()
		return nil
	}, func() error { drained++; return nil })
	if !errors.Is(err, context.Canceled) || launched != 1 || drained != 1 {
		t.Fatal(err, launched, drained)
	}
	launched, drained = 0, 0
	err = runLaunchGroups(context.Background(), 40, func(start, end int) error {
		launched++
		if start != 0 || end != 40 {
			t.Fatal(start, end)
		}
		return nil
	}, func() error { drained++; return nil })
	if err != nil || launched != 1 || drained != 1 {
		t.Fatal(err, launched, drained)
	}
	boom, drainFail := errors.New("launch"), errors.New("drain")
	drained = 0
	err = runLaunchGroups(context.Background(), 40, func(int, int) error { return boom }, func() error { drained++; return drainFail })
	if !errors.Is(err, boom) || !errors.Is(err, drainFail) || drained != 1 {
		t.Fatal(err, drained)
	}
	if err = runLaunchGroups(nil, 1, nil, nil); err == nil {
		t.Fatal("nil context")
	}
	if err = runLaunchGroups(ctx, 1, nil, nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	var groups [][2]int
	err = runLaunchGroups(ctx2, 35, func(a, b int) error { groups = append(groups, [2]int{a, b}); return nil }, func() error { return nil })
	if err != nil || !reflect.DeepEqual(groups, [][2]int{{0, 16}, {16, 32}, {32, 35}}) {
		t.Fatal(err, groups)
	}
	if a := testing.AllocsPerRun(20, func() {
		if e := runLaunchGroups(context.Background(), 30, func(int, int) error { return nil }, func() error { return nil }); e != nil {
			panic(e)
		}
	}); a != 0 {
		t.Fatal("background group allocation", a)
	}
}

func TestScorerContextWaitCancellation(t *testing.T) {
	s := &SIMDTextScorer{cpu: &TextScorer{}, branch: &qwen.Qwen35SIMDBranch{}}
	g := &NVIDIATextScorer{cpu: &TextScorer{}}
	for _, b := range []struct {
		name         string
		lock, unlock func()
		score        func(context.Context, EncodedRow) ([][]float32, error)
		text         func(context.Context, TextRequest, *tokenizer.Tokenizer, int, int) (*TextDecision, error)
	}{
		{"simd", s.mu.Lock, s.mu.Unlock, s.ScoreEncodedContext, s.ScoreTextContext},
		{"gpu", g.mu.Lock, g.mu.Unlock, g.ScoreEncodedContext, g.ScoreTextContext},
	} {
		t.Run(b.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if out, err := b.score(ctx, repairedTextRow()); !errors.Is(err, context.Canceled) || out != nil {
				t.Fatal(err)
			}
			if out, err := b.score(nil, repairedTextRow()); err == nil || out != nil {
				t.Fatal("nil context")
			}
			if out, err := b.text(ctx, TextRequest{}, nil, 1, 1); !errors.Is(err, context.Canceled) || out != nil {
				t.Fatal("canceled text", err)
			}
			if out, err := b.text(nil, TextRequest{}, nil, 1, 1); err == nil || out != nil {
				t.Fatal("nil text context")
			}
			deadline, stop := context.WithDeadline(context.Background(), time.Unix(0, 0))
			defer stop()
			if out, err := b.score(deadline, repairedTextRow()); !errors.Is(err, context.DeadlineExceeded) || out != nil {
				t.Fatal("deadline", err)
			}
			b.lock()
			defer b.unlock()
			waiting, cancelWait := context.WithCancel(context.Background())
			defer cancelWait()
			done := make(chan error, 1)
			go func() {
				out, err := b.score(waiting, repairedTextRow())
				if out != nil {
					err = errors.New("partial result")
				}
				done <- err
			}()
			cancelWait()
			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("canceled caller stuck behind mutex")
			}
		})
	}
}

// Cancels an actual standard Context at a deterministic execution poll. There
// are no sleeps, timing guesses or backend test hooks in the execution path.
type pollCancelContext struct {
	context.Context
	cancel    context.CancelFunc
	remaining atomic.Int32
}

func (c *pollCancelContext) Err() error {
	if c.remaining.Add(-1) == 0 {
		c.cancel()
	}
	return c.Context.Err()
}

func testReleasedCancellation(t *testing.T, b interface {
	ScoreEncoded(EncodedRow) ([][]float32, error)
	ScoreEncodedContext(context.Context, EncodedRow) ([][]float32, error)
}) {
	t.Helper()
	row := repairedTextRow()
	bad := repairedTextRow()
	bad.Candidates[0][0][0] = -1
	if out, err := b.ScoreEncodedContext(context.Background(), bad); err == nil || out != nil {
		t.Fatal("negative ID admitted")
	}
	want, err := b.ScoreEncoded(row)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	probe := &pollCancelContext{Context: ctx, cancel: cancel}
	probe.remaining.Store(10)
	out, err := b.ScoreEncodedContext(probe, row)
	if !errors.Is(err, context.Canceled) || out != nil || probe.remaining.Load() > 0 {
		t.Fatal("active cancellation", err, probe.remaining.Load())
	}
	recovered, err := b.ScoreEncoded(row)
	if err != nil || !reflect.DeepEqual(recovered, want) {
		t.Fatal("cancel recovery", err)
	}
	cancellable, stop := context.WithCancel(context.Background())
	defer stop()
	got, err := b.ScoreEncodedContext(cancellable, row)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("cancellable normal result", err)
	}
}
