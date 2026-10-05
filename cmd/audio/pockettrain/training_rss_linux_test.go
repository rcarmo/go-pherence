//go:build linux

package main

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/model/pockettts"
)

type rssContextBoundary struct {
	boundaryRunner
	checkpoints int
	cancelStep  context.CancelFunc
	breach      func()
}

func (r *rssContextBoundary) CheckpointContext(ctx context.Context, directory, generation string, limits pockettts.TrainingBundleLimits) (pockettts.TrainingCheckpointBinding, error) {
	if r.breach != nil {
		r.breach()
		select {
		case <-ctx.Done():
		case <-time.After(time.Second):
			return pockettts.TrainingCheckpointBinding{}, errors.New("resource context not cancelled")
		}
	}
	if err := context.Cause(ctx); err != nil {
		return pockettts.TrainingCheckpointBinding{}, err
	}
	r.checkpoints++
	return r.boundaryRunner.Checkpoint(directory, generation, limits)
}

func TestTrainingRSSCheckpointCancellationLanes(t *testing.T) {
	parent, userCancel := context.WithCancel(context.Background())
	ctx, finish, err := watchTrainingRSS(parent, 100, time.Millisecond, func() (int64, error) { return 100, nil })
	if err != nil {
		t.Fatal(err)
	}
	runner := &rssContextBoundary{}
	runner.afterStep = func(int) { userCancel() }
	if err = runPreparedUpdates(ctx, rssGuardedRunner{runner, ctx}, "unused", 2, 10, pockettts.TrainingBundleLimits{}, io.Discard, boundaryGeneration); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if runner.checkpoints != 1 || len(runner.saved) != 1 || runner.saved[0] != 1 {
		t.Fatal("user cancellation lost dirty save", runner)
	}
	if err = finish(); err != nil {
		t.Fatal(err)
	}
	var fail atomic.Bool
	ctx, finish, err = watchTrainingRSS(context.Background(), 100, time.Millisecond, func() (int64, error) {
		if fail.Load() {
			return 101, nil
		}
		return 100, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	runner = &rssContextBoundary{breach: func() { fail.Store(true) }}
	if _, err = (rssGuardedRunner{runner, ctx}).Checkpoint("unused", "unused", pockettts.TrainingBundleLimits{}); !errors.Is(err, errTrainingRSS) {
		t.Fatal("late RSS checkpoint allowed", err)
	}
	if runner.checkpoints != 0 || len(runner.saved) != 0 {
		t.Fatal("resource cancellation published", runner)
	}
	if err = finish(); !errors.Is(err, errTrainingRSS) {
		t.Fatal(err)
	}
}

func TestTrainingRSSExportPublicationFence(t *testing.T) {
	dir := t.TempDir()
	source, destination := filepath.Join(dir, "source"), filepath.Join(dir, "published")
	if err := os.WriteFile(source, []byte("complete scratch export"), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(errTrainingRSS)
	if err := publishNewExportContext(ctx, source, destination); !errors.Is(err, errTrainingRSS) {
		t.Fatal(err)
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatal("resource failure published export", err)
	}
	if _, err := os.Stat(source); err != nil {
		t.Fatal("publication fence removed caller source", err)
	}
	ctx, cancel = context.WithCancelCause(context.Background())
	err := publishNewExportWithLink(ctx, source, destination, func(temp, path string) error {
		if err := os.Link(temp, path); err != nil {
			return err
		}
		cancel(errTrainingRSS)
		return nil
	})
	var published *pockettts.PocketExportPublishedError
	if !errors.As(err, &published) || !errors.Is(err, errTrainingRSS) || published.Path != destination {
		t.Fatal("late resource failure lost publication visibility", err)
	}
	if _, err := os.Stat(destination); err != nil {
		t.Fatal("visible export rolled back", err)
	}
}

func TestTrainingRSSGuard(t *testing.T) {
	for _, kind := range []string{"ceiling", "capture"} {
		t.Run(kind, func(t *testing.T) {
			var fail atomic.Bool
			read := func() (int64, error) {
				if fail.Load() {
					if kind == "capture" {
						return 0, io.ErrUnexpectedEOF
					}
					return 101, nil
				}
				return 100, nil
			}
			ctx, finish, err := watchTrainingRSS(context.Background(), 100, time.Millisecond, read)
			if err != nil {
				t.Fatal(err)
			}
			runner := &boundaryRunner{}
			wrapped := rssGuardedRunner{runner, ctx}
			fail.Store(true)
			select {
			case <-ctx.Done():
			case <-time.After(time.Second):
				t.Fatal("guard not cancelled")
			}
			if !errors.Is(context.Cause(ctx), errTrainingRSS) {
				t.Fatal("wrong RSS cause")
			}
			if err = runPreparedUpdates(ctx, wrapped, "unused", 1, 1, pockettts.TrainingBundleLimits{}, io.Discard, boundaryGeneration); !errors.Is(err, errTrainingRSS) {
				t.Fatal("RSS failure lost", err)
			}
			if runner.step != 0 || len(runner.saved) != 0 {
				t.Fatal("RSS failure updated/published", runner)
			}
			if _, err = wrapped.Checkpoint("unused", "unused", pockettts.TrainingBundleLimits{}); !errors.Is(err, errTrainingRSS) {
				t.Fatal("RSS allowed checkpoint")
			}
			if err = finish(); !errors.Is(err, errTrainingRSS) {
				t.Fatal("finish lost cause")
			}
		})
	}
	ctx, finish, err := watchTrainingRSS(context.Background(), 100, time.Millisecond, func() (int64, error) { return 100, nil })
	if err != nil {
		t.Fatal(err)
	}
	if ctx.Err() != nil || finish() != nil {
		t.Fatal("exact ceiling rejected")
	}
	if _, _, err = watchTrainingRSS(context.Background(), 100, time.Millisecond, func() (int64, error) { return 101, nil }); !errors.Is(err, errTrainingRSS) {
		t.Fatal("initial overcap")
	}
	if _, _, err = watchTrainingRSS(context.Background(), 100, time.Millisecond, func() (int64, error) { return 0, io.ErrUnexpectedEOF }); !errors.Is(err, errTrainingRSS) {
		t.Fatal("initial capture fail")
	}
	if _, _, err = watchTrainingRSS(nil, 100, time.Millisecond, trainingRSS); err == nil {
		t.Fatal("nil context")
	}
	rss, err := trainingRSS()
	if err != nil || rss <= 0 {
		t.Fatal("actual process RSS", rss, err)
	}
}
