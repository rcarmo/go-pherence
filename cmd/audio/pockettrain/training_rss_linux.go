//go:build linux

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/model/pockettts"
)

var errTrainingRSS = errors.New("training RSS ceiling exceeded or capture failed")

type rssCheckpointContextKey struct{}

func trainingRSS() (int64, error) {
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0, err
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return 0, fmt.Errorf("RSS statm fields")
	}
	pages, err := strconv.ParseInt(fields[1], 10, 64)
	size := int64(os.Getpagesize())
	if err != nil || pages < 0 || pages > (1<<63-1)/size {
		return 0, fmt.Errorf("RSS extent")
	}
	return pages * size, nil
}

// Fail closed on observed RSS/capture failure; no GC, eviction, reset or retry.
// Sampling cannot prevent a between-sample overshoot; admission still reserves
// all declared owners before model/native allocation.
func watchTrainingRSS(parent context.Context, ceiling int64, interval time.Duration, read func() (int64, error)) (context.Context, func() error, error) {
	if parent == nil || ceiling <= 0 || interval <= 0 || read == nil {
		return nil, nil, fmt.Errorf("RSS guard inputs")
	}
	check := func() error {
		rss, err := read()
		if err != nil {
			return fmt.Errorf("%w: %v", errTrainingRSS, err)
		}
		if rss < 0 || rss > ceiling {
			return fmt.Errorf("%w: observed=%d ceiling=%d", errTrainingRSS, rss, ceiling)
		}
		return nil
	}
	if err := check(); err != nil {
		return nil, nil, err
	}
	// User cancellation still saves a dirty completed boundary. Checkpoint
	// serialization receives only the independent resource-cancellation lane.
	resource, resourceCancel := context.WithCancelCause(context.WithoutCancel(parent))
	step, cancel := context.WithCancelCause(parent)
	ctx := context.WithValue(step, rssCheckpointContextKey{}, resource)
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-resource.Done():
				return
			case <-ticker.C:
				if err := check(); err != nil {
					resourceCancel(err)
					cancel(err)
					return
				}
			}
		}
	}()
	finish := func() error {
		close(stop)
		<-done
		cause := context.Cause(resource)
		resourceCancel(nil)
		cancel(nil)
		if errors.Is(cause, errTrainingRSS) {
			return cause
		}
		return nil
	}
	return ctx, finish, nil
}

type rssGuardedRunner struct {
	preparedUpdateRunner
	ctx context.Context
}

func (r rssGuardedRunner) Step(ctx context.Context) (pockettts.TrainingStepMetrics, error) {
	if cause := context.Cause(r.ctx); errors.Is(cause, errTrainingRSS) {
		return pockettts.TrainingStepMetrics{}, cause
	}
	metrics, err := r.preparedUpdateRunner.Step(ctx)
	if cause := context.Cause(r.ctx); errors.Is(cause, errTrainingRSS) {
		return pockettts.TrainingStepMetrics{}, errors.Join(cause, err)
	}
	return metrics, err
}
func (r rssGuardedRunner) Checkpoint(directory, generation string, limits pockettts.TrainingBundleLimits) (pockettts.TrainingCheckpointBinding, error) {
	if cause := context.Cause(r.ctx); errors.Is(cause, errTrainingRSS) {
		return pockettts.TrainingCheckpointBinding{}, cause
	}
	if checkpoint, ok := r.preparedUpdateRunner.(interface {
		CheckpointContext(context.Context, string, string, pockettts.TrainingBundleLimits) (pockettts.TrainingCheckpointBinding, error)
	}); ok {
		checkpointCtx := r.ctx
		if resource, ok := r.ctx.Value(rssCheckpointContextKey{}).(context.Context); ok {
			checkpointCtx = resource
		}
		if cause := context.Cause(checkpointCtx); errors.Is(cause, errTrainingRSS) {
			return pockettts.TrainingCheckpointBinding{}, cause
		}
		return checkpoint.CheckpointContext(checkpointCtx, directory, generation, limits)
	}
	return r.preparedUpdateRunner.Checkpoint(directory, generation, limits)
}
