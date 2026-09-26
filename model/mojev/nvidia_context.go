package mojev

import (
	"context"
	"errors"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func (g *NVIDIATextScorer) launchContext(ctx context.Context) error {
	err := runLaunchGroups(ctx, g.commandCount, func(start, end int) error { return nvidia.LaunchBatch(g.commands[start:end]) }, nvidia.SyncErr)
	if err != nil && err != ctx.Err() {
		// A driver error can leave the context unusable. Disable inference; Close
		// retains module/buffer ownership until its synchronisation succeeds.
		g.closed = true
	}
	return err
}

// Cancellation cannot preempt a kernel. Cancellable requests submit at most 16
// commands at a time and drain them before inspecting cancellation or returning.
// Background requests retain the single-batch path and one synchronisation.
func runLaunchGroups(ctx context.Context, count int, launch func(int, int) error, drain func() error) error {
	if ctx == nil {
		return errors.New("mojev: nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	size := count
	if ctx.Done() != nil && size > 16 {
		size = 16
	}
	for start := 0; start < count; {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(count, start+size)
		launchErr := launch(start, end)
		// Even a failed launch may have enqueued a prefix. Always drain it.
		drainErr := drain()
		if launchErr != nil || drainErr != nil {
			return errors.Join(launchErr, drainErr)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		start = end
	}
	return nil
}
