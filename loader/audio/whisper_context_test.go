package audio

import (
	"context"
	"errors"
	"math"
	"testing"
)

type melCheckpointContext struct {
	context.Context
	cancel    context.CancelFunc
	at, calls int
}

func newMelCheckpointContext(at int) *melCheckpointContext {
	ctx, cancel := context.WithCancel(context.Background())
	return &melCheckpointContext{Context: ctx, cancel: cancel, at: at}
}
func (c *melCheckpointContext) Err() error {
	c.calls++
	if c.at > 0 && c.calls == c.at {
		c.cancel()
	}
	return c.Context.Err()
}

func TestWhisperLogMelContextParityAndCancellation(t *testing.T) {
	for _, bands := range []int{80, 128} {
		for _, silence := range []bool{false, true} {
			samples := make([]float32, 1600)
			if !silence {
				for i := range samples {
					samples[i] = float32(math.Sin(float64(i)*0.031)) * 0.1
				}
			}
			want, n, err := WhisperLogMel(samples, bands)
			if err != nil {
				t.Fatal(err)
			}
			count := newMelCheckpointContext(0)
			got, m, err := WhisperLogMelContext(count, samples, bands)
			count.cancel()
			if err != nil || m != n || len(got) != len(want) {
				t.Fatalf("feature parity %v", err)
			}
			for i := range want {
				if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
					t.Fatalf("changed feature%d", i)
				}
			}
			for at := 1; at <= count.calls; at++ {
				ctx := newMelCheckpointContext(at)
				out, frames, err := WhisperLogMelContext(ctx, samples, bands)
				ctx.cancel()
				if out != nil || frames != 0 || !errors.Is(err, context.Canceled) {
					t.Fatalf("mel cancellation%d/%d:%v", at, count.calls, err)
				}
			}
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if out, n, err := WhisperLogMelContext(ctx, nil, 0); out != nil || n != 0 || !errors.Is(err, context.Canceled) {
		t.Fatal("precancel did feature validation/work")
	}
}
