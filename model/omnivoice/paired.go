package omnivoice

import (
	"context"
	"fmt"
)

// forwardPair shares only streamed layer loading. Each branch retains its own
// full attention sequence, scratch, positions and SIMD matrix row grouping.
// Both preparations finish before loading; outputs after any error are invalid.
func (g *Generation) forwardPair(ctx context.Context, condIDs []int, condAudio []bool, uncondIDs []int, uncondAudio []bool, condPrefix, uncondPrefix []float32) error {
	b, u := g.conditional, g.unconditional
	if b == nil || u == nil || b == u || b.weights != u.weights || b.layer != u.layer || b.resident != nil || u.resident != nil || b.directQ8 != nil || u.directQ8 != nil {
		return fmt.Errorf("omnivoice: shared traversal requires distinct streamed float32 siblings")
	}
	if err := b.prepareForward(ctx, g.condTarget, condIDs, condAudio, nil, nil, g.target, condPrefix, g.activeTimes); err != nil {
		return err
	}
	if err := u.prepareForward(ctx, g.uncondTarget, uncondIDs, uncondAudio, nil, nil, g.target, uncondPrefix, g.activeTimes); err != nil {
		return err
	}
	b.bindExecution(ctx)
	defer b.clearExecution()
	u.bindExecution(ctx)
	defer u.clearExecution()
	for i := 0; i < b.weights.Config.LLMConfig.NumHiddenLayers; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := b.layer.Load(b.weights, i); err != nil {
			return err
		}
		if err := b.block.ForwardInto(b.hidden, b.hidden, b.tokens, nil, nil, b.scratch); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := u.block.ForwardInto(u.hidden, u.hidden, u.tokens, nil, nil, u.scratch); err != nil {
			return err
		}
	}
	if err := b.projectForward(ctx, g.condTarget, g.target, g.activeTimes); err != nil {
		return err
	}
	return u.projectForward(ctx, g.uncondTarget, g.target, g.activeTimes)
}
