package needle

import (
	"context"
	"fmt"
)

// GenerateCached returns the same bounded greedy contract as Generate, while
// ingesting each prompt/generated token once. Prepared weights and state live
// only for this invocation. Use NewDecoder directly for reusable sessions.
func (m *Model) GenerateCached(ctx context.Context, prompt []int, maxNew, eos int, opts DecoderOptions) ([]int, error) {
	if ctx == nil {
		return nil, fmt.Errorf("needle: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := m.validateTokens(prompt); err != nil {
		return nil, err
	}
	if maxNew < 0 || maxNew > m.config.MaxSeq-len(prompt) {
		return nil, fmt.Errorf("needle: generation exceeds context bound")
	}
	if eos < -1 || eos >= m.config.OutVocab {
		return nil, fmt.Errorf("needle: invalid EOS token")
	}
	if maxNew == 0 {
		return []int{}, nil
	}
	needed := len(prompt) + maxNew - 1
	if opts.Capacity == 0 {
		opts.Capacity = needed
	} else if opts.Capacity < needed {
		return nil, fmt.Errorf("needle: cache capacity smaller than generation request")
	}
	d, err := m.NewDecoder(opts)
	if err != nil {
		return nil, err
	}
	var logits []float32
	for _, token := range prompt {
		logits, err = d.Step(ctx, token)
		if err != nil {
			return nil, err
		}
	}
	generated := make([]int, 0, maxNew)
	for i := 0; i < maxNew; i++ {
		if err := ctx.Err(); err != nil {
			return generated, err
		}
		best := 0
		for token, v := range logits {
			if v > logits[best] {
				best = token
			}
		}
		generated = append(generated, best)
		if best == eos || i == maxNew-1 {
			break
		}
		logits, err = d.Step(ctx, best)
		if err != nil {
			return generated, err
		}
	}
	return generated, nil
}
