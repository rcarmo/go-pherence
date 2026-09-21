package needle

import (
	"context"
	"fmt"
	"math"
)

// Generate performs bounded greedy FP32 decoding by recomputing each prefix.
// It is a correctness baseline, not the KV-cached quantized deployment engine.
func (m *Model) Generate(ctx context.Context, prompt []int, maxNew, eos int, opts Options) ([]int, error) {
	if ctx == nil {
		return nil, fmt.Errorf("needle: nil context")
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
	ids := append([]int(nil), prompt...)
	generated := make([]int, 0, maxNew)
	for step := 0; step < maxNew; step++ {
		if err := ctx.Err(); err != nil {
			return generated, err
		}
		logits, err := m.Forward(ids, opts)
		if err != nil {
			return generated, err
		}
		if err = ctx.Err(); err != nil {
			return generated, err
		}
		last := logits[(len(ids)-1)*m.config.OutVocab:]
		best := 0
		for i, v := range last {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return generated, fmt.Errorf("needle: nonfinite logits")
			}
			if v > last[best] {
				best = i
			}
		}
		ids = append(ids, best)
		generated = append(generated, best)
		if best == eos {
			break
		}
	}
	return generated, nil
}
