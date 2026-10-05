package nvidia

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (d *PTXResidentDriver) ValidateScaledUpdate(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, step train.AdamWStep) error {
	if err := d.admit(ctx); err != nil {
		return err
	}
	b, err := ptxResidentBuffers(p, g, m, v, ema, clip)
	if err != nil {
		return err
	}
	return d.Operator.validateScaledAdamW(b, n, step)
}
func (d *PTXResidentDriver) UpdateScaled(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, step train.AdamWStep) error {
	if err := d.ValidateScaledUpdate(ctx, p, g, m, v, ema, clip, n, step); err != nil {
		return err
	}
	b, _ := ptxResidentBuffers(p, g, m, v, ema, clip)
	return d.Operator.AdamWScaled(b[0], b[1], b[2], b[3], b[4], b[5], n, step)
}
func (op *TrainingF32) validateScaledAdamW(b []*Buffer, n int, p train.AdamWStep) error {
	if err := p.Validate(); err != nil {
		return err
	}
	if p.GradientScale != 1 || n < 1 || n > 1<<26 || len(b) != 6 {
		return fmt.Errorf("device-scale AdamW constants/extent")
	}
	sizes := []int{n, n, n, n, n, 4}
	for i, x := range b {
		if !trainingExtent(x, sizes[i]) {
			return fmt.Errorf("device-scale AdamW buffer extent")
		}
		for j := 0; j < i; j++ {
			if trainingOverlap(x, sizes[i], b[j], sizes[j]) {
				return fmt.Errorf("device-scale AdamW buffer alias")
			}
		}
	}
	if op == nil {
		return fmt.Errorf("nil device-scale AdamW operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.poison != nil || op.dualModule == nil || op.dualModule.Function("training_adamw_scaled") == 0 {
		return fmt.Errorf("device-scale AdamW kernel unavailable")
	}
	return nil
}

// AdamWScaled consumes clip[3] directly. The caller must first validate the
// completed clip diagnostics; no device readback or fallback occurs here.
func (op *TrainingF32) AdamWScaled(parameter, gradient, m, v, ema, clip *Buffer, n int, p train.AdamWStep) error {
	if err := op.validateScaledAdamW([]*Buffer{parameter, gradient, m, v, ema, clip}, n, p); err != nil {
		return err
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.poison != nil || op.dualModule == nil {
		return fmt.Errorf("closed/terminal device-scale AdamW operator")
	}
	fn := op.dualModule.Function("training_adamw_scaled")
	if fn == 0 {
		return fmt.Errorf("device-scale AdamW kernel unavailable")
	}
	count := uint32(n)
	err := LaunchKernel(fn, uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&parameter.Ptr), unsafe.Pointer(&gradient.Ptr), unsafe.Pointer(&m.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&ema.Ptr), unsafe.Pointer(&clip.Ptr), unsafe.Pointer(&count),
		unsafe.Pointer(&p.LearningRate), unsafe.Pointer(&p.Beta1), unsafe.Pointer(&p.Beta2), unsafe.Pointer(&p.Epsilon), unsafe.Pointer(&p.WeightDecay), unsafe.Pointer(&p.Bias1), unsafe.Pointer(&p.Bias2), unsafe.Pointer(&p.EMADecay))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
