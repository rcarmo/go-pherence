package nvidia

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

type PTXResidentDriver struct{ Operator *TrainingF32 }

func (d *PTXResidentDriver) admit(ctx context.Context) error {
	if d == nil || d.Operator == nil || ctx == nil {
		return fmt.Errorf("explicit PTX resident context/operator required")
	}
	return ctx.Err()
}

type ptxResidentBuffer struct {
	buffer     *Buffer
	borrowed   bool
	sourceTape *TrainingTape
}

func (b *ptxResidentBuffer) Upload(ctx context.Context, x []float32) error {
	if b.borrowed {
		return fmt.Errorf("borrowed tape output is read-only")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return b.buffer.Upload(x)
}
func (b *ptxResidentBuffer) Download(ctx context.Context, x []float32) error {
	if b.sourceTape != nil && (b.sourceTape.closed || b.sourceTape.terminal != nil) {
		return fmt.Errorf("borrowed tape output expired")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return b.buffer.Download(x)
}
func (b *ptxResidentBuffer) Close() error {
	if b.borrowed {
		return fmt.Errorf("borrowed tape output cannot be closed")
	}
	return b.buffer.FreeChecked()
}
func (d *PTXResidentDriver) NewBuffer(ctx context.Context, n int) (train.ResidentBuffer, error) {
	if ctx == nil || d == nil || d.Operator == nil {
		return nil, fmt.Errorf("explicit PTX resident owner required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := train.NormStatsShape(n); err != nil {
		return nil, err
	}
	d.Operator.mu.Lock()
	ready := d.Operator.poison == nil && d.Operator.module != nil && d.Operator.dualModule != nil && d.Operator.module.Function("training_adamw") != 0 && d.Operator.dualModule.Function("training_vector") != 0 && d.Operator.dualModule.Function("training_norm_stats") != 0
	d.Operator.mu.Unlock()
	if !ready {
		return nil, fmt.Errorf("PTX resident kernels unavailable")
	}
	b, err := Malloc(n)
	if b == nil {
		return nil, err
	}
	return &ptxResidentBuffer{buffer: b}, err
}
func ptxResidentBuffers(values ...train.ResidentBuffer) ([]*Buffer, error) {
	out := make([]*Buffer, len(values))
	for i, x := range values {
		b, ok := x.(*ptxResidentBuffer)
		if !ok || b == nil || b.buffer == nil || (b.sourceTape != nil && (b.sourceTape.closed || b.sourceTape.terminal != nil)) {
			return nil, fmt.Errorf("foreign/nil PTX resident buffer")
		}
		out[i] = b.buffer
	}
	return out, nil
}
func (d *PTXResidentDriver) Accumulate(ctx context.Context, dst, a, b train.ResidentBuffer, n int, scale float32) error {
	if err := d.admit(ctx); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	v, err := ptxResidentBuffers(dst, a, b)
	if err != nil {
		return err
	}
	return d.Operator.Vector(v[0], v[1], v[2], n, n, 0, 0, scale)
}
func (d *PTXResidentDriver) NormStats(ctx context.Context, dst, x train.ResidentBuffer, n int, nonnegative bool) error {
	if err := d.admit(ctx); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	v, err := ptxResidentBuffers(dst, x)
	if err != nil {
		return err
	}
	return d.Operator.NormStats(v[0], v[1], n, nonnegative)
}
func (d *PTXResidentDriver) ValidateUpdate(ctx context.Context, p, g, m, v, ema train.ResidentBuffer, n int, step train.AdamWStep) error {
	if err := d.admit(ctx); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := step.Validate(); err != nil {
		return err
	}
	b, err := ptxResidentBuffers(p, g, m, v, ema)
	if err != nil {
		return err
	}
	if n < 1 || n > 1<<26 {
		return fmt.Errorf("PTX resident update bound")
	}
	for i, x := range b {
		if !trainingExtent(x, n) {
			return fmt.Errorf("PTX resident update extent")
		}
		for j := 0; j < i; j++ {
			if trainingOverlap(x, n, b[j], n) {
				return fmt.Errorf("PTX resident update alias")
			}
		}
	}
	if d == nil || d.Operator == nil {
		return fmt.Errorf("nil PTX resident operator")
	}
	op := d.Operator
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.poison != nil || op.module == nil || op.module.Function("training_adamw") == 0 {
		return fmt.Errorf("PTX resident update kernel unavailable")
	}
	return nil
}
func (d *PTXResidentDriver) Update(ctx context.Context, p, g, m, v, ema train.ResidentBuffer, n int, step train.AdamWStep) error {
	if err := d.ValidateUpdate(ctx, p, g, m, v, ema, n, step); err != nil {
		return err
	}
	b, _ := ptxResidentBuffers(p, g, m, v, ema)
	return d.Operator.AdamW(b[0], b[1], b[2], b[3], b[4], n, step)
}

// NormStats writes (maximum, scaled-square-sum, invalid) per <=256-value
// chunk; optional nonnegative admission is used for second moments. Native
// failure poisons the operator; callers retain owners until coherent recovery.
func (op *TrainingF32) NormStats(dst, x *Buffer, n int, nonnegative bool) error {
	count, err := train.NormStatsShape(n)
	if err != nil {
		return err
	}
	if err := dualExtents([]*Buffer{x, dst}, []int{n, count}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX norm operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.poison != nil || op.dualModule == nil {
		return fmt.Errorf("closed/terminal PTX norm operator")
	}
	fn := op.dualModule.Function("training_norm_stats")
	if fn == 0 {
		return fmt.Errorf("norm PTX entry unavailable")
	}
	nn, positive := uint32(n), uint32(0)
	if nonnegative {
		positive = 1
	}
	err = LaunchKernel(fn, uint32((count/3-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&positive))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
