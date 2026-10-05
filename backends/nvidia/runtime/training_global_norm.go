package nvidia

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (d *PTXResidentDriver) validateGlobalNorm(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32, maxNorm float32) ([]*Buffer, error) {
	if err := d.admit(ctx); err != nil {
		return nil, err
	}
	count := 4
	out := 4
	if mode != 1 {
		var err error
		count, err = train.NormMergeShape(n, previous)
		if err != nil {
			return nil, err
		}
	} else {
		out = 4
		if err := train.NormClipShape(previous, maxNorm); err != nil {
			return nil, err
		}
	}
	buffers, err := ptxResidentBuffers(a, b, dst)
	if err != nil {
		return nil, err
	}
	if err := dualExtents(buffers, []int{4, count, out}); err != nil {
		return nil, err
	}
	op := d.Operator
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.poison != nil || op.dualModule == nil || op.dualModule.Function("training_global_norm") == 0 {
		return nil, fmt.Errorf("global norm PTX entry unavailable")
	}
	return buffers, nil
}
func (d *PTXResidentDriver) ValidateNormMerge(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32) error {
	if mode != 0 && mode != 2 {
		return fmt.Errorf("global norm merge mode")
	}
	_, err := d.validateGlobalNorm(ctx, dst, a, b, n, previous, mode, 1)
	return err
}
func (d *PTXResidentDriver) ValidateNormClip(ctx context.Context, dst, a train.ResidentBuffer, elements int64, maxNorm float32) error {
	_, err := d.validateGlobalNorm(ctx, dst, a, a, 1, elements, 1, maxNorm)
	return err
}
func (d *PTXResidentDriver) runGlobalNorm(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32, maxNorm float32) error {
	buffers, err := d.validateGlobalNorm(ctx, dst, a, b, n, previous, mode, maxNorm)
	if err != nil {
		return err
	}
	op := d.Operator
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.poison != nil || op.dualModule == nil {
		return fmt.Errorf("terminal global norm operator")
	}
	fn := op.dualModule.Function("training_global_norm")
	if fn == 0 {
		return fmt.Errorf("global norm PTX entry unavailable")
	}
	nn, count := uint32(n), float32(previous)
	err = LaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&buffers[0].Ptr), unsafe.Pointer(&buffers[1].Ptr), unsafe.Pointer(&buffers[2].Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&mode), unsafe.Pointer(&count), unsafe.Pointer(&maxNorm))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
func (d *PTXResidentDriver) NormMerge(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32) error {
	if mode != 0 && mode != 2 {
		return fmt.Errorf("global norm merge mode")
	}
	return d.runGlobalNorm(ctx, dst, a, b, n, previous, mode, 1)
}
func (d *PTXResidentDriver) NormClip(ctx context.Context, dst, a train.ResidentBuffer, elements int64, maxNorm float32) error {
	return d.runGlobalNorm(ctx, dst, a, a, 1, elements, 1, maxNorm)
}
