package nvidia

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

type ptxResidentTape struct {
	tape *TrainingTape
	ran  bool
}

func (d *PTXResidentDriver) NewBoundTape(ctx context.Context, t train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	return d.newBoundTape(ctx, t, parameters, maxBytes, false, false)
}
func (d *PTXResidentDriver) NewReuseBoundTape(ctx context.Context, tape train.Tape, bindings map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	return d.newBoundTape(ctx, tape, bindings, maxBytes, true, false)
}
func (d *PTXResidentDriver) NewZeroBoundTape(ctx context.Context, t train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	return d.newBoundTape(ctx, t, parameters, maxBytes, true, true)
}
func (d *PTXResidentDriver) newBoundTape(ctx context.Context, t train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64, reuse, shareZero bool) (train.ResidentBoundTape, error) {
	if err := d.admit(ctx); err != nil {
		return nil, err
	}
	borrowed := map[int]*Buffer{}
	for id, input := range parameters {
		b, err := ptxResidentBuffers(input)
		if err != nil {
			return nil, err
		}
		borrowed[id] = b[0]
	}
	tape, err := newTrainingTapeBound(d.Operator, t, borrowed, maxBytes, reuse, shareZero)
	if tape == nil {
		return nil, err
	}
	return &ptxResidentTape{tape: tape}, err
}
func (d *PTXResidentDriver) ValidateAccumulate(ctx context.Context, dst, a, b train.ResidentBuffer, n int, scale float32) error {
	if err := d.admit(ctx); err != nil {
		return err
	}
	if err := train.VectorShape(n, n, 0, 0, scale); err != nil {
		return err
	}
	values, err := ptxResidentBuffers(a, b, dst)
	if err != nil {
		return err
	}
	if err := dualExtents(values, []int{n, n, n}); err != nil {
		return err
	}
	op := d.Operator
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.poison != nil || op.dualModule == nil || op.dualModule.Function("training_vector") == 0 {
		return fmt.Errorf("resident accumulation kernel unavailable")
	}
	return nil
}
func (t *ptxResidentTape) RunDevice(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("nil tape context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if t == nil || t.tape == nil || t.ran {
		return fmt.Errorf("resident tape already executed/nil")
	}
	if err := t.tape.runDevice(); err != nil {
		return err
	}
	t.ran = true
	return nil
}
func (t *ptxResidentTape) OutputBuffer(id int) (train.ResidentBuffer, error) {
	if t != nil && t.tape != nil && t.tape.reuse && !train.DeclaredTapeOutput(t.tape.outputs, id) {
		return nil, fmt.Errorf("lifetime tape buffer must be declared output")
	}
	if t == nil || t.tape == nil || !t.ran || t.tape.closed || t.tape.terminal != nil || id < 0 || id >= len(t.tape.buffers) {
		return nil, fmt.Errorf("resident output unavailable")
	}
	return &ptxResidentBuffer{buffer: t.tape.buffers[id], borrowed: true, sourceTape: t.tape}, nil
}
func (t *ptxResidentTape) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil tape context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if t == nil || t.tape == nil || !t.ran {
		return nil, fmt.Errorf("resident tape not executed")
	}
	return t.tape.downloadOutputs(ids)
}
func (t *ptxResidentTape) Close() error {
	if t == nil {
		return nil
	}
	return t.tape.Close()
}
