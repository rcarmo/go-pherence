package nvidia

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

var _ train.ResidentHandoffTapeDriver = (*PTXResidentDriver)(nil)

func (d *PTXResidentDriver) ValidateHandoffTape(ctx context.Context, t train.Tape, bindings map[int]train.ResidentBuffer, maxBytes int64) error {
	if err := d.admit(ctx); err != nil {
		return err
	}
	if err := t.Validate(maxBytes); err != nil {
		return err
	}
	if err := train.ValidateExternalTapeBindings(t, bindings); err != nil {
		return err
	}
	if err := d.Operator.admitTapeKernels(t); err != nil {
		return err
	}
	buffers := map[int]*Buffer{}
	for id, b := range bindings {
		if id < 0 || id >= len(t.Tensors) || (t.Tensors[id].Values == nil && !t.Tensors[id].External) {
			return fmt.Errorf("PTX handoff bound constant ID")
		}
		values, err := ptxResidentBuffers(b)
		if err != nil {
			return err
		}
		n, _ := t.Tensors[id].Elements()
		if !trainingExtent(values[0], n) || values[0].Size != n*4 {
			return fmt.Errorf("PTX handoff bound extent")
		}
		buffers[id] = values[0]
	}
	for id, a := range buffers {
		an, _ := t.Tensors[id].Elements()
		for other, b := range buffers {
			if other < id {
				bn, _ := t.Tensors[other].Elements()
				if trainingOverlap(a, an, b, bn) {
					return fmt.Errorf("PTX handoff bound overlap")
				}
			}
		}
	}
	return nil
}
