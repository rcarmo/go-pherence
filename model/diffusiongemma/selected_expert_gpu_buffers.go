package diffusiongemma

import (
	"fmt"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

type SelectedExpertWorkGPUBuffers struct {
	Positions *gpu.Buffer
	Experts   *gpu.Buffer
	Slots     *gpu.Buffer
	Weights   *gpu.Buffer
	Capacity  int
}

func (b *SelectedExpertWorkGPUBuffers) Free() {
	if b == nil {
		return
	}
	if b.Positions != nil {
		b.Positions.Free()
		b.Positions = nil
	}
	if b.Experts != nil {
		b.Experts.Free()
		b.Experts = nil
	}
	if b.Slots != nil {
		b.Slots.Free()
		b.Slots = nil
	}
	if b.Weights != nil {
		b.Weights.Free()
		b.Weights = nil
	}
	b.Capacity = 0
}

func (b *SelectedExpertWorkGPUBuffers) Ensure(n int) error {
	if n <= 0 {
		return fmt.Errorf("invalid selected expert work buffer size %d", n)
	}
	if b.Capacity >= n && b.Positions != nil && b.Experts != nil && b.Slots != nil && b.Weights != nil {
		return nil
	}
	b.Free()
	var err error
	if b.Positions, err = gpu.MallocBytes(n * 4); err != nil {
		b.Free()
		return err
	}
	if b.Experts, err = gpu.MallocBytes(n * 4); err != nil {
		b.Free()
		return err
	}
	if b.Slots, err = gpu.MallocBytes(n * 4); err != nil {
		b.Free()
		return err
	}
	if b.Weights, err = gpu.Malloc(n); err != nil {
		b.Free()
		return err
	}
	b.Capacity = n
	return nil
}

func (b *SelectedExpertWorkGPUBuffers) Upload(arr SelectedExpertWorkArrays) error {
	if err := arr.Validate(); err != nil {
		return err
	}
	n := arr.Len()
	if n == 0 {
		return nil
	}
	if err := b.Ensure(n); err != nil {
		return err
	}
	if err := b.Positions.UploadUint32(arr.PositionsU); err != nil {
		return err
	}
	if err := b.Experts.UploadUint32(arr.ExpertsU); err != nil {
		return err
	}
	if err := b.Slots.UploadUint32(arr.SlotsU); err != nil {
		return err
	}
	if err := b.Weights.Upload(arr.Weights); err != nil {
		return err
	}
	return nil
}
