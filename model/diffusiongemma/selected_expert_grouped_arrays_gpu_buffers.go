package diffusiongemma

import (
	"fmt"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

type SelectedExpertGroupedArraysGPUBuffers struct {
	WorkPositions    *gpu.Buffer
	WorkWeights      *gpu.Buffer
	WorkDownScales   *gpu.Buffer
	EffectiveWeights *gpu.Buffer
	WorkSlots        *gpu.Buffer
	WorkActive       *gpu.Buffer
	ActiveExperts    *gpu.Buffer
	Offsets          *gpu.Buffer
	WorkCapacity     int
	GroupCapacity    int
}

func (b *SelectedExpertGroupedArraysGPUBuffers) Free() {
	if b == nil {
		return
	}
	if b.WorkPositions != nil {
		b.WorkPositions.Free()
		b.WorkPositions = nil
	}
	if b.WorkWeights != nil {
		b.WorkWeights.Free()
		b.WorkWeights = nil
	}
	if b.WorkDownScales != nil {
		b.WorkDownScales.Free()
		b.WorkDownScales = nil
	}
	if b.EffectiveWeights != nil {
		b.EffectiveWeights.Free()
		b.EffectiveWeights = nil
	}
	if b.WorkSlots != nil {
		b.WorkSlots.Free()
		b.WorkSlots = nil
	}
	if b.WorkActive != nil {
		b.WorkActive.Free()
		b.WorkActive = nil
	}
	if b.ActiveExperts != nil {
		b.ActiveExperts.Free()
		b.ActiveExperts = nil
	}
	if b.Offsets != nil {
		b.Offsets.Free()
		b.Offsets = nil
	}
	b.WorkCapacity = 0
	b.GroupCapacity = 0
}

func (b *SelectedExpertGroupedArraysGPUBuffers) Ensure(workLen, groups int) error {
	if workLen <= 0 || groups <= 0 {
		return fmt.Errorf("invalid grouped array buffer sizes work=%d groups=%d", workLen, groups)
	}
	if b.WorkCapacity >= workLen && b.GroupCapacity >= groups && b.WorkPositions != nil && b.WorkWeights != nil && b.WorkDownScales != nil && b.EffectiveWeights != nil && b.WorkSlots != nil && b.WorkActive != nil && b.ActiveExperts != nil && b.Offsets != nil {
		return nil
	}
	b.Free()
	var err error
	if b.WorkPositions, err = gpu.MallocBytes(workLen * 4); err != nil {
		b.Free()
		return err
	}
	if b.WorkWeights, err = gpu.Malloc(workLen); err != nil {
		b.Free()
		return err
	}
	if b.WorkDownScales, err = gpu.Malloc(workLen); err != nil {
		b.Free()
		return err
	}
	if b.EffectiveWeights, err = gpu.Malloc(workLen); err != nil {
		b.Free()
		return err
	}
	if b.WorkSlots, err = gpu.MallocBytes(workLen * 4); err != nil {
		b.Free()
		return err
	}
	if b.WorkActive, err = gpu.MallocBytes(workLen * 4); err != nil {
		b.Free()
		return err
	}
	if b.ActiveExperts, err = gpu.MallocBytes(groups * 4); err != nil {
		b.Free()
		return err
	}
	if b.Offsets, err = gpu.MallocBytes((groups + 1) * 4); err != nil {
		b.Free()
		return err
	}
	b.WorkCapacity = workLen
	b.GroupCapacity = groups
	return nil
}

func (b *SelectedExpertGroupedArraysGPUBuffers) Upload(a SelectedExpertGroupedArrays) error {
	if err := a.Validate(); err != nil {
		return err
	}
	workLen, groups := len(a.WorkPositions), len(a.ActiveExperts)
	if workLen == 0 || groups == 0 {
		return nil
	}
	if err := b.Ensure(workLen, groups); err != nil {
		return err
	}
	if err := b.WorkPositions.UploadUint32(a.WorkPositionsU); err != nil {
		return err
	}
	if err := b.WorkWeights.Upload(a.WorkWeights); err != nil {
		return err
	}
	if err := b.WorkDownScales.Upload(a.WorkDownScales); err != nil {
		return err
	}
	if err := gpu.MulWeights(b.EffectiveWeights, b.WorkWeights, b.WorkDownScales, workLen); err != nil {
		return err
	}
	if err := b.WorkSlots.UploadUint32(a.WorkSlotsU); err != nil {
		return err
	}
	if err := b.WorkActive.UploadUint32(a.WorkActiveU); err != nil {
		return err
	}
	if err := b.ActiveExperts.UploadUint32(a.ActiveExpertsU); err != nil {
		return err
	}
	if err := b.Offsets.UploadUint32(a.OffsetsU); err != nil {
		return err
	}
	return nil
}
