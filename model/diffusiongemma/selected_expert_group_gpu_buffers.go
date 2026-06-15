package diffusiongemma

import (
	"fmt"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

type SelectedExpertGroupedWorkGPUBuffers struct {
	WorkOrder     *gpu.Buffer
	ActiveExperts *gpu.Buffer
	Offsets       *gpu.Buffer
	WorkCapacity  int
	GroupCapacity int
}

func (b *SelectedExpertGroupedWorkGPUBuffers) Free() {
	if b == nil {
		return
	}
	if b.WorkOrder != nil {
		b.WorkOrder.Free()
		b.WorkOrder = nil
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

func (b *SelectedExpertGroupedWorkGPUBuffers) Ensure(workLen, groups int) error {
	if workLen <= 0 || groups <= 0 {
		return fmt.Errorf("invalid grouped work buffer sizes work=%d groups=%d", workLen, groups)
	}
	if b.WorkCapacity >= workLen && b.GroupCapacity >= groups && b.WorkOrder != nil && b.ActiveExperts != nil && b.Offsets != nil {
		return nil
	}
	b.Free()
	var err error
	if b.WorkOrder, err = gpu.MallocBytes(workLen * 4); err != nil {
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

func (b *SelectedExpertGroupedWorkGPUBuffers) Upload(g SelectedExpertGroupedWork, workLen int) error {
	if err := g.Validate(workLen); err != nil {
		return err
	}
	if workLen == 0 || len(g.ActiveExperts) == 0 {
		return nil
	}
	if err := b.Ensure(workLen, len(g.ActiveExperts)); err != nil {
		return err
	}
	workOrder := make([]uint32, len(g.WorkOrder))
	active := make([]uint32, len(g.ActiveExperts))
	offsets := make([]uint32, len(g.Offsets))
	for i, v := range g.WorkOrder {
		workOrder[i] = uint32(v)
	}
	for i, v := range g.ActiveExperts {
		active[i] = uint32(v)
	}
	for i, v := range g.Offsets {
		offsets[i] = uint32(v)
	}
	if err := b.WorkOrder.UploadUint32(workOrder); err != nil {
		return err
	}
	if err := b.ActiveExperts.UploadUint32(active); err != nil {
		return err
	}
	if err := b.Offsets.UploadUint32(offsets); err != nil {
		return err
	}
	return nil
}
