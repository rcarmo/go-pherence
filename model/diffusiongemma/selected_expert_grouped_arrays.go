package diffusiongemma

import "fmt"

type SelectedExpertGroupedArrays struct {
	WorkPositions  []int
	WorkPositionsU []uint32
	WorkWeights    []float32
	WorkDownScales []float32
	WorkSlots      []int
	WorkSlotsU     []uint32
	WorkActive     []int
	WorkActiveU    []uint32
	ActiveExperts  []int
	ActiveExpertsU []uint32
	Offsets        []int
	OffsetsU       []uint32
}

func BuildSelectedExpertGroupedArrays(arr SelectedExpertWorkArrays, grouped SelectedExpertGroupedWork) (SelectedExpertGroupedArrays, error) {
	if err := arr.Validate(); err != nil {
		return SelectedExpertGroupedArrays{}, err
	}
	if err := grouped.Validate(arr.Len()); err != nil {
		return SelectedExpertGroupedArrays{}, err
	}
	out := SelectedExpertGroupedArrays{
		WorkPositions:  make([]int, len(grouped.WorkOrder)),
		WorkPositionsU: make([]uint32, len(grouped.WorkOrder)),
		WorkWeights:    make([]float32, len(grouped.WorkOrder)),
		WorkDownScales: make([]float32, len(grouped.WorkOrder)),
		WorkSlots:      make([]int, len(grouped.WorkOrder)),
		WorkSlotsU:     make([]uint32, len(grouped.WorkOrder)),
		WorkActive:     make([]int, len(grouped.WorkOrder)),
		WorkActiveU:    make([]uint32, len(grouped.WorkOrder)),
		ActiveExperts:  append([]int(nil), grouped.ActiveExperts...),
		ActiveExpertsU: make([]uint32, len(grouped.ActiveExperts)),
		Offsets:        append([]int(nil), grouped.Offsets...),
		OffsetsU:       make([]uint32, len(grouped.Offsets)),
	}
	for g := range grouped.ActiveExperts {
		for i := grouped.Offsets[g]; i < grouped.Offsets[g+1]; i++ {
			out.WorkActive[i] = g
			out.WorkActiveU[i] = uint32(g)
		}
	}
	for i, workIdx := range grouped.WorkOrder {
		if workIdx < 0 || workIdx >= arr.Len() {
			return SelectedExpertGroupedArrays{}, fmt.Errorf("grouped work index %d outside [0,%d)", workIdx, arr.Len())
		}
		out.WorkPositions[i] = arr.Positions[workIdx]
		out.WorkPositionsU[i] = uint32(arr.Positions[workIdx])
		out.WorkWeights[i] = arr.Weights[workIdx]
		out.WorkDownScales[i] = 1
		out.WorkSlots[i] = arr.Slots[workIdx]
		out.WorkSlotsU[i] = uint32(arr.Slots[workIdx])
	}
	for i, expert := range grouped.ActiveExperts {
		out.ActiveExpertsU[i] = uint32(expert)
	}
	for i, off := range grouped.Offsets {
		out.OffsetsU[i] = uint32(off)
	}
	return out, nil
}

func (a *SelectedExpertGroupedArrays) ApplyDownScalesByExpert(downScale []float32) error {
	if a == nil {
		return fmt.Errorf("nil grouped arrays")
	}
	if len(downScale) == 0 {
		return nil
	}
	if len(a.WorkDownScales) != len(a.WorkWeights) {
		return fmt.Errorf("grouped arrays down-scale len=%d want %d", len(a.WorkDownScales), len(a.WorkWeights))
	}
	for groupIdx, expert := range a.ActiveExperts {
		if expert < 0 || expert >= len(downScale) {
			return fmt.Errorf("active expert %d outside down-scale len=%d", expert, len(downScale))
		}
		for i := a.Offsets[groupIdx]; i < a.Offsets[groupIdx+1]; i++ {
			a.WorkDownScales[i] = downScale[expert]
		}
	}
	return nil
}

func (a SelectedExpertGroupedArrays) Validate() error {
	n := len(a.WorkPositions)
	if len(a.WorkPositionsU) != n || len(a.WorkWeights) != n || len(a.WorkDownScales) != n || len(a.WorkSlots) != n || len(a.WorkSlotsU) != n || len(a.WorkActive) != n || len(a.WorkActiveU) != n {
		return fmt.Errorf("grouped arrays work length mismatch positions=%d positions_u=%d weights=%d down_scales=%d slots=%d slots_u=%d", len(a.WorkPositions), len(a.WorkPositionsU), len(a.WorkWeights), len(a.WorkDownScales), len(a.WorkSlots), len(a.WorkSlotsU))
	}
	g := len(a.ActiveExperts)
	if len(a.ActiveExpertsU) != g || len(a.Offsets) != g+1 || len(a.OffsetsU) != g+1 {
		return fmt.Errorf("grouped arrays group length mismatch active=%d active_u=%d offsets=%d offsets_u=%d", len(a.ActiveExperts), len(a.ActiveExpertsU), len(a.Offsets), len(a.OffsetsU))
	}
	if len(a.Offsets) > 0 && (a.Offsets[0] != 0 || a.Offsets[len(a.Offsets)-1] != n) {
		return fmt.Errorf("grouped arrays offset endpoints invalid offsets=%v work=%d", a.Offsets, n)
	}
	for i := 1; i < len(a.Offsets); i++ {
		if a.Offsets[i] < a.Offsets[i-1] {
			return fmt.Errorf("grouped arrays offsets not monotonic: %v", a.Offsets)
		}
	}
	return nil
}
