package diffusiongemma

import "fmt"

type SelectedExpertWorkArrays struct {
	Positions  []int
	Experts    []int
	Slots      []int
	Weights    []float32
	PositionsU []uint32
	ExpertsU   []uint32
	SlotsU     []uint32
}

func BuildSelectedExpertWorkArrays(items []SelectedExpertWorkItem) SelectedExpertWorkArrays {
	out := SelectedExpertWorkArrays{
		Positions:  make([]int, len(items)),
		Experts:    make([]int, len(items)),
		Slots:      make([]int, len(items)),
		Weights:    make([]float32, len(items)),
		PositionsU: make([]uint32, len(items)),
		ExpertsU:   make([]uint32, len(items)),
		SlotsU:     make([]uint32, len(items)),
	}
	for i, item := range items {
		out.Positions[i] = item.Position
		out.Experts[i] = item.Expert
		out.Slots[i] = item.Slot
		out.Weights[i] = item.Weight
		out.PositionsU[i] = uint32(item.Position)
		out.ExpertsU[i] = uint32(item.Expert)
		out.SlotsU[i] = uint32(item.Slot)
	}
	return out
}

func (a SelectedExpertWorkArrays) Len() int { return len(a.Positions) }

func (a SelectedExpertWorkArrays) Validate() error {
	n := len(a.Positions)
	if len(a.Experts) != n || len(a.Slots) != n || len(a.Weights) != n || len(a.PositionsU) != n || len(a.ExpertsU) != n || len(a.SlotsU) != n {
		return fmt.Errorf("selected expert work arrays length mismatch positions=%d experts=%d slots=%d weights=%d positions_u=%d experts_u=%d slots_u=%d", len(a.Positions), len(a.Experts), len(a.Slots), len(a.Weights), len(a.PositionsU), len(a.ExpertsU), len(a.SlotsU))
	}
	return nil
}
