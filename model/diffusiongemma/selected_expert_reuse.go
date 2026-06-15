package diffusiongemma

import "fmt"

func FlattenSelectedExpertsInto(dst []SelectedExpertWorkItem, topKIDs []int, topKVals []float32, positions, topK, numExperts int) ([]SelectedExpertWorkItem, error) {
	if positions <= 0 || topK <= 0 || numExperts <= 0 {
		return nil, fmt.Errorf("invalid selected expert shape positions=%d topK=%d experts=%d", positions, topK, numExperts)
	}
	need := positions * topK
	if len(topKIDs) < need || len(topKVals) < need {
		return nil, fmt.Errorf("selected expert buffers too small ids=%d vals=%d need=%d", len(topKIDs), len(topKVals), need)
	}
	if cap(dst) < need {
		dst = make([]SelectedExpertWorkItem, 0, need)
	} else {
		dst = dst[:0]
	}
	for pos := 0; pos < positions; pos++ {
		for slot := 0; slot < topK; slot++ {
			i := pos*topK + slot
			expert := topKIDs[i]
			if expert < 0 {
				continue
			}
			if expert >= numExperts {
				return nil, fmt.Errorf("selected expert id %d outside [0,%d) at position=%d slot=%d", expert, numExperts, pos, slot)
			}
			dst = append(dst, SelectedExpertWorkItem{Position: pos, Expert: expert, Slot: slot, Weight: topKVals[i]})
		}
	}
	return dst, nil
}

func BuildSelectedExpertWorkArraysInto(out *SelectedExpertWorkArrays, items []SelectedExpertWorkItem) {
	if cap(out.Positions) < len(items) {
		out.Positions = make([]int, len(items))
	} else {
		out.Positions = out.Positions[:len(items)]
	}
	if cap(out.Experts) < len(items) {
		out.Experts = make([]int, len(items))
	} else {
		out.Experts = out.Experts[:len(items)]
	}
	if cap(out.Slots) < len(items) {
		out.Slots = make([]int, len(items))
	} else {
		out.Slots = out.Slots[:len(items)]
	}
	if cap(out.Weights) < len(items) {
		out.Weights = make([]float32, len(items))
	} else {
		out.Weights = out.Weights[:len(items)]
	}
	if cap(out.PositionsU) < len(items) {
		out.PositionsU = make([]uint32, len(items))
	} else {
		out.PositionsU = out.PositionsU[:len(items)]
	}
	if cap(out.ExpertsU) < len(items) {
		out.ExpertsU = make([]uint32, len(items))
	} else {
		out.ExpertsU = out.ExpertsU[:len(items)]
	}
	if cap(out.SlotsU) < len(items) {
		out.SlotsU = make([]uint32, len(items))
	} else {
		out.SlotsU = out.SlotsU[:len(items)]
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
}
