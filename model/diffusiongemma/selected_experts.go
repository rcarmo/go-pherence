package diffusiongemma

import "fmt"

type SelectedExpertWorkItem struct {
	Position int
	Expert   int
	Slot     int
	Weight   float32
}

func FlattenSelectedExperts(topKIDs []int, topKVals []float32, positions, topK, numExperts int) ([]SelectedExpertWorkItem, error) {
	if positions <= 0 || topK <= 0 || numExperts <= 0 {
		return nil, fmt.Errorf("invalid selected expert shape positions=%d topK=%d experts=%d", positions, topK, numExperts)
	}
	need := positions * topK
	if len(topKIDs) < need || len(topKVals) < need {
		return nil, fmt.Errorf("selected expert buffers too small ids=%d vals=%d need=%d", len(topKIDs), len(topKVals), need)
	}
	items := make([]SelectedExpertWorkItem, 0, need)
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
			items = append(items, SelectedExpertWorkItem{Position: pos, Expert: expert, Slot: slot, Weight: topKVals[i]})
		}
	}
	return items, nil
}

func GroupSelectedExpertWork(items []SelectedExpertWorkItem, numExperts int) ([][]SelectedExpertWorkItem, error) {
	if numExperts <= 0 {
		return nil, fmt.Errorf("invalid expert count %d", numExperts)
	}
	groups := make([][]SelectedExpertWorkItem, numExperts)
	for _, item := range items {
		if item.Expert < 0 || item.Expert >= numExperts {
			return nil, fmt.Errorf("selected expert id %d outside [0,%d)", item.Expert, numExperts)
		}
		groups[item.Expert] = append(groups[item.Expert], item)
	}
	return groups, nil
}
