package diffusiongemma

import "fmt"

type SelectedExpertGroupedWork struct {
	// WorkOrder stores indices into the original flat arrays, grouped by expert.
	WorkOrder []int
	// ActiveExperts stores expert IDs with at least one selected work item.
	ActiveExperts []int
	// Offsets has len ActiveExperts+1; items for active expert i are
	// WorkOrder[Offsets[i]:Offsets[i+1]].
	Offsets []int
}

func BuildSelectedExpertGroupedWork(arr SelectedExpertWorkArrays, numExperts int) (SelectedExpertGroupedWork, error) {
	if err := arr.Validate(); err != nil {
		return SelectedExpertGroupedWork{}, err
	}
	if numExperts <= 0 {
		return SelectedExpertGroupedWork{}, fmt.Errorf("invalid expert count %d", numExperts)
	}
	counts := make([]int, numExperts)
	for _, expert := range arr.Experts {
		if expert < 0 || expert >= numExperts {
			return SelectedExpertGroupedWork{}, fmt.Errorf("selected expert id %d outside [0,%d)", expert, numExperts)
		}
		counts[expert]++
	}
	active := make([]int, 0, numExperts)
	offsets := []int{0}
	for expert, n := range counts {
		if n == 0 {
			continue
		}
		active = append(active, expert)
		offsets = append(offsets, offsets[len(offsets)-1]+n)
	}
	order := make([]int, arr.Len())
	cursor := make([]int, len(offsets)-1)
	copy(cursor, offsets[:len(offsets)-1])
	activeIndex := make([]int, numExperts)
	for i := range activeIndex {
		activeIndex[i] = -1
	}
	for i, expert := range active {
		activeIndex[expert] = i
	}
	for workIdx, expert := range arr.Experts {
		ai := activeIndex[expert]
		if ai < 0 {
			return SelectedExpertGroupedWork{}, fmt.Errorf("selected expert id %d missing active index", expert)
		}
		order[cursor[ai]] = workIdx
		cursor[ai]++
	}
	return SelectedExpertGroupedWork{WorkOrder: order, ActiveExperts: active, Offsets: offsets}, nil
}

func (g SelectedExpertGroupedWork) Validate(workLen int) error {
	if len(g.Offsets) != len(g.ActiveExperts)+1 {
		return fmt.Errorf("grouped expert offsets len=%d want active+1=%d", len(g.Offsets), len(g.ActiveExperts)+1)
	}
	if len(g.WorkOrder) != workLen {
		return fmt.Errorf("grouped expert work order len=%d want %d", len(g.WorkOrder), workLen)
	}
	if len(g.Offsets) == 0 || g.Offsets[0] != 0 || g.Offsets[len(g.Offsets)-1] != workLen {
		return fmt.Errorf("grouped expert offsets endpoints invalid offsets=%v workLen=%d", g.Offsets, workLen)
	}
	for i := 1; i < len(g.Offsets); i++ {
		if g.Offsets[i] < g.Offsets[i-1] {
			return fmt.Errorf("grouped expert offsets not monotonic: %v", g.Offsets)
		}
	}
	return nil
}
