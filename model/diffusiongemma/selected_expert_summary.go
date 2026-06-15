package diffusiongemma

import "fmt"

type SelectedExpertGroupedSummary struct {
	WorkItems     int
	ActiveExperts int
	MaxGroup      int
	MinGroup      int
}

func SummarizeSelectedExpertGroupedWork(g SelectedExpertGroupedWork, workLen int) (SelectedExpertGroupedSummary, error) {
	if err := g.Validate(workLen); err != nil {
		return SelectedExpertGroupedSummary{}, err
	}
	out := SelectedExpertGroupedSummary{WorkItems: workLen, ActiveExperts: len(g.ActiveExperts)}
	if len(g.ActiveExperts) == 0 {
		return out, nil
	}
	out.MinGroup = int(^uint(0) >> 1)
	for i := 0; i < len(g.ActiveExperts); i++ {
		n := g.Offsets[i+1] - g.Offsets[i]
		if n < 0 {
			return SelectedExpertGroupedSummary{}, fmt.Errorf("negative group size at %d", i)
		}
		if n > out.MaxGroup {
			out.MaxGroup = n
		}
		if n < out.MinGroup {
			out.MinGroup = n
		}
	}
	return out, nil
}
