package pockettts

import (
	"fmt"
	"math"
	"sort"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Exact logical resident payload for this tape-only shared owner. Native
// alignment/driver requirements remain enforced by the global Vulkan budget.
func (b *ResidentTrainingBackend) stagedResidentPayload() (int64, error) {
	var elements, maxElements, maxStats int64
	if len(b.host.Params) == 0 {
		return 0, fmt.Errorf("stage resident inventory missing")
	}
	for _, p := range b.host.Params {
		n := int64(len(p.Values))
		if n < 1 || n > 1<<26 || elements > math.MaxInt64-n {
			return 0, fmt.Errorf("stage resident extent/overflow")
		}
		elements += n
		maxElements = max(maxElements, n)
		stats, err := train.NormStatsShape(int(n))
		if err != nil {
			return 0, err
		}
		maxStats = max(maxStats, int64(stats))
	}
	if elements > (math.MaxInt64-8*maxElements-4*maxStats-48)/20 {
		return 0, fmt.Errorf("stage resident byte overflow")
	}
	return 20*elements + 8*maxElements + 4*maxStats + 48, nil
}

// Full source/inventory is admitted before pruning. This preflights every
// actually executed stage root set against joint resident + owned slot payload;
// borrowed parameter storage is charged once in the resident owner above.
func (b *ResidentTrainingBackend) admitStagedStorage(source train.Tape, inventory map[string]train.ResidentTapeBinding, boundaries []int, groups bool) error {
	resident, err := b.stagedResidentPayload()
	if err != nil {
		return err
	}
	tape, bindings, compact, err := compactResidentEmbeddingTape(source, inventory, b.MaxTapeBytes)
	if err != nil {
		return err
	}
	if compact == nil {
		err = train.ValidateResidentTapeBindings(tape, bindings, b.MaxTapeBytes)
	} else {
		err = train.ValidateResidentCompactTapeBindings(tape, bindings, compact, b.MaxTapeBytes)
	}
	if err != nil {
		return err
	}
	limit := min(b.MaxResidentBytes, int64(2<<30))
	if resident >= limit {
		return fmt.Errorf("stage joint resident payload exceeds ceiling")
	}
	check := func(roots []int) error {
		pruned, mapping, err := train.PruneTape(tape, roots, b.MaxTapeBytes)
		if err != nil {
			return err
		}
		protected := map[int]bool{}
		for _, binding := range bindings {
			if mapping[binding.Parameter] >= 0 {
				protected[mapping[binding.Parameter]] = true
			}
		}
		// These staged calls have no external producers; do not silently deduct a
		// borrowed external input whose owner is absent from the resident ledger.
		for _, s := range tape.Tensors {
			if s.External {
				return fmt.Errorf("staged external source requires explicit owner accounting")
			}
		}
		plan, err := train.PlanZeroTapeStorage(pruned, b.MaxTapeBytes, protected)
		if err != nil {
			return err
		}
		borrowed := map[int]bool{}
		for id := range protected {
			borrowed[plan.Slots[id]] = true
		}
		var owned int64
		for slot, n := range plan.Elements {
			if !borrowed[slot] {
				bytes := int64(n) * 4
				if bytes > limit-resident-owned {
					return fmt.Errorf("stage joint native payload ceiling: resident=%d owned>%d cap=%d", resident, limit-resident, limit)
				}
				owned += bytes
			}
		}
		return nil
	}
	if len(boundaries) > 0 {
		if err = check(boundaries); err != nil {
			return err
		}
	}
	if groups {
		names := make([]string, 0, len(bindings))
		for name := range bindings {
			names = append(names, name)
		}
		sort.Strings(names)
		size := stepBackboneGradientGroupSize(tape)
		for start := 0; start < len(names); start += size {
			roots := make([]int, 0, 2*size)
			for _, name := range names[start:min(start+size, len(names))] {
				roots = append(roots, bindings[name].Gradient)
				if c, ok := compact[name]; ok {
					roots = append(roots, c.Index)
				}
			}
			if err = check(roots); err != nil {
				return err
			}
		}
	}
	return nil
}
