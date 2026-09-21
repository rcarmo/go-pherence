package main

import (
	"fmt"
	"math"
)

// Bound local diagnostic controls before model loading or byte conversions.
// These are planner limits, not permission to reserve that much physical memory.
func validateWorkloadFlags(steps, drafts, chunk, repeat, prefill int, acceptance, tolerance float64, budgetsMiB []int) error {
	if steps < 1 || steps > 65536 || drafts < 1 || drafts > 4096 || chunk < 1 || chunk > 65536 || repeat < 1 || repeat > 1024 || prefill < 0 || prefill > 65536 {
		return fmt.Errorf("invalid Qwen step/chunk/repeat limits")
	}
	if math.IsNaN(acceptance) || math.IsInf(acceptance, 0) || acceptance < 0 || acceptance > 1 || math.IsNaN(tolerance) || math.IsInf(tolerance, 0) || tolerance < 0 || tolerance > math.MaxFloat32 {
		return fmt.Errorf("invalid Qwen acceptance/tolerance")
	}
	for _, mib := range budgetsMiB {
		if mib < 0 || int64(mib) > math.MaxInt64/(1<<20) || mib > int(^uint(0)>>1)/(1<<20) {
			return fmt.Errorf("invalid/overflowing MiB budget")
		}
	}
	return nil
}
