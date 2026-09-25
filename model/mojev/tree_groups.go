package mojev

// candidateTreeEnd greedily fills one scratch-sized group. Callers validate
// nonempty candidates and prefix+len(candidate)<=capacity first, so every call
// advances at least one candidate. Grouping never changes local positions.
func candidateTreeEnd(candidates [][]int, first, prefix, capacity int) int {
	end, total := first, prefix
	for end < len(candidates) && len(candidates[end]) <= capacity-total {
		total += len(candidates[end])
		end++
	}
	return end
}
