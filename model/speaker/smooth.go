package speaker

// SmoothSingletonLabels conservatively reassigns singleton speaker labels to the
// most similar non-singleton cluster when their average embedding cosine is high
// enough. It is intended as a short-segment stabilizer after agglomerative
// clustering, not as a substitute for a proper diarization backend.
func SmoothSingletonLabels(labels []int, embeddings [][]float32, minAvgSim float32) []int {
	if len(labels) == 0 || len(labels) != len(embeddings) {
		return labels
	}
	out := append([]int(nil), labels...)
	for pass := 0; pass < len(out); pass++ {
		changed := false
		counts := map[int]int{}
		for _, label := range out {
			counts[label]++
		}
		for i, label := range out {
			if counts[label] != 1 {
				continue
			}
			bestLabel := label
			bestSim := float32(-2)
			for candidate, count := range counts {
				if candidate == label || count == 0 {
					continue
				}
				var sum float32
				var n int
				for j, other := range out {
					if other != candidate || i == j {
						continue
					}
					sum += CosineSimilarity(embeddings[i], embeddings[j])
					n++
				}
				if n == 0 {
					continue
				}
				avg := sum / float32(n)
				if avg > bestSim {
					bestSim = avg
					bestLabel = candidate
				}
			}
			if bestLabel != label && bestSim >= minAvgSim {
				out[i] = bestLabel
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return RenumberLabels(out)
}

// RenumberLabels maps arbitrary labels to contiguous IDs in first-seen order.
func RenumberLabels(labels []int) []int {
	remap := map[int]int{}
	next := 0
	out := make([]int, len(labels))
	for i, label := range labels {
		mapped, ok := remap[label]
		if !ok {
			mapped = next
			remap[label] = mapped
			next++
		}
		out[i] = mapped
	}
	return out
}
