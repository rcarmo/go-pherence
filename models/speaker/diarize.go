package speaker

import "math"

// VADSegment represents a voice-active segment.
type VADSegment struct {
	Start float64 // seconds
	End   float64 // seconds
}

// EnergyVAD performs simple energy-based voice activity detection.
// Returns segments where speech energy exceeds the threshold.
func EnergyVAD(samples []float32, sampleRate int, frameMs, hopMs int, threshold float32) []VADSegment {
	if len(samples) == 0 || sampleRate <= 0 {
		return nil
	}

	frameSize := sampleRate * frameMs / 1000
	hopSize := sampleRate * hopMs / 1000
	if frameSize <= 0 || hopSize <= 0 {
		return nil
	}

	numFrames := (len(samples) - frameSize) / hopSize
	if numFrames <= 0 {
		return nil
	}

	// Compute per-frame energy
	energies := make([]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		offset := i * hopSize
		var sum float32
		for j := 0; j < frameSize && offset+j < len(samples); j++ {
			v := samples[offset+j]
			sum += v * v
		}
		energies[i] = sum / float32(frameSize)
	}

	// If threshold is 0, auto-compute from median
	if threshold <= 0 {
		threshold = autoThreshold(energies)
	}

	// Find contiguous active regions
	var segments []VADSegment
	inSpeech := false
	var startFrame int

	for i, e := range energies {
		if e >= threshold && !inSpeech {
			inSpeech = true
			startFrame = i
		} else if e < threshold && inSpeech {
			inSpeech = false
			segments = append(segments, VADSegment{
				Start: float64(startFrame*hopSize) / float64(sampleRate),
				End:   float64(i*hopSize) / float64(sampleRate),
			})
		}
	}
	if inSpeech {
		segments = append(segments, VADSegment{
			Start: float64(startFrame*hopSize) / float64(sampleRate),
			End:   float64(len(samples)) / float64(sampleRate),
		})
	}

	// Merge segments closer than 300ms
	segments = mergeClose(segments, 0.3)

	// Remove segments shorter than 200ms
	var filtered []VADSegment
	for _, s := range segments {
		if s.End-s.Start >= 0.2 {
			filtered = append(filtered, s)
		}
	}

	return filtered
}

func autoThreshold(energies []float32) float32 {
	if len(energies) == 0 {
		return 0
	}
	// Use mean energy * 0.5 as threshold
	var sum float64
	for _, e := range energies {
		sum += float64(e)
	}
	mean := sum / float64(len(energies))
	return float32(mean * 0.5)
}

func mergeClose(segs []VADSegment, gap float64) []VADSegment {
	if len(segs) <= 1 {
		return segs
	}
	merged := []VADSegment{segs[0]}
	for _, s := range segs[1:] {
		last := &merged[len(merged)-1]
		if s.Start-last.End < gap {
			last.End = s.End
		} else {
			merged = append(merged, s)
		}
	}
	return merged
}

// CosineSimilarity computes cosine similarity between two embeddings.
func CosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		if i >= len(b) {
			break
		}
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return float32(dot / denom)
}

// AgglomerativeCluster performs agglomerative clustering on embeddings
// using cosine distance, stopping when the minimum similarity drops below threshold.
// Returns cluster labels (0-indexed) for each embedding.
func AgglomerativeCluster(embeddings [][]float32, threshold float32) []int {
	n := len(embeddings)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return []int{0}
	}

	// Initialize: each embedding is its own cluster
	labels := make([]int, n)
	for i := range labels {
		labels[i] = i
	}

	// Compute pairwise similarity matrix (upper triangle)
	sim := make([][]float32, n)
	for i := range sim {
		sim[i] = make([]float32, n)
		for j := i + 1; j < n; j++ {
			sim[i][j] = CosineSimilarity(embeddings[i], embeddings[j])
		}
	}

	active := make([]bool, n)
	for i := range active {
		active[i] = true
	}

	for {
		// Find most similar pair
		bestI, bestJ := -1, -1
		bestSim := float32(-2)
		for i := 0; i < n; i++ {
			if !active[i] {
				continue
			}
			for j := i + 1; j < n; j++ {
				if !active[j] {
					continue
				}
				if sim[i][j] > bestSim {
					bestSim = sim[i][j]
					bestI = i
					bestJ = j
				}
			}
		}

		if bestI < 0 || bestSim < threshold {
			break // No more merges
		}

		// Merge j into i
		active[bestJ] = false
		for k := 0; k < n; k++ {
			if labels[k] == labels[bestJ] {
				labels[k] = labels[bestI]
			}
		}

		// Update similarity: average linkage
		for k := 0; k < n; k++ {
			if !active[k] || k == bestI {
				continue
			}
			s1 := getSim(sim, bestI, k)
			s2 := getSim(sim, bestJ, k)
			setSim(sim, bestI, k, (s1+s2)/2)
		}
	}

	// Renumber labels contiguously
	remap := make(map[int]int)
	next := 0
	for i := range labels {
		if _, ok := remap[labels[i]]; !ok {
			remap[labels[i]] = next
			next++
		}
		labels[i] = remap[labels[i]]
	}

	return labels
}

func getSim(sim [][]float32, i, j int) float32 {
	if i > j {
		i, j = j, i
	}
	return sim[i][j]
}

func setSim(sim [][]float32, i, j int, v float32) {
	if i > j {
		i, j = j, i
	}
	sim[i][j] = v
}
