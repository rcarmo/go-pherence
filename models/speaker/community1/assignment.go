// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT
// Community-1 assignment policy follows pyannote; solver attribution in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
)

// AssignmentConfig describes scores [Chunks,Speakers,Clusters]. Bounds:
// chunks1..4096, speakers1..8, clusters1..64. -2 means unassigned when there are
// fewer clusters than local speakers. No persistent speaker name is assigned.
type AssignmentConfig struct{ Chunks, Speakers, Clusters int }

// ConstrainedSpeakerAssignment maximises the sum of one-to-one scores per
// chunk, matching BaseClustering.constrained_argmax. NaNs are replaced by the
// GLOBAL finite minimum across chunks before matching. All-NaN, Inf and finite
// |score|>1e100 inputs fail explicitly, unlike upstream's Inf replacement.
// Square, wide and tall matrices preserve the pinned SciPy solver's tie rules.
// The solver works in Go on a bounded per-window copy; source scores are never
// modified. Returns owned [Chunks,Speakers] labels or nil on error/cancellation.
func ConstrainedSpeakerAssignment(ctx context.Context, scores []float64, cfg AssignmentConfig) ([]int, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg
	if !validAssignmentConfig(c) || len(scores) != c.Chunks*c.Speakers*c.Clusters {
		return nil, fmt.Errorf("invalid constrained assignment geometry")
	}
	minimum := math.Inf(1)
	for i, value := range scores {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if math.IsInf(value, 0) || math.Abs(value) > 1e100 {
			return nil, fmt.Errorf("invalid constrained score")
		}
		if !math.IsNaN(value) && value < minimum {
			minimum = value
		}
	}
	if math.IsInf(minimum, 1) {
		return nil, fmt.Errorf("all constrained scores are NaN")
	}
	labels := make([]int, c.Chunks*c.Speakers)
	for i := range labels {
		labels[i] = -2
	}
	var workspace assignmentWorkspace
	for chunk := 0; chunk < c.Chunks; chunk++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		offset := chunk * c.Speakers * c.Clusters
		if err := workspace.solve(ctx, scores[offset:offset+c.Speakers*c.Clusters], c.Speakers, c.Clusters, minimum, labels[chunk*c.Speakers:(chunk+1)*c.Speakers]); err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return labels, nil
}
func validAssignmentConfig(c AssignmentConfig) bool {
	return c.Chunks >= 1 && c.Chunks <= 4096 && c.Speakers >= 1 && c.Speakers <= 8 && c.Clusters >= 1 && c.Clusters <= 64
}

// CosineAssignmentConfig uses original embeddings [Chunks,Speakers,Dimension],
// centroids [Clusters,Dimension] and binary/NaN segmentations
// [Chunks,Frames,Speakers]. Dimension1..512, frames1..4096. Input elements per
// array<=2^24, dot work chunks*speakers*clusters*dimension<=2^27.
type CosineAssignmentConfig struct {
	Chunks, Speakers, Clusters, Dimension, Frames int
	Constrained                                   bool
}

// SpeakerAssignment owns original cosine-derived scores and resulting labels.
// In constrained mode scores include inactive-speaker suppression to GLOBAL
// min(score)-1. This penalty does not guarantee inactive rows are unassigned:
// in a square/wide matrix every row is assigned, exactly as upstream. A later
// pipeline must suppress inactive turns using segmentation, not labels alone.
type SpeakerAssignment struct {
	Scores []float64
	Labels []int
}

// AssignCosineSpeakers computes SciPy cosine distance clipped to [0,2], then
// scores=2-distance. Constrained mode sets rows with zero segmentation sum to
// min(scores)-1 before rectangular assignment. Unconstrained mode ignores that
// suppression and selects the first maximum per row. NaN segmentations make
// that row's sum unknown, not zero. Soft/Inf segmentations fail.
//
// Embeddings and centroids must be finite with finite NONZERO squared norms.
// NaN/zero embeddings are explicitly rejected instead of propagating undefined
// cosine through the pipeline. The lower-level constrained API independently
// supports partial-NaN score matrices. No L2-normalised copies are made. No
// metric switch, forced-count KMeans, AHC, reconstruction or turn timing here.
func AssignCosineSpeakers(ctx context.Context, embeddings, centroids []float64, segmentations []float32, cfg CosineAssignmentConfig) (*SpeakerAssignment, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg
	geometry := AssignmentConfig{c.Chunks, c.Speakers, c.Clusters}
	if !validAssignmentConfig(geometry) || c.Dimension < 1 || c.Dimension > 512 || c.Frames < 1 || c.Frames > 4096 {
		return nil, fmt.Errorf("invalid cosine assignment geometry")
	}
	rows := c.Chunks * c.Speakers
	if rows*c.Frames > 1<<24 || rows*c.Dimension > 1<<24 || len(embeddings) != rows*c.Dimension || len(centroids) != c.Clusters*c.Dimension || len(segmentations) != rows*c.Frames || int64(rows)*int64(c.Clusters)*int64(c.Dimension) > 1<<27 {
		return nil, fmt.Errorf("invalid cosine assignment input/work bound")
	}
	if err := finiteClustering64(ctx, embeddings); err != nil {
		return nil, err
	}
	if err := finiteClustering64(ctx, centroids); err != nil {
		return nil, err
	}
	if err := checkBinarySegmentations(ctx, segmentations); err != nil {
		return nil, err
	}
	centroidNorms := make([]float64, c.Clusters)
	for k := range centroidNorms {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := cosineNorm(centroids[k*c.Dimension : (k+1)*c.Dimension])
		if err != nil {
			return nil, err
		}
		centroidNorms[k] = n
	}
	result := &SpeakerAssignment{Scores: make([]float64, rows*c.Clusters)}
	minimum := math.Inf(1)
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		x := embeddings[row*c.Dimension : (row+1)*c.Dimension]
		norm, err := cosineNorm(x)
		if err != nil {
			return nil, err
		}
		for k := 0; k < c.Clusters; k++ {
			dot := float64(0)
			for d, value := range x {
				dot += value * centroids[k*c.Dimension+d]
			}
			denominator := norm * centroidNorms[k]
			similarity := dot / denominator
			if denominator == 0 || math.IsInf(denominator, 0) || math.IsNaN(similarity) || math.IsInf(similarity, 0) {
				return nil, fmt.Errorf("nonfinite cosine reduction")
			}
			// SciPy computes distance after clipping similarity magnitude to one.
			similarity = math.Max(-1, math.Min(1, similarity))
			score := 2 - (1 - similarity)
			result.Scores[row*c.Clusters+k] = score
			minimum = math.Min(minimum, score)
		}
	}
	if c.Constrained {
		for chunk := 0; chunk < c.Chunks; chunk++ {
			var support [8]float32
			for frame := 0; frame < c.Frames; frame++ {
				if frame%256 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				for s := 0; s < c.Speakers; s++ {
					support[s] += segmentations[(chunk*c.Frames+frame)*c.Speakers+s]
				}
			}
			for s := 0; s < c.Speakers; s++ {
				if support[s] == 0 {
					row := chunk*c.Speakers + s
					for k := 0; k < c.Clusters; k++ {
						result.Scores[row*c.Clusters+k] = minimum - 1
					}
				}
			}
		}
		labels, err := ConstrainedSpeakerAssignment(ctx, result.Scores, geometry)
		if err != nil {
			return nil, err
		}
		result.Labels = labels
	} else {
		result.Labels = make([]int, rows)
		for row := 0; row < rows; row++ {
			if row%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			best := 0
			for k := 1; k < c.Clusters; k++ {
				if result.Scores[row*c.Clusters+k] > result.Scores[row*c.Clusters+best] {
					best = k
				}
			}
			result.Labels[row] = best
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
func cosineNorm(values []float64) (float64, error) {
	sum := float64(0)
	for _, v := range values {
		sum += v * v
	}
	if sum <= 0 || math.IsInf(sum, 0) || math.IsNaN(sum) {
		return 0, fmt.Errorf("undefined cosine norm")
	}
	return math.Sqrt(sum), nil
}
