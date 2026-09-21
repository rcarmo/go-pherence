// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT
// VBxClustering centroid reduction follows the reference in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
)

// VBxCentroidConfig describes original (not PLDA-transformed or L2-normalised)
// embeddings [Rows,Dimension], responsibilities [Rows,Speakers] and priors
// [Speakers]. Bounds: rows1..4096, dimension1..512, speakers1..64, product<=2^27.
// All arrays are float64. Float32 embeddings can be widened exactly by callers.
type VBxCentroidConfig struct{ Rows, Dimension, Speakers int }

// VBxCentroids owns [retained,Dimension] unnormalised weighted means, selected
// original VBx slot indices in increasing order, and posterior weight sums.
// These compact row indices are the later assignment labels, not stable names.
type VBxCentroids struct {
	Centroids, WeightSums []float64
	SpeakerIndices        []int
}

// ComputeVBxCentroids implements W=q[:,priors>1e-7], W.T@embeddings/W.sum(0).
// Pruning is STRICT; equality to 1e-7 is removed. Priors/each posterior row must
// be finite probabilities summing to one within 1e-6. Retained columns must
// have positive support; no survivor or nonfinite reduction is an error rather
// than an empty/NaN centroid. Empty-training and forced-count fallbacks belong
// to a future pipeline. No additional normalisation, merging, copying expanded
// feature matrices or model work. Sources must stay immutable during the call.
func ComputeVBxCentroids(ctx context.Context, embeddings, q, priors []float64, cfg VBxCentroidConfig) (*VBxCentroids, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg
	if c.Rows < 1 || c.Rows > 4096 || c.Dimension < 1 || c.Dimension > 512 || c.Speakers < 1 || c.Speakers > 64 || len(embeddings) != c.Rows*c.Dimension || len(q) != c.Rows*c.Speakers || len(priors) != c.Speakers || int64(c.Rows)*int64(c.Dimension)*int64(c.Speakers) > 1<<27 {
		return nil, fmt.Errorf("invalid VBx centroid geometry/work")
	}
	if err := finiteClustering64(ctx, embeddings); err != nil {
		return nil, err
	}
	if err := centroidProbabilityRows(ctx, priors, 1, c.Speakers); err != nil {
		return nil, err
	}
	if err := centroidProbabilityRows(ctx, q, c.Rows, c.Speakers); err != nil {
		return nil, err
	}
	indices := make([]int, 0, c.Speakers)
	for s, p := range priors {
		if p > 1e-7 {
			indices = append(indices, s)
		}
	}
	if len(indices) == 0 {
		return nil, fmt.Errorf("no retained VBx speaker")
	}
	result := &VBxCentroids{Centroids: make([]float64, len(indices)*c.Dimension), WeightSums: make([]float64, len(indices)), SpeakerIndices: indices}
	for k, s := range indices {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		mass := float64(0)
		for row := 0; row < c.Rows; row++ {
			if row%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			mass += q[row*c.Speakers+s]
		}
		if mass <= 0 {
			return nil, fmt.Errorf("retained VBx speaker has no posterior support")
		}
		result.WeightSums[k] = mass
		for d := 0; d < c.Dimension; d++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			sum := float64(0)
			for row := 0; row < c.Rows; row++ {
				sum += q[row*c.Speakers+s] * embeddings[row*c.Dimension+d]
			}
			result.Centroids[k*c.Dimension+d] = sum / mass
		}
	}
	if err := finiteClustering64(ctx, result.Centroids); err != nil {
		return nil, err
	}
	return result, nil
}
func centroidProbabilityRows(ctx context.Context, values []float64, rows, columns int) error {
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		sum := float64(0)
		for _, value := range values[row*columns : (row+1)*columns] {
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
				return fmt.Errorf("invalid posterior probability")
			}
			sum += value
		}
		if math.Abs(sum-1) > 1e-6 {
			return fmt.Errorf("posterior row not normalised")
		}
	}
	return ctx.Err()
}
