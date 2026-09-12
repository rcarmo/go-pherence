// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: Apache-2.0
// Prepared PLDA inference follows vbx_setup, attributed in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
)

// PLDAConfig specifies input, intermediate and retained latent dimensions.
// All are 1..512; OutputDim <= ProjectedDim. This is explicit metadata, not
// inferred from a model file. Only float64 prepared coefficients are supported.
type PLDAConfig struct{ InputDim, ProjectedDim, OutputDim int }

// PreparedPLDAWeights is the output contract of checkpoint preparation:
// Mean1[InputDim], Mean2/Mu/Phi[ProjectedDim], LDA[InputDim,ProjectedDim],
// Transform[ProjectedDim,ProjectedDim], all row-major. Transform rows and Phi
// must be in the SAME descending-eigenvalue order from the generalised B,W
// eigensystem in pinned vbx_setup. Phi must be finite/nonnegative/descending.
// Raw checkpoint tr/psi are NOT these prepared arrays. This API does not solve
// eigenproblems, invert covariance matrices or read npz/pickle. Provenance and
// prepared-vs-raw validation belong to a future checked checkpoint adapter.
type PreparedPLDAWeights struct{ Mean1, Mean2, LDA, Mu, Transform, Phi []float64 }

// PreparedPLDA owns immutable coefficients and is safe for concurrent calls.
// It has no native runtime, global workspace, file handle or GPU allocation.
type PreparedPLDA struct {
	cfg     PLDAConfig
	weights PreparedPLDAWeights
}

func NewPreparedPLDA(ctx context.Context, cfg PLDAConfig, w PreparedPLDAWeights) (*PreparedPLDA, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.InputDim < 1 || cfg.InputDim > 512 || cfg.ProjectedDim < 1 || cfg.ProjectedDim > 512 || cfg.OutputDim < 1 || cfg.OutputDim > cfg.ProjectedDim {
		return nil, fmt.Errorf("invalid prepared PLDA geometry")
	}
	values := [][]float64{w.Mean1, w.Mean2, w.LDA, w.Mu, w.Transform, w.Phi}
	lengths := []int{cfg.InputDim, cfg.ProjectedDim, cfg.InputDim * cfg.ProjectedDim, cfg.ProjectedDim, cfg.ProjectedDim * cfg.ProjectedDim, cfg.ProjectedDim}
	for i, v := range values {
		if len(v) != lengths[i] {
			return nil, fmt.Errorf("invalid prepared PLDA tensor %d length", i)
		}
		if err := finiteClustering64(ctx, v); err != nil {
			return nil, err
		}
	}
	for i, v := range w.Phi {
		if v < 0 || (i > 0 && v > w.Phi[i-1]) {
			return nil, fmt.Errorf("prepared PLDA covariance must be nonnegative and descending")
		}
	}
	var owned PreparedPLDAWeights
	targets := []*[]float64{&owned.Mean1, &owned.Mean2, &owned.LDA, &owned.Mu, &owned.Transform, &owned.Phi}
	for i, v := range values {
		*targets[i] = make([]float64, len(v))
		for start := 0; start < len(v); start += 256 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			copy((*targets[i])[start:min(start+256, len(v))], v[start:min(start+256, len(v))])
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &PreparedPLDA{cfg, owned}, nil
}

// Phi returns an owned copy of the retained between-class covariance diagonal.
// A nil or zero-value model has no retained covariance and returns nil.
func (p *PreparedPLDA) Phi() []float64 {
	if p == nil || p.cfg.OutputDim < 1 {
		return nil
	}
	return append([]float64(nil), p.weights.Phi[:p.cfg.OutputDim]...)
}

// Transform applies sqrt(InputDim)*L2(x-Mean1), then LDA.T, then
// sqrt(ProjectedDim)*L2(projected-Mean2), then (normalised-Mu)*Transform.T,
// retaining OutputDim columns. Inputs/outputs are row-major. Bounds: rows1..4096.
// No extra L2 normalisation follows the final projection. Zero/nonfinite norms
// and intermediate overflow are errors, instead of propagating upstream NaNs.
// Every input, output and intermediate is float64. This component is qualified
// on synthetic float64 oracles only; mixed-dtype trained assets need a separate
// contract. No partial output escapes errors or cancellation.
func (p *PreparedPLDA) Transform(ctx context.Context, embeddings []float64, rows int) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if p == nil || rows < 1 || rows > 4096 {
		return nil, fmt.Errorf("invalid prepared PLDA transform")
	}
	c, w := p.cfg, p.weights
	if c.InputDim < 1 || c.ProjectedDim < 1 || c.OutputDim < 1 || len(embeddings) != rows*c.InputDim {
		return nil, fmt.Errorf("invalid prepared PLDA input")
	}
	if err := finiteClustering64(ctx, embeddings); err != nil {
		return nil, err
	}
	output := make([]float64, rows*c.OutputDim)
	centered := make([]float64, c.InputDim)
	projected := make([]float64, c.ProjectedDim)
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for i := range centered {
			centered[i] = embeddings[row*c.InputDim+i] - w.Mean1[i]
		}
		if err := pldaLengthNorm(centered, math.Sqrt(float64(c.InputDim))); err != nil {
			return nil, err
		}
		for j := range projected {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			sum := float64(0)
			for i, x := range centered {
				sum += w.LDA[i*c.ProjectedDim+j] * x
			}
			projected[j] = sum - w.Mean2[j]
		}
		if err := pldaLengthNorm(projected, math.Sqrt(float64(c.ProjectedDim))); err != nil {
			return nil, err
		}
		for j := range projected {
			projected[j] -= w.Mu[j]
		}
		for j := 0; j < c.OutputDim; j++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			sum := float64(0)
			for i, x := range projected {
				sum += x * w.Transform[j*c.ProjectedDim+i]
			}
			output[row*c.OutputDim+j] = sum
		}
	}
	if err := finiteClustering64(ctx, output); err != nil {
		return nil, err
	}
	return output, nil
}
func pldaLengthNorm(values []float64, scale float64) error {
	sum := float64(0)
	for _, v := range values {
		sum += v * v
	}
	norm := math.Sqrt(sum)
	if norm <= 0 || math.IsNaN(norm) || math.IsInf(norm, 0) {
		return fmt.Errorf("invalid PLDA length norm")
	}
	for i, v := range values {
		values[i] = (v / norm) * scale
	}
	return nil
}
func finiteClustering64(ctx context.Context, values []float64) error {
	for i, v := range values {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("nonfinite clustering value")
		}
	}
	return ctx.Err()
}
