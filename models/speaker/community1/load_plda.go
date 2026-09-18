// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: Apache-2.0
package community1

import (
	"context"
	"fmt"
	"io"

	"github.com/rcarmo/go-pherence/loader/numpy"
)

// LoadRawPLDANPZ reads exact numeric schemas from xvec_transform.npz and
// plda.npz using bounded numeric-only NPY/ZIP parsing, then prepares the owned
// model. cfg is explicit; file names/config are not inferred or downloaded.
// ReaderAt inputs remain caller-owned and immutable through return. F32 values
// are widened exactly to float64, but parity with a mixed-dtype SciPy pipeline
// is NOT established; qualified synthetic numerical comparisons use F64.
// No pickle/object arrays, executable deserialisation, native runtime or GPU.
func LoadRawPLDANPZ(ctx context.Context, xvec io.ReaderAt, xvecSize int64, plda io.ReaderAt, pldaSize int64, cfg PLDAConfig) (*RawPLDAPreparation, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.InputDim < 1 || cfg.InputDim > 512 || cfg.ProjectedDim < 1 || cfg.ProjectedDim > 512 || cfg.OutputDim < 1 || cfg.OutputDim > cfg.ProjectedDim {
		return nil, fmt.Errorf("invalid PLDA NPZ configuration")
	}
	d, n := cfg.InputDim, cfg.ProjectedDim
	x, err := numpy.ReadNPZ(ctx, xvec, xvecSize, map[string][]int{"mean1": {d}, "mean2": {n}, "lda": {d, n}})
	if err != nil {
		return nil, fmt.Errorf("xvec NPZ: %w", err)
	}
	p, err := numpy.ReadNPZ(ctx, plda, pldaSize, map[string][]int{"mu": {n}, "tr": {n, n}, "psi": {n}})
	if err != nil {
		return nil, fmt.Errorf("PLDA NPZ: %w", err)
	}
	return PrepareRawPLDA(ctx, cfg, RawPLDAWeights{x["mean1"].Values, x["mean2"].Values, x["lda"].Values, p["mu"].Values, p["tr"].Values, p["psi"].Values})
}
