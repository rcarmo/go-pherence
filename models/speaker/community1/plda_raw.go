// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: Apache-2.0
package community1

import (
	"context"
	"fmt"
	"math"
	"sort"
)

// RawPLDAWeights holds the numeric xvec_transform.npz and plda.npz arrays:
// Mean1[InputDim], Mean2/Mu/Psi[ProjectedDim], LDA[InputDim,ProjectedDim],
// TR[ProjectedDim,ProjectedDim], all float64 row-major. TR/Psi are RAW checkpoint
// coefficients. Unlike PreparedPLDAWeights, they need not be spectrum-sorted.
type RawPLDAWeights struct{ Mean1, Mean2, LDA, Mu, TR, Psi []float64 }

// RawPLDAPreparation owns the model and sorted raw row indices, and reports
// numerical admission diagnostics. No eigensolver signs or bases are promised.
// ConditionInf = ||TR||inf*||inverse(TR)||inf; InverseResidual is max absolute
// element of TR*inverse(TR)-I and inverse(TR)*TR-I. These are guardrails, not a
// trained-model error bound or a condition estimate of the squared Gram matrix.
type RawPLDAPreparation struct {
	Model                         *PreparedPLDA
	Order                         []int
	ConditionInf, InverseResidual float64
}

// PrepareRawPLDA avoids squaring/inverting covariance matrices or solving a
// general eigenproblem. For invertible T and D=diag(psi)>0, the source defines
// W=T^-1 T^-T, B=T^-1 D T^-T. V=T^T satisfies V^T W V=I and B V=W V D.
// Thus sorted RAW rows T already give a valid prepared transform. SciPy's basis
// can differ by signs and orthogonal rotations inside repeated-eigenvalue
// blocks. VBx is invariant to these transformations when whole blocks are kept.
// Transformed feature values need not equal SciPy's coordinates elementwise.
//
// Admission: normal PLDA geometry (dimensions<=512), positive finite Psi,
// invertible TR, conditionInf<=1e6, absolute two-sided inverse residual<=1e-8.
// If truncating, the adjacent spectral gap must exceed 1e-8*max(Psi), refusing
// both exact and close boundary degeneracy. These are explicit conservative
// development policies, not source defaults. No repaired eigenvalues, jitter,
// implicit dimension change or tolerance widening. Large condition numbers,
// mixed-dtype parity and actual trained assets require separate qualification.
// The inverse is used ONLY for rank/conditioning diagnostics, not inference.
// Inputs immutable during call, coefficients owned, no partial cancellation.
func PrepareRawPLDA(ctx context.Context, cfg PLDAConfig, w RawPLDAWeights) (*RawPLDAPreparation, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.InputDim < 1 || cfg.InputDim > 512 || cfg.ProjectedDim < 1 || cfg.ProjectedDim > 512 || cfg.OutputDim < 1 || cfg.OutputDim > cfg.ProjectedDim {
		return nil, fmt.Errorf("invalid raw PLDA geometry")
	}
	n := cfg.ProjectedDim
	values := [][]float64{w.Mean1, w.Mean2, w.LDA, w.Mu, w.TR, w.Psi}
	lengths := []int{cfg.InputDim, n, cfg.InputDim * n, n, n * n, n}
	for i, v := range values {
		if len(v) != lengths[i] {
			return nil, fmt.Errorf("invalid raw PLDA tensor %d length", i)
		}
		if err := finiteClustering64(ctx, v); err != nil {
			return nil, err
		}
	}
	order := make([]int, n)
	for i, v := range w.Psi {
		if v <= 0 {
			return nil, fmt.Errorf("raw PLDA Psi must be positive")
		}
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return w.Psi[order[i]] > w.Psi[order[j]] })
	if cfg.OutputDim < n {
		gap := w.Psi[order[cfg.OutputDim-1]] - w.Psi[order[cfg.OutputDim]]
		if gap <= 1e-8*w.Psi[order[0]] {
			return nil, fmt.Errorf("raw PLDA truncation splits degenerate/close eigenvalues")
		}
	}
	condition, residual, err := rawPLDACondition(ctx, w.TR, n)
	if err != nil {
		return nil, err
	}
	transform := make([]float64, n*n)
	phi := make([]float64, n)
	for row, index := range order {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		copy(transform[row*n:(row+1)*n], w.TR[index*n:(index+1)*n])
		phi[row] = w.Psi[index]
	}
	model, err := NewPreparedPLDA(ctx, cfg, PreparedPLDAWeights{w.Mean1, w.Mean2, w.LDA, w.Mu, transform, phi})
	if err != nil {
		return nil, err
	}
	return &RawPLDAPreparation{model, order, condition, residual}, nil
}

// Partial-pivot Gauss-Jordan diagnostics on bounded square matrices. This is a
// scalar Go validation path, not a model kernel, BLAS shim or general public LA.
func rawPLDACondition(ctx context.Context, source []float64, n int) (float64, float64, error) {
	matrix := append([]float64(nil), source...)
	inverse := make([]float64, n*n)
	norm := float64(0)
	for row := 0; row < n; row++ {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		sum := float64(0)
		for col := 0; col < n; col++ {
			sum += math.Abs(source[row*n+col])
		}
		norm = math.Max(norm, sum)
		inverse[row*n+row] = 1
	}
	if norm == 0 || math.IsInf(norm, 0) {
		return 0, 0, fmt.Errorf("invalid raw PLDA matrix norm")
	}
	for col := 0; col < n; col++ {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		pivot := col
		for row := col + 1; row < n; row++ {
			if math.Abs(matrix[row*n+col]) > math.Abs(matrix[pivot*n+col]) {
				pivot = row
			}
		}
		value := matrix[pivot*n+col]
		if value == 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, 0, fmt.Errorf("singular/nonfinite raw PLDA matrix")
		}
		if pivot != col {
			for j := 0; j < n; j++ {
				matrix[pivot*n+j], matrix[col*n+j] = matrix[col*n+j], matrix[pivot*n+j]
				inverse[pivot*n+j], inverse[col*n+j] = inverse[col*n+j], inverse[pivot*n+j]
			}
		}
		for j := 0; j < n; j++ {
			matrix[col*n+j] /= value
			inverse[col*n+j] /= value
		}
		matrix[col*n+col] = 1
		for row := 0; row < n; row++ {
			if err := ctx.Err(); err != nil {
				return 0, 0, err
			}
			if row == col {
				continue
			}
			factor := matrix[row*n+col]
			for j := 0; j < n; j++ {
				matrix[row*n+j] -= factor * matrix[col*n+j]
				inverse[row*n+j] -= factor * inverse[col*n+j]
			}
			matrix[row*n+col] = 0
		}
	}
	if err := finiteClustering64(ctx, inverse); err != nil {
		return 0, 0, err
	}
	inverseNorm := float64(0)
	for row := 0; row < n; row++ {
		sum := float64(0)
		for col := 0; col < n; col++ {
			sum += math.Abs(inverse[row*n+col])
		}
		inverseNorm = math.Max(inverseNorm, sum)
	}
	condition := norm * inverseNorm
	if math.IsNaN(condition) || math.IsInf(condition, 0) || condition > 1e6 {
		return 0, 0, fmt.Errorf("raw PLDA condition exceeds 1e6")
	}
	residual := float64(0)
	for row := 0; row < n; row++ {
		for col := 0; col < n; col++ {
			if err := ctx.Err(); err != nil {
				return 0, 0, err
			}
			left, right := float64(0), float64(0)
			for k := 0; k < n; k++ {
				left += source[row*n+k] * inverse[k*n+col]
				right += inverse[row*n+k] * source[k*n+col]
			}
			if row == col {
				left--
				right--
			}
			residual = math.Max(residual, math.Max(math.Abs(left), math.Abs(right)))
		}
	}
	if math.IsNaN(residual) || math.IsInf(residual, 0) || residual > 1e-8 {
		return 0, 0, fmt.Errorf("raw PLDA inverse residual exceeds 1e-8")
	}
	return condition, residual, ctx.Err()
}
