// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: Apache-2.0
// Deterministic non-HMM VBx equations follow the source attributed in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
)

// VBxConfig describes deterministic cluster_vbx's supported subset. X is
// float64 [Rows,Dimension]; Phi[Dimension] is finite/nonnegative; labels[Rows]
// initialise responsibilities with max(labels)+1 speakers (gaps preserved).
// Bounds: rows1..4096, dimension1..512, speakers1..64, iterations1..100,
// Rows*Dimension*Speakers*MaxIterations <= 2^28 (admission before allocation).
// Fa/Fb > 0, Epsilon >= 0, and all scalar options finite. Negative smoothing
// keeps hard one-hot labels; nonnegative smoothing uses stable row softmax.
// No random initialisation, HMM, supplied initial speaker models or AHC here.
type VBxConfig struct {
	Rows, Dimension, MaxIterations int
	Fa, Fb, Epsilon, InitSmoothing float64
}

// Community1VBxConfig provides the pinned Community-1 update parameters. It
// does not qualify a checkpoint, perform AHC, select speaker counts or schedule
// the work. Explicit configs remain useful for source-parity fixtures.
func Community1VBxConfig(rows, dimension int) VBxConfig {
	return VBxConfig{rows, dimension, 20, .07, .8, 1e-4, 7}
}

// VBxResult owns responsibilities [Rows,Speakers], priors [Speakers], and the
// final speaker models Alpha/InvL [Speakers,Dimension], plus ELBO per iteration.
// All initial speaker slots are retained; the source's later sp>1e-7 pruning
// and weighted centroids are separate operations. StoppedByTolerance includes
// a decrease; DecreasedELBO records that diagnostic rather than hiding it.
type VBxResult struct {
	Speakers                                    int
	Responsibilities, Priors, Alpha, InvL, ELBO []float64
	StoppedByTolerance, DecreasedELBO           bool
}

// VBxObserver receives transient read-only iteration slices (zero-based index).
// It must copy any retained values and must not mutate them. Errors abort the
// call with no result; this diagnostic boundary does not promise pre-emption
// inside a dot product. Each dot is bounded by validated geometry.
type VBxObserver func(iteration int, gamma, priors, alpha, invL []float64, elbo float64) error

func ClusterVBx(ctx context.Context, x, phi []float64, labels []int, cfg VBxConfig) (*VBxResult, error) {
	return ClusterVBxObserved(ctx, x, phi, labels, cfg, nil)
}

// ClusterVBxObserved implements pinned cluster_vbx smoothing and deterministic
// VBx GMM updates in float64 scalar Go, with log-sum-exp, learned priors and the
// original early-stop inequality. No hidden BLAS or new SIMD/speed claim. Input
// buffers are never changed or retained. Invalid values/intermediate overflow
// fail; cancellation returns no partial output. Output arrays are reusable only
// by the caller, not global caches or later invocations.
func ClusterVBxObserved(ctx context.Context, x, phi []float64, labels []int, cfg VBxConfig, observe VBxObserver) (*VBxResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg
	if c.Rows < 1 || c.Rows > 4096 || c.Dimension < 1 || c.Dimension > 512 || c.MaxIterations < 1 || c.MaxIterations > 100 || len(x) != c.Rows*c.Dimension || len(phi) != c.Dimension || len(labels) != c.Rows {
		return nil, fmt.Errorf("invalid VBx geometry")
	}
	if err := finiteClustering64(ctx, []float64{c.Fa, c.Fb, c.Epsilon, c.InitSmoothing}); err != nil {
		return nil, err
	}
	if c.Fa <= 0 || c.Fb <= 0 || c.Epsilon < 0 {
		return nil, fmt.Errorf("invalid VBx parameters")
	}
	for _, values := range [][]float64{x, phi} {
		if err := finiteClustering64(ctx, values); err != nil {
			return nil, err
		}
	}
	for _, value := range phi {
		if value < 0 {
			return nil, fmt.Errorf("negative VBx covariance")
		}
	}
	speakers := 0
	for i, label := range labels {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if label < 0 || label >= 64 {
			return nil, fmt.Errorf("invalid VBx label")
		}
		speakers = max(speakers, label+1)
	}
	if int64(c.Rows)*int64(c.Dimension)*int64(speakers)*int64(c.MaxIterations) > 1<<28 {
		return nil, fmt.Errorf("VBx work bound exceeded")
	}
	ratio := c.Fa / c.Fb
	if math.IsInf(ratio, 0) || ratio == 0 {
		return nil, fmt.Errorf("VBx Fa/Fb not representable")
	}
	result := &VBxResult{Speakers: speakers, Responsibilities: make([]float64, c.Rows*speakers), Priors: make([]float64, speakers), Alpha: make([]float64, speakers*c.Dimension), InvL: make([]float64, speakers*c.Dimension), ELBO: make([]float64, 0, c.MaxIterations)}
	gamma, pi, alpha, invL := result.Responsibilities, result.Priors, result.Alpha, result.InvL
	low, high := float64(0), float64(1)
	if c.InitSmoothing >= 0 {
		e := math.Exp(-c.InitSmoothing)
		denom := 1 + float64(speakers-1)*e
		low = e / denom
		high = 1 / denom
	}
	for row, label := range labels {
		if row%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for s := 0; s < speakers; s++ {
			gamma[row*speakers+s] = low
		}
		gamma[row*speakers+label] = high
	}
	for s := range pi {
		pi[s] = 1 / float64(speakers)
	}
	g := make([]float64, c.Rows)
	rho := make([]float64, len(x))
	v := make([]float64, c.Dimension)
	counts := make([]float64, speakers)
	penalty := make([]float64, speakers)
	scores := make([]float64, speakers)
	logPrior := make([]float64, speakers)
	for d, value := range phi {
		v[d] = math.Sqrt(value)
	}
	for row := 0; row < c.Rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		sum := float64(0)
		for d := 0; d < c.Dimension; d++ {
			value := x[row*c.Dimension+d]
			sum += value * value
			rho[row*c.Dimension+d] = value * v[d]
		}
		g[row] = -.5 * (sum + float64(c.Dimension)*math.Log(2*math.Pi))
	}
	if err := finiteClustering64(ctx, g); err != nil {
		return nil, err
	}
	if err := finiteClustering64(ctx, rho); err != nil {
		return nil, err
	}
	for iteration := 0; iteration < c.MaxIterations; iteration++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		clear(counts)
		clear(alpha)
		for row := 0; row < c.Rows; row++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			for s := 0; s < speakers; s++ {
				counts[s] += gamma[row*speakers+s]
			}
		}
		for s := 0; s < speakers; s++ {
			for d := 0; d < c.Dimension; d++ {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				sum := float64(0)
				for row := 0; row < c.Rows; row++ {
					sum += gamma[row*speakers+s] * rho[row*c.Dimension+d]
				}
				index := s*c.Dimension + d
				invL[index] = 1 / (1 + ratio*counts[s]*phi[d])
				alpha[index] = (ratio * invL[index]) * sum
			}
			penalty[s] = 0
			for d := 0; d < c.Dimension; d++ {
				i := s*c.Dimension + d
				penalty[s] += (invL[i] + alpha[i]*alpha[i]) * phi[d]
			}
			logPrior[s] = math.Log(pi[s] + 1e-8)
		}
		if err := finiteClustering64(ctx, alpha); err != nil {
			return nil, err
		}
		if err := finiteClustering64(ctx, invL); err != nil {
			return nil, err
		}
		clear(counts)
		logLikelihood := float64(0)
		for row := 0; row < c.Rows; row++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			maxScore := math.Inf(-1)
			for s := 0; s < speakers; s++ {
				sum := float64(0)
				for d := 0; d < c.Dimension; d++ {
					sum += rho[row*c.Dimension+d] * alpha[s*c.Dimension+d]
				}
				scores[s] = c.Fa*(sum-.5*penalty[s]+g[row]) + logPrior[s]
				maxScore = math.Max(maxScore, scores[s])
			}
			total := float64(0)
			for _, score := range scores {
				total += math.Exp(score - maxScore)
			}
			marginal := maxScore + math.Log(total)
			if math.IsNaN(marginal) || math.IsInf(marginal, 0) {
				return nil, fmt.Errorf("nonfinite VBx marginal")
			}
			logLikelihood += marginal
			for s, score := range scores {
				value := math.Exp(score - marginal)
				gamma[row*speakers+s] = value
				counts[s] += value
			}
		}
		total := float64(0)
		for _, count := range counts {
			total += count
		}
		for s := range pi {
			pi[s] = counts[s] / total
		}
		regularisation := float64(0)
		for i, value := range invL {
			if i%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			regularisation += math.Log(value) - value - alpha[i]*alpha[i] + 1
		}
		elbo := logLikelihood + c.Fb*.5*regularisation
		if math.IsNaN(elbo) || math.IsInf(elbo, 0) {
			return nil, fmt.Errorf("nonfinite VBx ELBO")
		}
		result.ELBO = append(result.ELBO, elbo)
		if observe != nil {
			if err := observe(iteration, gamma, pi, alpha, invL, elbo); err != nil {
				return nil, err
			}
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if iteration > 0 {
			delta := elbo - result.ELBO[iteration-1]
			if delta < 0 {
				result.DecreasedELBO = true
			}
			if delta < c.Epsilon {
				result.StoppedByTolerance = true
				break
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
