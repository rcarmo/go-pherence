// Package training defines portable admission for GPU training primitives.
// It does not initialise a backend or execute device code.
package training

import (
	"fmt"
	"math"
)

// AdamWStep supplies per-update F32 constants. Bias corrections are computed on
// the host once, matching the CPU trainer's float64 Pow followed by F32 rounding.
// GradientScale applies already computed global clipping; no local per-tensor
// norm or implicit change to the global clipping policy occurs in the kernel.
// EMA is updated only when EMADecay>0. The caller owns successful-step accounting
// and recovery: kernels mutate parameters/moments and cannot roll back failures.
type AdamWStep struct {
	LearningRate, Beta1, Beta2, Epsilon, WeightDecay float32
	Bias1, Bias2, EMADecay, GradientScale            float32
}

func NewAdamWStep(lr, beta1, beta2, eps, decay, ema, scale float32, step uint64) (AdamWStep, error) {
	if step == 0 || step > uint64(math.MaxInt) {
		return AdamWStep{}, fmt.Errorf("invalid AdamW update step")
	}
	p := AdamWStep{lr, beta1, beta2, eps, decay, float32(1 - math.Pow(float64(beta1), float64(step))), float32(1 - math.Pow(float64(beta2), float64(step))), ema, scale}
	return p, p.Validate()
}
func (p AdamWStep) Validate() error {
	for _, v := range []float32{p.LearningRate, p.Beta1, p.Beta2, p.Epsilon, p.WeightDecay, p.Bias1, p.Bias2, p.EMADecay, p.GradientScale} {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("nonfinite AdamW constant")
		}
	}
	if p.LearningRate <= 0 || p.Beta1 < 0 || p.Beta1 >= 1 || p.Beta2 < 0 || p.Beta2 >= 1 || p.Epsilon <= 0 || p.WeightDecay < 0 || p.Bias1 <= 0 || p.Bias1 > 1 || p.Bias2 <= 0 || p.Bias2 > 1 || p.EMADecay < 0 || p.EMADecay >= 1 || p.GradientScale < 0 || p.GradientScale > 1 {
		return fmt.Errorf("invalid AdamW constants")
	}
	if math.IsInf(float64(p.LearningRate*p.WeightDecay), 0) {
		return fmt.Errorf("AdamW decay product overflows")
	}
	return nil
}

// LinearShape bounds the initial portable row-major linear backward kernels:
// X[M,K], W[N,K], dY[M,N], dX[M,K], dW[N,K], dB[N]. Each output element has
// one owner. dW/dB overwrite, never atomically accumulate; callers accumulate
// branches explicitly. A zero/empty dimension is rejected.
func LinearShape(m, k, n int) error {
	if m < 1 || m > 1024 || k < 1 || k > 16384 || n < 1 || n > 16384 {
		return fmt.Errorf("training linear dimensions exceed M<=1024,K/N<=16384")
	}
	return nil
}
