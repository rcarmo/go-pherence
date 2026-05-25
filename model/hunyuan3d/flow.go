package hunyuan3d

import (
	"fmt"
	"math/rand"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// BlendClassifierFreeGuidance applies the Hunyuan3D/DiT classifier-free
// guidance convention used by the flow pipeline:
//
//	uncond + guidanceScale*(cond-uncond)
//
// The returned slice is newly allocated so callers can safely retain it across
// scheduler steps.
func BlendClassifierFreeGuidance(uncond, cond []float32, guidanceScale float32) ([]float32, error) {
	if len(uncond) != len(cond) {
		return nil, fmt.Errorf("hunyuan3d cfg blend: shape mismatch uncond=%d cond=%d", len(uncond), len(cond))
	}
	out := make([]float32, len(cond))
	BlendClassifierFreeGuidanceInto(out, uncond, cond, guidanceScale)
	return out, nil
}

// BlendClassifierFreeGuidanceInto is the allocation-free form of
// BlendClassifierFreeGuidance.
func BlendClassifierFreeGuidanceInto(dst, uncond, cond []float32, guidanceScale float32) error {
	if len(dst) != len(cond) || len(uncond) != len(cond) {
		return fmt.Errorf("hunyuan3d cfg blend: shape mismatch dst=%d uncond=%d cond=%d", len(dst), len(uncond), len(cond))
	}
	for i := range cond {
		dst[i] = uncond[i] + guidanceScale*(cond[i]-uncond[i])
	}
	return nil
}

// DeterministicLatents returns a reproducible standard-normal latent tensor for
// Go-side scheduler/loop parity tests. This is not intended to match Torch RNG
// bit-for-bit; Python fixtures should persist their exact latents when strict
// cross-runtime parity is required.
func DeterministicLatents(shape []int, seed int64) ([]float32, error) {
	n, err := shapeNumel(shape)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out, nil
}

// ApplyFlowMatchStepInto applies one Euler flow-matching scheduler update to a
// full latent tensor: sample + (sigmaNext-sigma)*modelOutput.
func ApplyFlowMatchStepInto(dst, sample, modelOutput []float32, sigma, sigmaNext float32) error {
	if len(dst) != len(sample) || len(modelOutput) != len(sample) {
		return fmt.Errorf("hunyuan3d flow step: shape mismatch dst=%d sample=%d model=%d", len(dst), len(sample), len(modelOutput))
	}
	delta := sigmaNext - sigma
	for i := range sample {
		dst[i] = sample[i] + delta*modelOutput[i]
	}
	return nil
}

// RunFlowMatchReference runs a scheduler loop when the caller already has one
// model-output tensor per schedule transition. It is a small native target for
// low-step fixture validation before the Hunyuan3D DiT runtime exists.
func RunFlowMatchReference(initial []float32, schedule loaderconfig.Hunyuan3DFlowMatchSchedule, modelOutputs [][]float32) ([]float32, error) {
	sigmas := schedule.SchedulerSigmasWithTerminalOne
	if len(sigmas) == 0 {
		sigmas = schedule.Sigmas
	}
	steps := len(sigmas) - 1
	if steps < 0 || len(modelOutputs) != steps {
		return nil, fmt.Errorf("hunyuan3d flow loop: got %d model outputs for %d scheduler transitions", len(modelOutputs), steps)
	}
	cur := append([]float32(nil), initial...)
	next := make([]float32, len(cur))
	for i := 0; i < steps; i++ {
		if err := ApplyFlowMatchStepInto(next, cur, modelOutputs[i], float32(sigmas[i]), float32(sigmas[i+1])); err != nil {
			return nil, err
		}
		cur, next = next, cur
	}
	return cur, nil
}

func shapeNumel(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("hunyuan3d latent shape: empty")
	}
	n := 1
	maxInt := int(^uint(0) >> 1)
	for _, d := range shape {
		if d <= 0 {
			return 0, fmt.Errorf("hunyuan3d latent shape: invalid dimension %d in %v", d, shape)
		}
		if n > maxInt/d {
			return 0, fmt.Errorf("hunyuan3d latent shape: overflow for %v", shape)
		}
		n *= d
	}
	return n, nil
}
