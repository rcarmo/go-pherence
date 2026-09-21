package qwenimage21

import (
	"fmt"
	"math"
)

type FlowStep struct {
	Index                             int
	Timestep, Sigma, SigmaNext, Delta float32
}

func CalculateShift(seqLen int, c SchedulerConfig) (float64, error) {
	if seqLen <= 0 || c.MaxImageSeqLen <= c.BaseImageSeqLen {
		return 0, fmt.Errorf("qwen-image-2.1: invalid shift inputs")
	}
	m := (c.MaxShift - c.BaseShift) / float64(c.MaxImageSeqLen-c.BaseImageSeqLen)
	return float64(seqLen)*m + c.BaseShift - m*float64(c.BaseImageSeqLen), nil
}
func TimeShiftExponential(mu, sigma, t float64) float64 {
	return math.Exp(mu) / (math.Exp(mu) + math.Pow(1/t-1, sigma))
}
func Schedule(seqLen, steps int, c SchedulerConfig) ([]FlowStep, error) {
	if steps <= 0 {
		return nil, fmt.Errorf("qwen-image-2.1: invalid step count")
	}
	mu, e := CalculateShift(seqLen, c)
	if e != nil {
		return nil, e
	}
	sigmas := make([]float64, steps+1)
	for i := 0; i < steps; i++ {
		base := 1 - float64(i)/float64(steps)
		sigmas[i] = TimeShiftExponential(mu, 1, base)
	}
	last := sigmas[steps-1]
	scale := (1 - last) / (1 - c.ShiftTerminal)
	for i := 0; i < steps; i++ {
		sigmas[i] = 1 - (1-sigmas[i])/scale
	}
	sigmas[steps] = 0
	out := make([]FlowStep, steps)
	for i := range out {
		out[i] = FlowStep{Index: i, Timestep: float32(sigmas[i] * float64(c.NumTrainTimesteps)), Sigma: float32(sigmas[i]), SigmaNext: float32(sigmas[i+1]), Delta: float32(sigmas[i+1] - sigmas[i])}
	}
	return out, nil
}
func EulerStep(dst, sample, velocity []float32, step FlowStep) error {
	if len(dst) != len(sample) || len(velocity) != len(sample) || len(sample) == 0 {
		return fmt.Errorf("qwen-image-2.1: scheduler shape mismatch")
	}
	for i := range dst {
		dst[i] = sample[i] + step.Delta*velocity[i]
	}
	return nil
}
func CFG(dst, cond, uncond []float32, scale float32) error {
	if len(dst) != len(cond) || len(uncond) != len(cond) || len(cond) == 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return fmt.Errorf("qwen-image-2.1: CFG shape or scale invalid")
	}
	for i := range dst {
		dst[i] = uncond[i] + scale*(cond[i]-uncond[i])
	}
	return nil
}
