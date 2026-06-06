package ideogram4

import (
	"fmt"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gpuCFGStepEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_CFG")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuCFGStepStrict() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_CFG_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuCFGStepOrFallback(sched *FlowMatchScheduler, plan SamplingPlan, latents Latents, cond Latents, uncond Latents, step FlowStep, stepIndex int) (Latents, error) {
	if gpuCFGStepEnabled() {
		if stepIndex < 0 || stepIndex >= len(plan.GuidanceSchedule) {
			return Latents{}, fmt.Errorf("invalid Ideogram4 GPU CFG step index=%d steps=%d", stepIndex, len(plan.GuidanceSchedule))
		}
		if out, err := gpuCFGStep(latents, cond, uncond, plan.GuidanceSchedule[stepIndex], step.Sigma); err == nil || gpuCFGStepStrict() {
			return out, err
		}
	}
	guided, err := plan.CombineCFG(cond, uncond, stepIndex)
	if err != nil {
		return Latents{}, err
	}
	return sched.Step(latents, guided, step)
}

func gpuCFGStep(latents Latents, cond Latents, uncond Latents, guidance float32, sigma float32) (Latents, error) {
	if err := latents.validate(); err != nil {
		return Latents{}, err
	}
	if err := cond.validate(); err != nil {
		return Latents{}, fmt.Errorf("conditional velocity: %w", err)
	}
	if err := uncond.validate(); err != nil {
		return Latents{}, fmt.Errorf("unconditional velocity: %w", err)
	}
	if latents.Batch != cond.Batch || latents.Tokens != cond.Tokens || latents.Channels != cond.Channels || cond.Batch != uncond.Batch || cond.Tokens != uncond.Tokens || cond.Channels != uncond.Channels {
		return Latents{}, fmt.Errorf("Ideogram4 GPU CFG/step shape mismatch latent=%dx%dx%d cond=%dx%dx%d uncond=%dx%dx%d",
			latents.Batch, latents.Tokens, latents.Channels, cond.Batch, cond.Tokens, cond.Channels, uncond.Batch, uncond.Tokens, uncond.Channels)
	}
	out := Latents{Batch: latents.Batch, Tokens: latents.Tokens, Channels: latents.Channels, Data: make([]float32, len(latents.Data))}
	if !nvidia.Available() {
		return Latents{}, fmt.Errorf("nvidia runtime unavailable")
	}
	if err := nvidia.IdeogramCFGStep(out.Data, latents.Data, cond.Data, uncond.Data, guidance, sigma); err != nil {
		return Latents{}, err
	}
	return out, nil
}
