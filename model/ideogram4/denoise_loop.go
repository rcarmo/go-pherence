package ideogram4

import "fmt"

// DenoiseLoop runs the full FlowMatch sampling loop with asymmetric CFG over a
// conditional/unconditional DiT pair.
//
//   - cond:        conditional transformer (text+image joint sequence)
//   - uncond:      unconditional transformer (image-only sequence)
//   - latents:     initial noise latents [imageTokens, in_channels]
//   - gridH/gridW: latent grid layout
//   - textFeatures:[textTokens, llm_features_dim] Qwen3-VL conditioning
//   - plan:        sampling plan (steps + guidance schedule) for this resolution
//
// Returns the denoised latents [imageTokens, in_channels].
func DenoiseLoop(cond, uncond *DiTModel, sched *FlowMatchScheduler, plan SamplingPlan, latents []float32, gridH, gridW int, textFeatures []float32) ([]float32, error) {
	if cond == nil || uncond == nil {
		return nil, fmt.Errorf("ideogram4 denoise: nil transformer")
	}
	if sched == nil {
		return nil, fmt.Errorf("ideogram4 denoise: nil scheduler")
	}
	cfg := cond.Config
	imgTokens := gridH * gridW
	if imgTokens <= 0 || len(latents) != imgTokens*cfg.InChannels {
		return nil, fmt.Errorf("ideogram4 denoise: latents=%d want %d*%d", len(latents), imgTokens, cfg.InChannels)
	}
	if plan.ImageTokens != 0 && plan.ImageTokens != imgTokens {
		return nil, fmt.Errorf("ideogram4 denoise: plan image tokens=%d grid=%d", plan.ImageTokens, imgTokens)
	}
	if len(plan.Steps) == 0 {
		return nil, fmt.Errorf("ideogram4 denoise: empty sampling plan")
	}

	x := Latents{Batch: 1, Tokens: imgTokens, Channels: cfg.InChannels, Data: append([]float32(nil), latents...)}
	for si, step := range plan.Steps {
		condVel, err := cond.Velocity(x.Data, gridH, gridW, textFeatures, step.T)
		if err != nil {
			return nil, fmt.Errorf("ideogram4 denoise step %d cond: %w", si, err)
		}
		uncondVel, err := uncond.Velocity(x.Data, gridH, gridW, nil, step.T)
		if err != nil {
			return nil, fmt.Errorf("ideogram4 denoise step %d uncond: %w", si, err)
		}
		condL := Latents{Batch: 1, Tokens: imgTokens, Channels: cfg.InChannels, Data: condVel}
		uncondL := Latents{Batch: 1, Tokens: imgTokens, Channels: cfg.InChannels, Data: uncondVel}
		x, err = gpuCFGStepOrFallback(sched, plan, x, condL, uncondL, step, si)
		if err != nil {
			return nil, fmt.Errorf("ideogram4 denoise step %d cfg/update: %w", si, err)
		}
	}
	return x.Data, nil
}
