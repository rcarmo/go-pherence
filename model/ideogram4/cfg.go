package ideogram4

import "fmt"

// CombineCFG applies asymmetric classifier-free guidance to per-image-token
// velocities. Ideogram4 runs the conditional branch over [text || image] tokens
// and the unconditional branch over image tokens only; both DiT passes emit a
// velocity for every image token, so the guidance combination itself is a
// shape-matched elementwise blend:
//
//	guided = uncond + scale * (cond - uncond)
//
// The scale is taken from the sampling plan's per-step guidance schedule.
func (p SamplingPlan) CombineCFG(cond Latents, uncond Latents, stepIndex int) (Latents, error) {
	if stepIndex < 0 || stepIndex >= len(p.GuidanceSchedule) {
		return Latents{}, fmt.Errorf("invalid Ideogram4 CFG step index=%d steps=%d", stepIndex, len(p.GuidanceSchedule))
	}
	if err := cond.validate(); err != nil {
		return Latents{}, fmt.Errorf("conditional velocity: %w", err)
	}
	if err := uncond.validate(); err != nil {
		return Latents{}, fmt.Errorf("unconditional velocity: %w", err)
	}
	if cond.Batch != uncond.Batch || cond.Tokens != uncond.Tokens || cond.Channels != uncond.Channels {
		return Latents{}, fmt.Errorf("Ideogram4 CFG velocity shape mismatch cond=%dx%dx%d uncond=%dx%dx%d",
			cond.Batch, cond.Tokens, cond.Channels, uncond.Batch, uncond.Tokens, uncond.Channels)
	}
	if p.ImageTokens > 0 && cond.Tokens != p.ImageTokens {
		return Latents{}, fmt.Errorf("Ideogram4 CFG velocity tokens=%d want image tokens=%d", cond.Tokens, p.ImageTokens)
	}
	scale := p.GuidanceSchedule[stepIndex]
	out := Latents{Batch: cond.Batch, Tokens: cond.Tokens, Channels: cond.Channels, Data: make([]float32, len(cond.Data))}
	for i := range cond.Data {
		out.Data[i] = uncond.Data[i] + scale*(cond.Data[i]-uncond.Data[i])
	}
	return out, nil
}
