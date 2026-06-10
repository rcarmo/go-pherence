//go:build riscv64

package ideogram4

func k3CFGStep(latents Latents, cond, uncond Latents, guidance, sigma float32) (Latents, bool, error) {
	if !k3Enabled() {
		return Latents{}, false, nil
	}
	if err := latents.validate(); err != nil {
		return Latents{}, true, err
	}
	if err := cond.validate(); err != nil {
		return Latents{}, true, err
	}
	if err := uncond.validate(); err != nil {
		return Latents{}, true, err
	}
	out := Latents{Batch: latents.Batch, Tokens: latents.Tokens, Channels: latents.Channels, Data: make([]float32, len(latents.Data))}
	// Placeholder K3 vector surface: scalar Go loop today, isolated so the next
	// riscv64 RVV kernel can replace it without touching scheduler semantics.
	for i := range out.Data {
		guided := uncond.Data[i] + guidance*(cond.Data[i]-uncond.Data[i])
		out.Data[i] = latents.Data[i] + sigma*guided
	}
	return out, true, nil
}
