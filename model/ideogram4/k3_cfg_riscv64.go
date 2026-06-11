//go:build riscv64

package ideogram4

import simdruntime "github.com/rcarmo/go-pherence/backends/simd/runtime"

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
	// Use existing RVV vector primitives on riscv64:
	//   delta  = cond - uncond
	//   guided = uncond + guidance*delta
	//   out    = latents + sigma*guided
	delta := make([]float32, len(out.Data))
	guided := make([]float32, len(out.Data))
	simdruntime.VecScaleAdd(delta, cond.Data, uncond.Data, -1)
	simdruntime.VecScaleAdd(guided, uncond.Data, delta, guidance)
	simdruntime.VecScaleAdd(out.Data, latents.Data, guided, sigma)
	return out, true, nil
}
