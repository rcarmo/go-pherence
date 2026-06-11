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
	// Use existing RVV vector primitives on riscv64 without extra temporaries:
	//   out = cond - uncond
	//   out = uncond + guidance*out
	//   out = latents + sigma*out
	simdruntime.VecScaleAdd(out.Data, cond.Data, uncond.Data, -1)
	simdruntime.VecScaleAdd(out.Data, uncond.Data, out.Data, guidance)
	simdruntime.VecScaleAdd(out.Data, latents.Data, out.Data, sigma)
	return out, true, nil
}
