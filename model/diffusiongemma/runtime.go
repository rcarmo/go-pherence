package diffusiongemma

// BlockDiffusionState is a placeholder for the native block-diffusion decoding
// state. The scaffold keeps this separate from autoregressive KV state because
// DiffusionGemma denoises a canvas of masked tokens over multiple steps.
type BlockDiffusionState struct {
	CanvasLength int `json:"canvas_length"`
	Step         int `json:"step"`
}
