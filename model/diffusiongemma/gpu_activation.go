package diffusiongemma

import (
	"fmt"
	"sync"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

var f32GELUExactMulScratch = struct {
	sync.Mutex
	gate []float32
	up   []float32
}{}

func ensureF32GELUExactMulScratch(n int) (gate, up []float32) {
	if cap(f32GELUExactMulScratch.gate) < n {
		f32GELUExactMulScratch.gate = make([]float32, n)
	}
	if cap(f32GELUExactMulScratch.up) < n {
		f32GELUExactMulScratch.up = make([]float32, n)
	}
	return f32GELUExactMulScratch.gate[:n], f32GELUExactMulScratch.up[:n]
}

func freeF32GELUExactMulScratch() {
	f32GELUExactMulScratch.Lock()
	f32GELUExactMulScratch.gate = nil
	f32GELUExactMulScratch.up = nil
	f32GELUExactMulScratch.Unlock()
}

// f32GELUExactMulBuffer computes gate = exact_gelu(gate) * up in-place.
// llama.cpp's Gemma4/DiffusionGemma graph uses ggml_gelu (erf-based), not the
// tanh approximation. Until we add an exact CUDA activation kernel, keep
// opt-in device-resident MLP paths numerically aligned by making this explicit
// host boundary rather than silently using the faster gelu_tanh kernel.
func f32GELUExactMulBuffer(gate, up *gpu.Buffer, n int) error {
	if gate == nil || up == nil || n <= 0 {
		return fmt.Errorf("invalid exact GELU device activation buffers")
	}
	maxInt := int(^uint(0) >> 1)
	if n > maxInt/4 {
		return fmt.Errorf("exact GELU device activation byte-size overflow n=%d", n)
	}
	needBytes := n * 4
	if needBytes > gate.Size || needBytes > up.Size {
		return fmt.Errorf("exact GELU device activation short buffer n=%d bytes=%d gate=%d up=%d", n, needBytes, gate.Size, up.Size)
	}
	f32GELUExactMulScratch.Lock()
	defer f32GELUExactMulScratch.Unlock()
	gateHost, upHost := ensureF32GELUExactMulScratch(n)
	if err := gate.Download(gateHost); err != nil {
		return fmt.Errorf("download exact GELU gate: %w", err)
	}
	if err := up.Download(upHost); err != nil {
		return fmt.Errorf("download exact GELU up: %w", err)
	}
	if !simd.GELUExactMulTo(gateHost, gateHost, upHost) {
		return fmt.Errorf("exact GELU activation rejected n=%d", n)
	}
	if err := gate.Upload(gateHost); err != nil {
		return fmt.Errorf("upload exact GELU activation: %w", err)
	}
	return nil
}
