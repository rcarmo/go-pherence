package model

import (
	"math"
	"os/exec"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestGemma4MTPVerifierPostAttentionRMSNormGPUParity(t *testing.T) {
	gpuHost := exec.Command("nvidia-smi", "-L").Run() == nil
	if !nvidia.Available() {
		if gpuHost {
			t.Fatal("CUDA device is visible via nvidia-smi but NVIDIA runtime is unavailable")
		}
		t.Skip("CUDA device unavailable")
	}
	t.Cleanup(nvidia.Shutdown)

	const n = 256
	attnWO := make([]float32, n)
	postNorm := make([]float32, n)
	for i := 0; i < n; i++ {
		v := float32(math.Sin(float64(i)*0.173)*0.9 + math.Cos(float64(i)*0.037)*0.2)
		if i%37 == 0 {
			v *= 5.5
		}
		attnWO[i] = v
		postNorm[i] = float32(0.75 + math.Sin(float64(i)*0.061)*0.25)
	}

	cpu := append([]float32(nil), attnWO...)
	simd.RMSNorm(cpu, postNorm, 1e-6)

	x := nvidia.NewDevBufFrom(append([]float32(nil), attnWO...))
	w := nvidia.NewDevBufFrom(append([]float32(nil), postNorm...))
	out := nvidia.NewDevBuf(n)
	if err := x.ToGPU(); err != nil {
		t.Fatalf("x ToGPU: %v", err)
	}
	if err := w.ToGPU(); err != nil {
		t.Fatalf("w ToGPU: %v", err)
	}
	if err := out.ToGPU(); err != nil {
		t.Fatalf("out ToGPU: %v", err)
	}
	if !nvidia.DevRMSNormOK(out, x, w, 1e-6) {
		t.Fatal("DevRMSNormOK fell back instead of running CUDA RMSNorm")
	}
	out.ToCPU()
	gpu := out.Data()
	if len(gpu) != n {
		t.Fatalf("gpu len=%d want %d", len(gpu), n)
	}
	var maxAbs, meanAbs float64
	maxIdx := -1
	for i := range cpu {
		d := math.Abs(float64(cpu[i] - gpu[i]))
		meanAbs += d
		if d > maxAbs {
			maxAbs = d
			maxIdx = i
		}
	}
	meanAbs /= float64(n)
	if maxAbs > 2e-5 || meanAbs > 2e-6 {
		t.Fatalf("Gemma4 MTP verifier post-attention RMSNorm GPU drift max=%g mean=%g idx=%d cpu=%g gpu=%g", maxAbs, meanAbs, maxIdx, cpu[maxIdx], gpu[maxIdx])
	}
}
