package whisper

import (
	"os"
	"testing"
	"time"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestGPUEncoderForward(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available")
	}
	if !nv.SgemmReady() {
		t.Skip("GPU not available")
	}

	cfg := Tiny()
	enc, _, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	gpuEnc := NewGPUEncoder(enc, cfg)
	if !gpuEnc.ready {
		t.Skip("GPU encoder not ready")
	}

	// 3 seconds of mel input
	T := 187
	melFlat := make([]float32, cfg.NumMelBins*T)
	for i := range melFlat {
		melFlat[i] = float32(i%80) * 0.01
	}

	// CPU baseline
	start := time.Now()
	cpuOut := enc.Forward(melFlat, T)
	cpuTime := time.Since(start)

	// GPU path
	start = time.Now()
	gpuOut := gpuEnc.ForwardGPU(melFlat, T)
	gpuTime := time.Since(start)

	// Verify same output shape
	if len(gpuOut) != len(cpuOut) {
		t.Fatalf("output length mismatch: gpu=%d cpu=%d", len(gpuOut), len(cpuOut))
	}

	// Verify numerical similarity (should be identical since GPU path falls back to CPU GEMV with same Sdot)
	var maxDiff float32
	for i := range cpuOut {
		d := cpuOut[i] - gpuOut[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 0.01 {
		t.Fatalf("GPU vs CPU max diff = %f (too large)", maxDiff)
	}

	t.Logf("CPU: %v, GPU: %v, speedup: %.2fx, max_diff: %e",
		cpuTime, gpuTime, float64(cpuTime)/float64(gpuTime), maxDiff)
}
