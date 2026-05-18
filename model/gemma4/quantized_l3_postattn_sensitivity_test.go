//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia"
)

func TestGemma4QuantizedLayer3PostAttnSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer3 post-attn sensitivity")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !nvidia.Available() {
		t.Skip("GPU not available")
	}
	t.Cleanup(nvidia.Shutdown)

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized cpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	cpuOps := map[opTraceKey][]float32{}
	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if step != traceStep || layer != 3 {
			return
		}
		if op != "hidden_in" && op != "o" && op != "mlp_input" {
			return
		}
		cp := append([]float32(nil), vec...)
		k := opTraceKey{layer: layer, op: op}
		if backend == "cpu" {
			cpuOps[k] = cp
		} else if backend == "gpu" {
			gpuOps[k] = cp
		}
	}
	defer func() { debugOpHook = nil }()

	_ = m.Generate(tok.Encode("Hello"), 1)

	mgpu, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized gpu model: %v", err)
	}
	mcuda.Tok = tok
	g, err := LoadGPUModel(mgpu)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	gpuHiddenIn := gpuOps[opTraceKey{3, "hidden_in"}]
	gpuO := gpuOps[opTraceKey{3, "o"}]
	gpuMLPIn := gpuOps[opTraceKey{3, "mlp_input"}]
	cpuMLPIn := cpuOps[opTraceKey{3, "mlp_input"}]
	if len(gpuHiddenIn) == 0 || len(gpuO) == 0 || len(gpuMLPIn) == 0 || len(cpuMLPIn) == 0 {
		t.Fatal("missing layer3 hidden_in/o/mlp_input traces")
	}

	cpuONorm := append([]float32(nil), gpuO...)
	rmsNormInPlace(cpuONorm, m.Layers[3].PostNorm.Data(), float32(m.Config.RMSNormEps))
	cpuHidden := make([]float32, len(gpuHiddenIn))
	for i := range cpuHidden {
		cpuHidden[i] = gpuHiddenIn[i] + cpuONorm[i]
	}
	cpuRecon := append([]float32(nil), cpuHidden...)
	rmsNormInPlace(cpuRecon, m.Layers[3].PreFFNNorm.Data(), float32(m.Config.RMSNormEps))
	bf16Slice(cpuRecon)
	maxAbs, meanAbs := diffStats(gpuMLPIn, cpuRecon)
	t.Logf("L3 cpu(preffn(residual+postnorm(gpuO))) vs gpu mlp_input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	maxAbs, meanAbs = diffStats(cpuMLPIn, cpuRecon)
	t.Logf("L3 cpu(preffn(residual+postnorm(gpuO))) vs cpu mlp_input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	hidBuf := nvidia.NewDevBufFrom(append([]float32(nil), gpuHiddenIn...))
	oBuf := nvidia.NewDevBufFrom(append([]float32(nil), gpuO...))
	tmpBuf := nvidia.NewDevBuf(len(gpuHiddenIn))
	outBuf := nvidia.NewDevBuf(len(gpuHiddenIn))
	defer hidBuf.Free()
	defer oBuf.Free()
	defer tmpBuf.Free()
	defer outBuf.Free()
	for _, b := range []*nvidia.DevBuf{hidBuf, oBuf, tmpBuf, outBuf} {
		if err := b.ToGPU(); err != nil {
			t.Fatalf("ToGPU: %v", err)
		}
	}
	nvidia.DevRMSNorm(tmpBuf, oBuf, g.Layers[3].PostNorm, float32(m.Config.RMSNormEps))
	nvidia.DevAdd(outBuf, hidBuf, tmpBuf)
	nvidia.DevRMSNorm(outBuf, outBuf, g.Layers[3].PreFFNNorm, float32(m.Config.RMSNormEps))
	nvidia.DevToBF16(outBuf, len(gpuHiddenIn))
	nvidia.Sync()
	gpuRecon := append([]float32(nil), outBuf.Data()[:len(gpuMLPIn)]...)
	maxAbs, meanAbs = diffStats(gpuMLPIn, gpuRecon)
	t.Logf("L3 fresh gpu(preffn(residual+postnorm(gpuO))) vs gpu mlp_input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
