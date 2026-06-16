//go:build diagnostic && gemma4fixtures && gemma4private
// +build diagnostic,gemma4fixtures,gemma4private

package model

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gemma4PerLayerInputsForWrappedStep(m *LlamaModel, wrapped []int, step int) [][]float32 {
	cfg := m.Config
	h := cfg.HiddenSize
	hpl := cfg.HiddenPerLayer
	nl := cfg.NumLayers
	tokID := wrapped[step]

	hidden := append([]float32(nil), m.EmbedTokens.Data()[tokID*h:(tokID+1)*h]...)
	scale := float32(math.Sqrt(float64(h)))
	for i := range hidden {
		hidden[i] *= scale
	}
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		bf16Slice(hidden)
	}

	totalDim := nl * hpl
	proj := make([]float32, totalDim)
	gemvNT(proj, hidden, m.PerLayerModelProj, h, totalDim)
	for i := range proj {
		proj[i] *= m.PerLayerProjScale
	}
	for l := 0; l < nl; l++ {
		sl := proj[l*hpl : (l+1)*hpl]
		rmsNormInPlace(sl, m.PerLayerProjNorm, float32(cfg.RMSNormEps))
	}
	if m.EmbedPerLayer != nil && tokID < cfg.VocabPerLayer {
		embRow := m.EmbedPerLayer[tokID*totalDim : (tokID+1)*totalDim]
		for i := range proj {
			proj[i] = (proj[i] + embRow[i]*m.EmbedPerLayerScale) * m.PerLayerInputScale
		}
	}
	out := make([][]float32, nl)
	for l := 0; l < nl; l++ {
		out[l] = proj[l*hpl : (l+1)*hpl]
	}
	return out
}

func TestGemma4QuantizedLayer7PLISensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer7 PLI sensitivity")
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
		t.Fatalf("load gemma4 quantized model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "gpu" || step != traceStep || layer != 7 {
			return
		}
		if op != "hidden_post_ffn" && op != "hidden_post_pli" {
			return
		}
		gpuOps[opTraceKey{layer, op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()

	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	hiddenPostFFN := gpuOps[opTraceKey{7, "hidden_post_ffn"}]
	hiddenPostPLI := gpuOps[opTraceKey{7, "hidden_post_pli"}]
	if len(hiddenPostFFN) == 0 || len(hiddenPostPLI) == 0 {
		t.Fatal("missing layer7 hidden_post_ffn/hidden_post_pli traces")
	}

	perLayerInputs := gemma4PerLayerInputsForWrappedStep(m, wrapped, traceStep)
	hpl := m.Config.HiddenPerLayer
	gpuPLI := append([]float32(nil), g.perLayerProjBuf.Data()[7*hpl:(7+1)*hpl]...)
	cpuPLI := append([]float32(nil), perLayerInputs[7]...)
	maxAbs, meanAbs := diffStats(cpuPLI, gpuPLI)
	t.Logf("L7 cpu per-layer input vs gpu per-layer input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	layer := m.Layers[7]
	gate2 := make([]float32, hpl)
	gemvNT(gate2, hiddenPostFFN, layer.PLIGate, len(hiddenPostFFN), hpl)
	for i := range gate2 {
		gate2[i] = geluTanh(gate2[i])
		gate2[i] *= gpuPLI[i]
	}
	proj2 := make([]float32, len(hiddenPostFFN))
	gemvNT(proj2, gate2, layer.PLIProj, hpl, len(hiddenPostFFN))
	rmsNormInPlace(proj2, layer.PLIPostNorm, float32(m.Config.RMSNormEps))
	cpuRecon := append([]float32(nil), hiddenPostFFN...)
	for i := range cpuRecon {
		cpuRecon[i] += proj2[i]
	}
	maxAbs, meanAbs = diffStats(hiddenPostPLI, cpuRecon)
	t.Logf("L7 cpu(PLI from gpu hidden_post_ffn) vs gpu hidden_post_pli: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	hidBuf := nvidia.NewDevBufFrom(append([]float32(nil), hiddenPostFFN...))
	pliBuf := nvidia.NewDevBufFrom(append([]float32(nil), gpuPLI...))
	gateBuf := nvidia.NewDevBuf(hpl)
	projBuf := nvidia.NewDevBuf(len(hiddenPostFFN))
	outBuf := nvidia.NewDevBuf(len(hiddenPostFFN))
	defer hidBuf.Free()
	defer pliBuf.Free()
	defer gateBuf.Free()
	defer projBuf.Free()
	defer outBuf.Free()
	for _, b := range []*nvidia.DevBuf{hidBuf, pliBuf, gateBuf, projBuf, outBuf} {
		if err := b.ToGPU(); err != nil {
			t.Fatalf("ToGPU: %v", err)
		}
	}
	nvidia.DevGemv(gateBuf, hidBuf, g.Layers[7].PLIGate, hpl, len(hiddenPostFFN))
	nvidia.DevGELUTanhMul(gateBuf, pliBuf, hpl)
	nvidia.DevGemv(projBuf, gateBuf, g.Layers[7].PLIProj, len(hiddenPostFFN), hpl)
	nvidia.DevRMSNorm(projBuf, projBuf, g.Layers[7].PLIPostNorm, float32(m.Config.RMSNormEps))
	nvidia.DevAdd(outBuf, hidBuf, projBuf)
	nvidia.Sync()
	gpuRecon := append([]float32(nil), outBuf.Data()[:len(hiddenPostPLI)]...)
	maxAbs, meanAbs = diffStats(hiddenPostPLI, gpuRecon)
	t.Logf("L7 fresh gpu PLI from gpu hidden_post_ffn vs gpu hidden_post_pli: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
