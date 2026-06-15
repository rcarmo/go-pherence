package diffusiongemma

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	fp8cpu "github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestLocalFP8ProjectionCPUvsGPUParity(t *testing.T) {
	modelDir := filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8")
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors.index.json")); err != nil {
		t.Skip("local FP8 DiffusionGemma model not present")
	}
	if !gpu.SgemmReady() {
		t.Skip("CUDA SGEMM unavailable")
	}
	m, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	fp8w, err := OpenFP8TextWeights(modelDir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer fp8w.Close()
	gpuModel, err := UploadFP8Layers(fp8w)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuModel.Free()

	layer := 0
	lw := fp8w.Layers[layer]
	gl := gpuModel.Layers[layer]
	rng := rand.New(rand.NewSource(42))
	check := func(name string, lin fp8cpu.Linear, gpuLin *gpu.GPUFP8E4M3Linear) {
		x := make([]float32, lin.InDim)
		for i := range x {
			x[i] = float32(rng.NormFloat64()) * 0.02
		}
		cpuOut := make([]float32, lin.OutDim)
		gpuOut := make([]float32, lin.OutDim)
		if err := lin.Validate(); err != nil {
			t.Fatalf("%s CPU validate: %v", name, err)
		}
		if gpuLin.InDim != lin.InDim || gpuLin.OutDim != lin.OutDim {
			t.Fatalf("%s GPU shape [%d,%d] CPU [%d,%d]", name, gpuLin.OutDim, gpuLin.InDim, lin.OutDim, lin.InDim)
		}
		if err := lin.GemvTo(x, cpuOut); err != nil {
			t.Fatalf("%s CPU gemv: %v", name, err)
		}
		if err := gpu.GemvFP8E4M3(gpuOut, x, gpuLin); err != nil {
			t.Fatalf("%s GPU gemv: %v", name, err)
		}
		if max, mean, idx := maxMeanDiff(cpuOut, gpuOut); max > 1e-3 {
			t.Fatalf("%s diff max=%g mean=%g idx=%d cpu=%g gpu=%g", name, max, mean, idx, cpuOut[idx], gpuOut[idx])
		}
	}
	check("q", fp8cpu.Linear{OutDim: lw.QShape[0], InDim: lw.QShape[1], Weight: lw.QWeight, Scale: lw.QScale}, gl.Q)
	check("k", fp8cpu.Linear{OutDim: lw.KShape[0], InDim: lw.KShape[1], Weight: lw.KWeight, Scale: lw.KScale}, gl.K)
	if lw.VWeight != nil && gl.V != nil {
		check("v", fp8cpu.Linear{OutDim: lw.VShape[0], InDim: lw.VShape[1], Weight: lw.VWeight, Scale: lw.VScale}, gl.V)
	}
	check("o", fp8cpu.Linear{OutDim: lw.OShape[0], InDim: lw.OShape[1], Weight: lw.OWeight, Scale: lw.OScale}, gl.O)
	check("mlp_gate", fp8cpu.Linear{OutDim: lw.GateShape[0], InDim: lw.GateShape[1], Weight: lw.GateWeight, Scale: lw.GateScale}, gl.Gate)
	check("mlp_up", fp8cpu.Linear{OutDim: lw.UpShape[0], InDim: lw.UpShape[1], Weight: lw.UpWeight, Scale: lw.UpScale}, gl.Up)
	check("mlp_down", fp8cpu.Linear{OutDim: lw.DownShape[0], InDim: lw.DownShape[1], Weight: lw.DownWeight, Scale: lw.DownScale}, gl.Down)
}

func TestLocalFP8ProjectionCPUvsGPUParityOnPromptHidden(t *testing.T) {
	modelDir := filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8")
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors.index.json")); err != nil {
		t.Skip("local FP8 DiffusionGemma model not present")
	}
	if !gpu.SgemmReady() {
		t.Skip("CUDA SGEMM unavailable")
	}
	m, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := OpenTextWeights(modelDir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Layers) == 0 || fp.Layers[0].InputLayerNorm == nil {
		t.Fatalf("incomplete forward plan")
	}
	fp8w, err := OpenFP8TextWeights(modelDir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer fp8w.Close()
	gpuModel, err := UploadFP8Layers(fp8w)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuModel.Free()

	hidden := m.Shape.TextHiddenSize
	row, dtype, shape, err := weights.RawTensorRow(fp.Globals.EmbedTokens.Name, 98357) // "Lisbon"
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 1 || shape[0] != hidden {
		t.Fatalf("embed row shape=%v hidden=%d", shape, hidden)
	}
	x := make([]float32, hidden)
	if err := decodeFloatRowTo(x, row, dtype); err != nil {
		t.Fatal(err)
	}
	embedScale := float32(math.Sqrt(float64(hidden)))
	for i := range x {
		x[i] *= embedScale
	}
	norm, err := loadFloatVector(weights, fp.Layers[0].InputLayerNorm)
	if err != nil {
		t.Fatal(err)
	}
	if !simd.RMSNormTo(x, norm, 1e-6) {
		t.Fatalf("input norm rejected")
	}

	lw := fp8w.Layers[0]
	gl := gpuModel.Layers[0]
	check := func(name string, lin fp8cpu.Linear, gpuLin *gpu.GPUFP8E4M3Linear) {
		cpuOut := make([]float32, lin.OutDim)
		gpuOut := make([]float32, lin.OutDim)
		if err := lin.GemvTo(x, cpuOut); err != nil {
			t.Fatalf("%s CPU gemv: %v", name, err)
		}
		if err := gpu.GemvFP8E4M3(gpuOut, x, gpuLin); err != nil {
			t.Fatalf("%s GPU gemv: %v", name, err)
		}
		if max, mean, idx := maxMeanDiff(cpuOut, gpuOut); max > 2e-3 {
			t.Fatalf("%s prompt-hidden diff max=%g mean=%g idx=%d cpu=%g gpu=%g", name, max, mean, idx, cpuOut[idx], gpuOut[idx])
		}
	}
	check("q", fp8cpu.Linear{OutDim: lw.QShape[0], InDim: lw.QShape[1], Weight: lw.QWeight, Scale: lw.QScale}, gl.Q)
	check("k", fp8cpu.Linear{OutDim: lw.KShape[0], InDim: lw.KShape[1], Weight: lw.KWeight, Scale: lw.KScale}, gl.K)
	check("v", fp8cpu.Linear{OutDim: lw.VShape[0], InDim: lw.VShape[1], Weight: lw.VWeight, Scale: lw.VScale}, gl.V)
	check("mlp_gate", fp8cpu.Linear{OutDim: lw.GateShape[0], InDim: lw.GateShape[1], Weight: lw.GateWeight, Scale: lw.GateScale}, gl.Gate)
	check("mlp_up", fp8cpu.Linear{OutDim: lw.UpShape[0], InDim: lw.UpShape[1], Weight: lw.UpWeight, Scale: lw.UpScale}, gl.Up)
}
