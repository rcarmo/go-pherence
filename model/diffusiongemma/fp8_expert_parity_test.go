package diffusiongemma

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func maxMeanDiff(a, b []float32) (max, mean float64, idx int) {
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		mean += d
		if d > max {
			max = d
			idx = i
		}
	}
	if len(a) > 0 {
		mean /= float64(len(a))
	}
	return max, mean, idx
}

func TestLocalFP8ExpertCPUvsGPUParity(t *testing.T) {
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
	idx, err := BuildFP8ExpertIndex(fp8w, m.Shape.TextLayers, m.Shape.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewExpertLRUCache(2 << 30)
	defer cache.ClearAll()
	layer, expertID := 4, 0
	gateL, upL, downL, err := cache.Put(layer, expertID, fp8w)
	if err != nil {
		t.Fatal(err)
	}
	ep := idx.entries[layer][expertID]
	hidden, intermediate := idx.HiddenSize, idx.Intermediate
	x := make([]float32, hidden)
	rng := rand.New(rand.NewSource(1))
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 0.1
	}
	cpuGate, cpuUp := make([]float32, intermediate), make([]float32, intermediate)
	gpuGate, gpuUp := make([]float32, intermediate), make([]float32, intermediate)
	if err := ep.gate.GemvTo(x, cpuGate); err != nil {
		t.Fatal(err)
	}
	if err := ep.up.GemvTo(x, cpuUp); err != nil {
		t.Fatal(err)
	}
	if err := gpu.GemvFP8E4M3(gpuGate, x, gateL); err != nil {
		t.Fatal(err)
	}
	if err := gpu.GemvFP8E4M3(gpuUp, x, upL); err != nil {
		t.Fatal(err)
	}
	if max, mean, idx := maxMeanDiff(cpuGate, gpuGate); max > 1e-3 {
		t.Fatalf("gate diff max=%g mean=%g idx=%d cpu=%g gpu=%g", max, mean, idx, cpuGate[idx], gpuGate[idx])
	}
	if max, mean, idx := maxMeanDiff(cpuUp, gpuUp); max > 1e-3 {
		t.Fatalf("up diff max=%g mean=%g idx=%d cpu=%g gpu=%g", max, mean, idx, cpuUp[idx], gpuUp[idx])
	}
	cpuAct, gpuAct := make([]float32, intermediate), make([]float32, intermediate)
	if !simd.GELUExactMulTo(cpuAct, cpuGate, cpuUp) || !simd.GELUExactMulTo(gpuAct, gpuGate, gpuUp) {
		t.Fatal("activation rejected")
	}
	cpuDown, gpuDown := make([]float32, hidden), make([]float32, hidden)
	if err := ep.down.GemvTo(cpuAct, cpuDown); err != nil {
		t.Fatal(err)
	}
	if err := gpu.GemvFP8E4M3(gpuDown, gpuAct, downL); err != nil {
		t.Fatal(err)
	}
	if max, mean, idx := maxMeanDiff(cpuDown, gpuDown); max > 1e-3 {
		t.Fatalf("down diff max=%g mean=%g idx=%d cpu=%g gpu=%g", max, mean, idx, cpuDown[idx], gpuDown[idx])
	}
}
