package diffusiongemma

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestLocalGGUFGPUCPUCanvas1Parity(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA SGEMM unavailable")
	}
	ggufPath := filepath.Join("..", "..", "..", "llama.cpp", "models", "diffusiongemma-gguf", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skip("local DiffusionGemma GGUF Q4_K_M reference not present")
	}
	modelDir := filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8")
	meta, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := OpenTextWeightsFromGGUF(g, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	idx, err := BuildGGUFExpertIndex(g, meta.Shape.TextLayers, meta.Shape.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 2202, 106, 107, 105, 4368, 107}
	cfg := DefaultDenoisingConfig()
	cfg.StabilityThreshold = 1
	cfg.ConfidenceThreshold = 0.005

	cpuDen, err := NewTextDenoiserWithDispatcher(meta.Shape, weights, CPUDispatcher{GGUFExpertIndex: idx, FinalLogitSoftcapping: float32(meta.Config.TextConfig.FinalLogitSoftcapping), SkipEviction: true})
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := GenerateCanvas(cpuDen, prompt, cfg, 1, meta.Shape.VocabSize, NewMT19937RNG(1))
	if err != nil {
		t.Fatal(err)
	}
	gpuDen, err := NewTextDenoiserWithDispatcher(meta.Shape, weights, GPUDispatcher{GGUFExpertIndex: idx, FinalLogitSoftcapping: float32(meta.Config.TextConfig.FinalLogitSoftcapping), SkipEviction: true, LMHeadTopK: 8})
	if err != nil {
		t.Fatal(err)
	}
	got, err := GenerateCanvas(gpuDen, prompt, cfg, 1, meta.Shape.VocabSize, NewMT19937RNG(1))
	if err != nil {
		t.Fatal(err)
	}
	if len(cpu.Canvas) != 1 || len(got.Canvas) != 1 || cpu.Canvas[0] != got.Canvas[0] {
		t.Fatalf("canvas mismatch cpu=%v gpu=%v", cpu.Canvas, got.Canvas)
	}
	if len(cpu.Steps) == 0 || len(got.Steps) == 0 {
		t.Fatalf("missing steps cpu=%+v gpu=%+v", cpu.Steps, got.Steps)
	}
	cs, gs := cpu.Steps[0], got.Steps[0]
	if cs.FirstArgmax != gs.FirstArgmax || cs.FirstSampled != gs.FirstSampled || cs.Accepted != gs.Accepted || cs.FirstAccepted != gs.FirstAccepted {
		t.Fatalf("first step mismatch cpu=%+v gpu=%+v", cs, gs)
	}
	// GPU sampling/entropy uses backend reduction order and is allowed to drift
	// until the dense sampler gets its own strict oracle. The token, sample and
	// acceptance decisions are the current GPU↔CPU contract for this smoke.
	t.Logf("GPU/CPU entropy delta: cpu=%g gpu=%g delta=%g", cs.MeanEntropy, gs.MeanEntropy, math.Abs(cs.MeanEntropy-gs.MeanEntropy))
}
