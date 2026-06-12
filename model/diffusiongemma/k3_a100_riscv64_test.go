//go:build riscv64

package diffusiongemma

import (
	"math"
	"os"
	"testing"
)

func TestK3A100Q80ModelProjection(t *testing.T) {
	modelDir := os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_TEST_MODEL")
	if modelDir == "" {
		t.Skip("set GO_PHERENCE_DIFFUSIONGEMMA_TEST_MODEL to run model-backed K3 A100 Q8 projection test")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3", "1")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_Q8", "1")
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
	if len(fp.Layers) == 0 || fp.Layers[0].QProj == nil {
		t.Fatal("missing layer 0 q_proj binding")
	}
	binding := fp.Layers[0].QProj
	rows, cols := binding.Shape[0], binding.Shape[1]
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%17)-8) / 17
	}
	got := make([]float32, rows)
	done, err := k3GemmRowsQ80(got, x, 1, weights, binding)
	if err != nil {
		t.Fatal(err)
	}
	if !done {
		t.Fatalf("K3 A100 Q8 path did not accept %s shape=%v", binding.Name, binding.Shape)
	}
	ref := make([]float32, rows)
	raw, dtype, shape, err := weights.RawTensor(binding.Name)
	if err != nil {
		t.Fatal(err)
	}
	if dtype != "F8_E4M3" && dtype != "F8_E4M3FN" {
		t.Fatalf("unexpected dtype %s", dtype)
	}
	if shape[0] != rows || shape[1] != cols {
		t.Fatalf("shape mismatch %v", shape)
	}
	scales, err := loadK3WeightScales(weights, binding.Name, rows)
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < rows; r++ {
		scale := scales[0]
		if len(scales) != 1 {
			scale = scales[r]
		}
		base := r * cols
		var sum float32
		for c := 0; c < cols; c++ {
			sum += fp8DecodeE4M3(raw[base+c]) * scale * x[c]
		}
		ref[r] = sum
	}
	var maxAbs, maxRel float32
	for i := range ref {
		abs := float32(math.Abs(float64(got[i] - ref[i])))
		if abs > maxAbs {
			maxAbs = abs
		}
		den := float32(math.Abs(float64(ref[i]))) + 1e-5
		rel := abs / den
		if rel > maxRel {
			maxRel = rel
		}
	}
	if maxAbs > 0.35 && maxRel > 0.08 {
		t.Fatalf("K3 A100 Q8 projection drift too high max_abs=%.6f max_rel=%.6f", maxAbs, maxRel)
	}
	t.Logf("K3 A100 Q8 projection %s rows=%d cols=%d max_abs=%.6f max_rel=%.6f", binding.Name, rows, cols, maxAbs, maxRel)
}
