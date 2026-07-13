package model

import (
	"math"
	"os"
	"testing"
)

func TestMOSSQwen3GPUFirstPositionLayerParity(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" || os.Getenv("MOSS_TRANSCRIBE_GPU_PARITY") == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR and MOSS_TRANSCRIBE_GPU_PARITY=1")
	}
	m, err := LoadLlama(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Skipf("GPU model unavailable: %v", err)
	}
	// Do not close here: the CUDA driver test process can tear down the primary
	// context while Go finalizers are still draining mmap-backed model tensors.
	h := m.Config.HiddenSize
	embeddings := make([]float32, 2*h)
	for i := range embeddings {
		embeddings[i] = float32(0.08*math.Sin(float64(i)*0.019) + 0.03*math.Cos(float64(i)*0.047))
	}
	cpuK, cpuV := make([][]float32, len(m.Layers)), make([][]float32, len(m.Layers))
	wantLayers := make([][]float32, len(m.Layers))
	for position := 0; position < 2; position++ {
		want := append([]float32(nil), embeddings[position*h:(position+1)*h]...)
		for layer := range m.Layers {
			want = m.ForwardLayer(want, layer, position, position, cpuK, cpuV)
			if position == 1 {
				wantLayers[layer] = append([]float32(nil), want...)
			}
		}
	}
	cpuOps := map[string][]float32{}
	oldOpHook := debugOpHook
	debugOpHook = func(backend string, step, layer int, op string, values []float32) {
		if backend == "cpu" && step == 1 && layer == 0 {
			cpuOps[op] = append([]float32(nil), values...)
		}
	}
	if _, err := m.GenerateFromEmbeddings([]int{0, 1}, embeddings, 1); err != nil {
		t.Fatal(err)
	}
	gpuOps := map[string][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, values []float32) {
		if backend == "gpu" && step == 1 && layer == 0 {
			gpuOps[op] = append([]float32(nil), values...)
		}
	}
	defer func() { debugOpHook = oldOpHook }()

	gotLayers := make([][]float32, len(m.Layers))
	oldHook := debugLayerHook
	debugLayerHook = func(backend string, step, layer int, values []float32) {
		if backend == "gpu" && step == 1 && layer >= 0 && layer < len(gotLayers) {
			gotLayers[layer] = append([]float32(nil), values[:h]...)
		}
	}
	defer func() { debugLayerHook = oldHook }()
	if _, err := g.GenerateFromEmbeddingsUntil([]int{0, 1}, embeddings, 1, -1); err != nil {
		t.Fatal(err)
	}
	for _, op := range []string{"hidden_in", "normed", "q", "k", "v", "q_qknorm", "k_qknorm", "q_attn", "k_attn", "attn", "o", "mlp_input", "gate_pre", "up", "gate_act", "down", "hidden_post_ffn"} {
		cpu, gpu := cpuOps[op], gpuOps[op]
		if len(cpu) > 0 && len(cpu) == len(gpu) {
			maxDiff, index := maxAbsDiffModel(gpu, cpu)
			t.Logf("op=%s max_abs_diff=%.6g index=%d", op, maxDiff, index)
		}
	}
	for layer := range wantLayers {
		if len(gotLayers[layer]) != h {
			t.Fatalf("missing GPU layer %d capture", layer)
		}
		maxDiff, index := maxAbsDiffModel(gotLayers[layer], wantLayers[layer])
		t.Logf("layer=%d max_abs_diff=%.6g index=%d", layer, maxDiff, index)
		if maxDiff > 3e-3 {
			t.Fatalf("layer %d GPU drift %.6g exceeds 3e-3", layer, maxDiff)
		}
	}
}

func maxAbsDiffModel(got, want []float32) (float64, int) {
	maxDiff, index := float64(0), -1
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > maxDiff {
			maxDiff, index = diff, i
		}
	}
	return maxDiff, index
}
