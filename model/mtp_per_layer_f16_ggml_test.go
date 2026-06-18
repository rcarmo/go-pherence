//go:build ggml && cgo && linux

package model

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/backends/ggmlcompute"
	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGemma4PerLayerTokenEmbeddingRealGGMLGetRowsOracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	tensor, ok := g.TensorByName("per_layer_token_embd.weight")
	if !ok {
		t.Fatal("per_layer_token_embd.weight not found")
	}
	raw, err := g.Raw(tensor)
	if err != nil {
		t.Fatal(err)
	}
	if m.EmbedPerLayerGGUF == nil {
		t.Fatal("Go per-layer GGUF embedding not loaded")
	}
	probeTokens := []int{10979, 236764, 564, 236789}
	for _, tok := range probeTokens {
		gotGGML := make([]float32, m.EmbedPerLayerGGUF.InDim)
		if err := ggmlcompute.GetRowToF32(int(tensor.QType), gotGGML, raw, m.EmbedPerLayerGGUF.InDim, m.EmbedPerLayerGGUF.OutDim, tok); err != nil {
			t.Fatalf("ggml get row token %d: %v", tok, err)
		}
		gotGo := make([]float32, m.EmbedPerLayerGGUF.InDim)
		if err := m.EmbedPerLayerGGUF.DequantRowTo(gotGo, tok); err != nil {
			t.Fatalf("go dequant row token %d: %v", tok, err)
		}
		maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotGo)
		t.Logf("per_layer_token_embd token=%d ggml-vs-go max=%g mean=%g", tok, maxDiff, meanDiff)
		if maxDiff > 1e-6 {
			t.Fatalf("per_layer_token_embd token=%d ggml-vs-go max=%g mean=%g", tok, maxDiff, meanDiff)
		}
	}
}

func TestGemma4PerLayerModelProjRealGGMLF16Oracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	tensor, ok := g.TensorByName("per_layer_model_proj.weight")
	if !ok {
		t.Fatal("per_layer_model_proj.weight not found")
	}
	if tensor.QType != gguf.QuantF16 {
		t.Fatalf("per_layer_model_proj.weight type=%s, want F16", tensor.QType)
	}
	if len(tensor.Shape) != 2 || int(tensor.Shape[0]) != m.Config.HiddenSize || int(tensor.Shape[1]) != m.Config.NumLayers*m.Config.HiddenPerLayer {
		t.Fatalf("per_layer_model_proj.weight shape=%v does not match hidden/layers/hpl=%d/%d/%d", tensor.Shape, m.Config.HiddenSize, m.Config.NumLayers, m.Config.HiddenPerLayer)
	}
	raw, err := g.Raw(tensor)
	if err != nil {
		t.Fatal(err)
	}
	wF16 := make([]uint16, len(raw)/2)
	for i := range wF16 {
		wF16[i] = binary.LittleEndian.Uint16(raw[i*2:])
	}

	fx := []int{10979, 236764}
	if fixturePath := os.Getenv("GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE"); fixturePath != "" {
		_ = fixturePath // this test only needs the canonical first prompt token, kept for env discoverability
	}
	hidden := make([]float32, m.Config.HiddenSize)
	if err := m.ScaledTokenEmbeddingInto(hidden, fx[0]); err != nil {
		t.Fatal(err)
	}
	totalDim := m.Config.NumLayers * m.Config.HiddenPerLayer
	gotGGML := make([]float32, totalDim)
	if err := ggmlcompute.MulMatF16F32(gotGGML, wF16, hidden, m.Config.HiddenSize, totalDim); err != nil {
		t.Fatal(err)
	}
	gotGo := make([]float32, totalDim)
	gemvNT(gotGo, hidden, m.PerLayerModelProj, m.Config.HiddenSize, totalDim)
	hiddenRounded := make([]float32, len(hidden))
	for i, v := range hidden {
		hiddenRounded[i] = half.F16ToF32(half.F32ToF16(v))
	}
	gotGoRoundedInput := make([]float32, totalDim)
	gemvNT(gotGoRoundedInput, hiddenRounded, m.PerLayerModelProj, m.Config.HiddenSize, totalDim)

	maxGo, meanGo := maxMeanAbsDiff(gotGGML, gotGo)
	maxRounded, meanRounded := maxMeanAbsDiff(gotGGML, gotGoRoundedInput)
	t.Logf("real per_layer_model_proj ggml-vs-go max=%g mean=%g; ggml-vs-rounded-input max=%g mean=%g", maxGo, meanGo, maxRounded, meanRounded)
	if maxRounded > maxGo {
		t.Fatalf("rounded-input oracle is farther from ggml than current Go: rounded max=%g current max=%g", maxRounded, maxGo)
	}
	if maxRounded > 1e-4 {
		t.Fatalf("rounded-input oracle did not match ggml closely enough: max=%g mean=%g", maxRounded, meanRounded)
	}
	if maxGo < 1e-6 {
		t.Fatal("current Go unexpectedly matches ggml F16 mul_mat exactly; oracle no longer distinguishes input rounding")
	}
}

func maxMeanAbsDiff(a, b []float32) (float64, float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var max, sum float64
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i] - b[i]))
		if d > max {
			max = d
		}
		sum += d
	}
	if n == 0 {
		return 0, 0
	}
	return max, sum / float64(n)
}
