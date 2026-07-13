package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

// lcg is a tiny deterministic generator for synthetic weights.
type lcg struct{ s uint32 }

func (g *lcg) f() float32 {
	g.s = g.s*1664525 + 1013904223
	// map to roughly [-0.1, 0.1]
	return (float32(g.s>>8)/float32(1<<24) - 0.5) * 0.2
}

func (g *lcg) tensor(shape ...int) *tensor.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = g.f()
	}
	return tensor.FromFloat32(data, shape)
}

// buildPrefillTestModel builds a small synthetic decoder whose weight layout
// matches the requested m.Large flag, so the sequential and prefill paths read
// identical weights.
func buildPrefillTestModel(modelType string, large, qkNorm, preFFN bool) *LlamaModel {
	const (
		h        = 16
		numHeads = 2
		numKV    = 1
		headDim  = h / numHeads // 8
		inter    = 2 * h
		vocab    = 32
		layers   = 2
	)
	qDim := numHeads * headDim
	kvDim := numKV * headDim
	g := &lcg{s: 12345}

	cfg := LlamaConfig{
		ModelType:    modelType,
		HiddenSize:   h,
		NumHeads:     numHeads,
		NumKVHeads:   numKV,
		HeadDim:      headDim,
		Intermediate: inter,
		VocabSize:    vocab,
		NumLayers:    layers,
		RMSNormEps:   1e-6,
		MaxSeqLen:    64,
		RopeTheta:    10000,
		HiddenAct:    "silu",
	}
	m := &LlamaModel{Config: cfg, Large: large}
	m.EmbedTokens = g.tensor(vocab, h)
	m.Norm = g.tensor(h)
	m.LMHead = g.tensor(vocab, h)
	m.precomputeRoPE()

	denseProj := func(inDim, outDim int) *tensor.Tensor {
		if large {
			return g.tensor(outDim, inDim) // gemvNT layout [outDim, inDim]
		}
		return g.tensor(inDim, outDim) // gemv layout [inDim, outDim]
	}

	m.Layers = make([]LlamaLayer, layers)
	for l := range m.Layers {
		L := &m.Layers[l]
		L.HasKV = true
		L.LayerScalar = 1.0
		L.InputNorm = g.tensor(h)
		L.PostNorm = g.tensor(h)
		L.QW = denseProj(h, qDim)
		L.KW = denseProj(h, kvDim)
		L.VW = denseProj(h, kvDim)
		L.OW = denseProj(qDim, h)
		L.GateW = denseProj(h, inter)
		L.UpW = denseProj(h, inter)
		L.DownW = denseProj(inter, h)
		if qkNorm {
			L.QNorm = g.tensor(headDim)
			L.KNorm = g.tensor(headDim)
		}
		if preFFN {
			L.PreFFNNorm = g.tensor(h)
			L.PostFFNNorm = g.tensor(h)
			L.VNorm = g.tensor(headDim)
		}
	}
	return m
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestCPUPrefillMatchesSequential verifies that batched CPU prefill produces
// byte-identical generated tokens to the sequential decode loop across model
// variants (QK-norm, Gemma-style pre-FFN norm/BF16, and the Large weight
// layout). The prompt is long enough (>=2 tokens) to engage prefill.
func TestCPUPrefillMatchesSequential(t *testing.T) {
	prompt := []int{1, 5, 9, 2, 7, 3}
	const maxNew = 5

	cases := []struct {
		name      string
		modelType string
		large     bool
		qkNorm    bool
		preFFN    bool
	}{
		{"llama_plain", "llama", false, false, false},
		{"llama_large", "llama", true, false, false},
		{"qwen3_qknorm", "qwen3", false, true, false},
		{"gemma3_preffn_bf16", "gemma3_text", false, false, true},
		{"gemma3_qknorm_preffn", "gemma3_text", false, true, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := buildPrefillTestModel(tc.modelType, tc.large, tc.qkNorm, tc.preFFN)
			if !m.prefillCPUEligible(len(prompt)) {
				t.Fatalf("model expected to be prefill-eligible")
			}

			// Sequential reference.
			t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "1")
			seq := m.Generate(append([]int(nil), prompt...), maxNew)

			// Prefill path.
			t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "0")
			pre := m.Generate(append([]int(nil), prompt...), maxNew)

			if !equalInts(seq, pre) {
				t.Fatalf("prefill output mismatch\n seq=%v\n pre=%v", seq, pre)
			}
			if len(pre) != len(prompt)+maxNew {
				t.Fatalf("unexpected output length=%d want %d", len(pre), len(prompt)+maxNew)
			}
		})
	}
}

func TestCPUPrefillEmbeddingsMatchesSequential(t *testing.T) {
	m := buildPrefillTestModel("qwen3", false, true, false)
	prompt := []int{1, 5, 9, 2, 7, 3}
	h := m.Config.HiddenSize
	embeddings := make([]float32, len(prompt)*h)
	for i, token := range prompt {
		if err := m.TokenEmbeddingInto(embeddings[i*h:(i+1)*h], token); err != nil {
			t.Fatal(err)
		}
	}
	// Replace two rows as a multimodal caller would.
	for i := 2 * h; i < 4*h; i++ {
		embeddings[i] = float32(i%h-8) * 0.03
	}
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "1")
	sequential, err := m.GenerateFromEmbeddings(prompt, embeddings, 5)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "0")
	prefilled, err := m.GenerateFromEmbeddings(prompt, embeddings, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !equalInts(sequential, prefilled) {
		t.Fatalf("embedding prefill mismatch\n sequential=%v\n prefilled=%v", sequential, prefilled)
	}
}

// TestCPUPrefillIneligibleCases confirms prefill declines unsupported configs.
func TestCPUPrefillIneligibleCases(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "0")

	// Single-token prompt: nothing to batch.
	m := buildPrefillTestModel("llama", false, false, false)
	if m.prefillCPUEligible(1) {
		t.Fatal("B=1 should be ineligible")
	}

	// MoE layer.
	m2 := buildPrefillTestModel("llama", false, false, false)
	m2.Layers[0].IsMoE = true
	if m2.prefillCPUEligible(4) {
		t.Fatal("MoE should be ineligible")
	}

	// Gemma4 per-layer inputs.
	m3 := buildPrefillTestModel("gemma4_text", false, false, false)
	if m3.prefillCPUEligible(4) {
		t.Fatal("gemma4_text should be ineligible")
	}

	// Shared-KV layer.
	m4 := buildPrefillTestModel("llama", false, false, false)
	m4.Layers[1].HasKV = false
	if m4.prefillCPUEligible(4) {
		t.Fatal("shared-KV should be ineligible")
	}

	// Kill switch.
	m5 := buildPrefillTestModel("llama", false, false, false)
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "1")
	if m5.prefillCPUEligible(4) {
		t.Fatal("kill switch should disable prefill")
	}
}

func TestCPUPrefillRejectsOverflowingScratchWithoutPanic(t *testing.T) {
	m := buildPrefillTestModel("llama", false, false, false)
	maxInt := int(^uint(0) >> 1)
	m.Layers[0].HeadDimLocal = maxInt/2 + 1
	kvK := make([][]float32, len(m.Layers))
	kvV := make([][]float32, len(m.Layers))
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("prefillCPU panicked on overflowing scratch dims: %v", r)
		}
	}()
	if hidden, ok := m.prefillCPU([]int{1, 2}, kvK, kvV); ok || hidden != nil {
		t.Fatalf("prefillCPU hidden=%v ok=%v, want fallback", hidden, ok)
	}
}

func TestCPUPrefillRejectsOverflowingBatchBuffersWithoutPanic(t *testing.T) {
	m := buildPrefillTestModel("llama", false, false, false)
	maxInt := int(^uint(0) >> 1)
	m.Config.HiddenSize = maxInt/2 + 1
	kvK := make([][]float32, len(m.Layers))
	kvV := make([][]float32, len(m.Layers))
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("prefillCPU panicked on overflowing batch dims: %v", r)
		}
	}()
	if hidden, ok := m.prefillCPU([]int{1, 2}, kvK, kvV); ok || hidden != nil {
		t.Fatalf("prefillCPU hidden=%v ok=%v, want fallback", hidden, ok)
	}
}

func TestGeneratePreparedDenseKEqVOmittedV(t *testing.T) {
	m := buildPrefillTestModel("llama", false, false, false)
	m.Config.AttentionKEqV = true
	for i := range m.Layers {
		m.Layers[i].VW = nil
	}
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "1")
	out := m.Generate([]int{1, 5, 9}, 2)
	if len(out) != 5 {
		t.Fatalf("Generate len=%d output=%v", len(out), out)
	}
}

func TestCPUPrefillDenseKEqVOmittedVMatchesSequential(t *testing.T) {
	prompt := []int{1, 5, 9, 2, 7, 3}
	m := buildPrefillTestModel("llama", false, false, false)
	m.Config.AttentionKEqV = true
	for i := range m.Layers {
		m.Layers[i].VW = nil
	}
	if !m.prefillCPUEligible(len(prompt)) {
		t.Fatal("omitted V K=V model should be prefill eligible")
	}
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "1")
	seq := m.Generate(append([]int(nil), prompt...), 3)
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "0")
	pre := m.Generate(append([]int(nil), prompt...), 3)
	if !equalInts(seq, pre) {
		t.Fatalf("omitted V K=V prefill mismatch seq=%v pre=%v", seq, pre)
	}
}
