package model

import (
	"testing"
)

// buildGemma4PLITestModel builds a small synthetic Gemma4-style model that
// exercises the per-layer-input (PLI) path, which is the most allocation-heavy
// branch of the sequential decode loop.
func buildGemma4PLITestModel() *LlamaModel {
	const (
		h        = 16
		numHeads = 2
		numKV    = 1
		headDim  = h / numHeads
		inter    = 2 * h
		vocab    = 32
		layers   = 2
		hpl      = 4
	)
	qDim := numHeads * headDim
	kvDim := numKV * headDim
	totalDim := layers * hpl
	g := &lcg{s: 999}

	cfg := LlamaConfig{
		ModelType:      "gemma4_text",
		HiddenSize:     h,
		NumHeads:       numHeads,
		NumKVHeads:     numKV,
		HeadDim:        headDim,
		Intermediate:   inter,
		VocabSize:      vocab,
		NumLayers:      layers,
		RMSNormEps:     1e-6,
		MaxSeqLen:      64,
		RopeTheta:      10000,
		HiddenAct:      "gelu_pytorch_tanh",
		HiddenPerLayer: hpl,
		VocabPerLayer:  vocab,
	}
	m := &LlamaModel{Config: cfg, Large: false}
	m.EmbedTokens = g.tensor(vocab, h)
	m.Norm = g.tensor(h)
	m.LMHead = g.tensor(vocab, h)
	m.precomputeRoPE()

	// Per-layer-input projection front-end.
	m.PerLayerModelProj = make([]float32, totalDim*h) // gemvNT [totalDim, h]
	for i := range m.PerLayerModelProj {
		m.PerLayerModelProj[i] = g.f()
	}
	m.PerLayerProjNorm = make([]float32, hpl)
	for i := range m.PerLayerProjNorm {
		m.PerLayerProjNorm[i] = g.f() + 1
	}
	m.PerLayerProjScale = 0.5
	m.PerLayerInputScale = 0.7
	m.EmbedPerLayerScale = 0.3

	m.Layers = make([]LlamaLayer, layers)
	for l := range m.Layers {
		L := &m.Layers[l]
		L.HasKV = true
		L.LayerScalar = 1.0
		L.InputNorm = g.tensor(h)
		L.PostNorm = g.tensor(h)
		L.PreFFNNorm = g.tensor(h)
		L.PostFFNNorm = g.tensor(h)
		L.QW = g.tensor(h, qDim) // non-Large [inDim, outDim]
		L.KW = g.tensor(h, kvDim)
		L.VW = g.tensor(h, kvDim)
		L.OW = g.tensor(qDim, h)
		L.GateW = g.tensor(h, inter)
		L.UpW = g.tensor(h, inter)
		L.DownW = g.tensor(inter, h)
		// PLI gate/proj/norm.
		L.PLIGate = make([]float32, hpl*h) // gemvNT [hpl, h]
		for i := range L.PLIGate {
			L.PLIGate[i] = g.f()
		}
		L.PLIProj = make([]float32, h*hpl) // gemvNT [h, hpl]
		for i := range L.PLIProj {
			L.PLIProj[i] = g.f()
		}
		L.PLIPostNorm = make([]float32, h)
		for i := range L.PLIPostNorm {
			L.PLIPostNorm[i] = g.f() + 1
		}
	}
	return m
}

// TestDecodeLoopGolden pins the sequential decode output for representative
// synthetic models so that allocation/scratch-reuse refactors of the hot loop
// remain numerically identical. Prefill is disabled so this exercises the
// per-token sequential path exclusively.
func TestDecodeLoopGolden(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "1")
	prompt := []int{1, 5, 9, 2, 7, 3}
	const maxNew = 6

	cases := []struct {
		name  string
		build func() *LlamaModel
		want  []int
	}{
		{"llama_plain", func() *LlamaModel { return buildPrefillTestModel("llama", false, false, false) },
			[]int{1, 5, 9, 2, 7, 3, 21, 29, 7, 17, 17, 17}},
		{"qwen3_qknorm", func() *LlamaModel { return buildPrefillTestModel("qwen3", false, true, false) },
			[]int{1, 5, 9, 2, 7, 3, 21, 29, 7, 17, 17, 17}},
		{"gemma3_preffn", func() *LlamaModel { return buildPrefillTestModel("gemma3_text", false, false, true) },
			[]int{1, 5, 9, 2, 7, 3, 21, 29, 7, 19, 1, 24}},
		{"gemma4_pli", buildGemma4PLITestModel,
			[]int{1, 5, 9, 2, 7, 3, 6, 9, 12, 7, 6, 9}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.build()
			got := m.Generate(append([]int(nil), prompt...), maxNew)
			if tc.want != nil && !equalInts(got, tc.want) {
				t.Fatalf("decode output changed\n got=%v\nwant=%v", got, tc.want)
			}
			if tc.want == nil {
				t.Logf("%s => %v", tc.name, got)
			}
		})
	}
}
