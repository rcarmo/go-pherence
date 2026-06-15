package model

import "testing"

func TestNewCPUDecodeStateFromMTPPromptContext(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	ctx := MTPPromptContext{Tokens: []int{1, 2}, PreviousToken: 2, Activation: []float32{0.5, 0.25}, SeqLen: 2, KVCacheK: [][]float32{{1, 2, 3, 4}}, KVCacheV: [][]float32{{5, 6, 7, 8}}}
	st, err := NewCPUDecodeStateFromMTPPromptContext(m, ctx, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(st.Output, ctx.Tokens) || !sameFloat32s(st.KVCacheK[0], ctx.KVCacheK[0]) || !sameFloat32s(st.KVCacheV[0], ctx.KVCacheV[0]) {
		t.Fatalf("state output=%v K=%v V=%v", st.Output, st.KVCacheK, st.KVCacheV)
	}
	ctx.KVCacheK[0][0] = 99
	if st.KVCacheK[0][0] == 99 {
		t.Fatal("decode state aliases prompt context KV")
	}
}

func TestNewCPUDecodeStateFromMTPPromptContextValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	base := MTPPromptContext{Tokens: []int{1, 2}, PreviousToken: 2, Activation: []float32{0.5, 0.25}, SeqLen: 2, KVCacheK: [][]float32{{1, 2, 3, 4}}, KVCacheV: [][]float32{{5, 6, 7, 8}}}
	if _, err := NewCPUDecodeStateFromMTPPromptContext(nil, base, 1); err == nil {
		t.Fatal("accepted nil model")
	}
	bad := base
	bad.SeqLen = 1
	if _, err := NewCPUDecodeStateFromMTPPromptContext(m, bad, 1); err == nil {
		t.Fatal("accepted bad seqLen")
	}
	bad = base
	bad.Activation = []float32{1}
	if _, err := NewCPUDecodeStateFromMTPPromptContext(m, bad, 1); err == nil {
		t.Fatal("accepted bad activation")
	}
	bad = base
	bad.KVCacheK = [][]float32{{1, 2}}
	if _, err := NewCPUDecodeStateFromMTPPromptContext(m, bad, 1); err == nil {
		t.Fatal("accepted bad KV width")
	}
}

func TestGenerateMTPGraphFromPromptContext(t *testing.T) {
	m := newZeroLayerVerifierModel()
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	ctx := MTPPromptContext{Tokens: []int{1}, PreviousToken: 1, Activation: []float32{0.5, 0.25}, SeqLen: 1, KVCacheK: [][]float32{}, KVCacheV: [][]float32{}}
	res, err := m.GenerateMTPGraphFromPromptContext(d, ctx, nil, MTPGraphGenerationOptions{MaxTokens: 3, Policy: MTPAdaptiveDraftPolicy{InitialDrafts: 2, MaxDrafts: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) != 1 || res.Stats.Steps != 1 || res.Stats.DraftedTokens != 2 {
		t.Fatalf("result steps=%v stats=%+v", res.Steps, res.Stats)
	}
	if !sameInts(res.Output[:len(ctx.Tokens)], ctx.Tokens) {
		t.Fatalf("output prefix=%v want %v", res.Output, ctx.Tokens)
	}
	if len(res.Output) != len(ctx.Tokens)+len(res.Steps[0].OutputTokens) {
		t.Fatalf("output=%v step=%+v", res.Output, res.Steps[0])
	}
	if res.FinalState.PreviousToken < 0 || res.FinalState.PreviousToken >= m.Config.VocabSize {
		t.Fatalf("final state=%+v", res.FinalState)
	}
}

func TestGenerateMTPGraphFromPromptContextLeavesSingleTokenTail(t *testing.T) {
	m := newZeroLayerVerifierModel()
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	ctx := MTPPromptContext{Tokens: []int{1}, PreviousToken: 1, Activation: []float32{0.5, 0.25}, SeqLen: 1, KVCacheK: [][]float32{}, KVCacheV: [][]float32{}}
	res, err := m.GenerateMTPGraphFromPromptContext(d, ctx, nil, MTPGraphGenerationOptions{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) != 0 || !sameInts(res.Output, ctx.Tokens) {
		t.Fatalf("single-tail result=%+v", res)
	}
}
