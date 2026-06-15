package model

import "testing"

func TestMapMTPDrafterKVSourceLayersByWidth(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{NumKVHeads: 2, NumGlobalKVHeads: 1, HeadDim: 2, GlobalHeadDim: 4, LayerTypes: []string{"sliding_attention", "full_attention"}},
		Layers: []LlamaLayer{{HasKV: true}, {HasKV: true}},
	}
	d := validDrafterStepScaffold()
	d.Config.NumLayers = 2
	d.Config.LayerTypes = []string{"sliding_attention", "full_attention"}
	d.Config.NumKVHeads = 2
	d.Config.NumGlobalKVHeads = 1
	d.Config.HeadDim = 2
	d.Config.GlobalHeadDim = 4
	d.Layers = []Gemma4MTPDrafterLayer{{KVSourceLayer: -1}, {KVSourceLayer: -1}}
	sources, err := MapMTPDrafterKVSourceLayersByWidth(m, d, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(sources, []int{0, 1}) {
		t.Fatalf("sources=%v", sources)
	}
}

func TestNewMTPDrafterExternalKVFromPromptContext(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	d := validDrafterStepScaffold()
	ctx := MTPPromptContext{SeqLen: 2, KVCacheK: [][]float32{{1, 2, 3, 4}}, KVCacheV: [][]float32{{5, 6, 7, 8}}}
	ext, err := NewMTPDrafterExternalKVFromPromptContext(m, d, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ext.SeqLen != 2 || !sameInts(ext.SourceLayers, []int{0}) || !sameFloat32s(ext.K[0], ctx.KVCacheK[0]) || !sameFloat32s(ext.V[0], ctx.KVCacheV[0]) {
		t.Fatalf("external KV=%+v", ext)
	}
}

func TestNewMTPDrafterExternalKVFromPromptContextValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	d := validDrafterStepScaffold()
	if _, err := NewMTPDrafterExternalKVFromPromptContext(m, d, MTPPromptContext{}); err == nil {
		t.Fatal("accepted empty prompt context")
	}
	badMain := &LlamaModel{Config: LlamaConfig{NumKVHeads: 1, HeadDim: 4}, Layers: []LlamaLayer{{HasKV: true}}}
	ctx := MTPPromptContext{SeqLen: 1, KVCacheK: [][]float32{{1, 2, 3, 4}}, KVCacheV: [][]float32{{5, 6, 7, 8}}}
	if _, err := NewMTPDrafterExternalKVFromPromptContext(badMain, d, ctx); err == nil {
		t.Fatal("accepted unmatched main/drafter KV widths")
	}
}
