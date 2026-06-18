package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

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

func TestMapGemma4MTPDrafterKVSourceLayersUsesLlamaCppSharedTargets(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{ModelType: "gemma4_text", NumLayers: 6, NumKVHeads: 2, NumGlobalKVHeads: 1, HeadDim: 2, GlobalHeadDim: 4, LayerTypes: []string{"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"}},
		Layers: []LlamaLayer{{HasKV: true}, {HasKV: true}, {HasKV: true}, {HasKV: true}, {HasKV: true}, {HasKV: true}},
	}
	d := validDrafterStepScaffold()
	d.Config.ModelType = "gemma4_text"
	d.Config.NumLayers = 4
	d.Config.LayerTypes = []string{"sliding_attention", "sliding_attention", "full_attention", "sliding_attention"}
	d.Config.NumKVHeads = 2
	d.Config.NumGlobalKVHeads = 1
	d.Config.HeadDim = 2
	d.Config.GlobalHeadDim = 4
	d.Layers = make([]Gemma4MTPDrafterLayer, d.Config.NumLayers)
	for i := range d.Layers {
		headDim := d.Config.HeadDim
		if d.Config.LayerTypes[i] == "full_attention" {
			headDim = d.Config.GlobalHeadDim
		}
		qDim := d.Config.NumHeads * headDim
		d.Layers[i] = Gemma4MTPDrafterLayer{
			InputNorm:     tensor.Ones([]int{d.Config.HiddenSize}),
			PostNorm:      tensor.Ones([]int{d.Config.HiddenSize}),
			PreFFNNorm:    tensor.Ones([]int{d.Config.HiddenSize}),
			PostFFNNorm:   tensor.Ones([]int{d.Config.HiddenSize}),
			QNorm:         tensor.Ones([]int{headDim}),
			LayerScalar:   1,
			HeadDimLocal:  headDim,
			KVSourceLayer: -1,
			QW:            make([]float32, qDim*d.Config.HiddenSize),
			OW:            make([]float32, d.Config.HiddenSize*qDim),
			GateW:         make([]float32, d.Config.Intermediate*d.Config.HiddenSize),
			UpW:           make([]float32, d.Config.Intermediate*d.Config.HiddenSize),
			DownW:         make([]float32, d.Config.HiddenSize*d.Config.Intermediate),
		}
	}
	sources, err := MapMTPDrafterKVSourceLayersByWidth(m, d, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(sources, []int{4, 4, 5, 4}) {
		t.Fatalf("Gemma4 sources=%v, want shared llama.cpp targets [4 4 5 4]", sources)
	}
	ext := &MTPDrafterExternalKV{K: make([][]float32, 6), V: make([][]float32, 6), SourceLayers: sources, SeqLen: 3}
	ext.K[4], ext.V[4] = make([]float32, 3*4), make([]float32, 3*4)
	ext.K[5], ext.V[5] = make([]float32, 3*4), make([]float32, 3*4)
	if err := validateMTPDrafterExternalKV(d, ext); err != nil {
		t.Fatalf("Gemma4 shared external KV rejected: %v", err)
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
