package model

import (
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
)

func TestNewCPUTokenStateForLegacyGenerateAllocatesPlainFloatKV(t *testing.T) {
	m := newGemma4CPUTokenStateTestModel()
	prepared := []int{7, 8}
	st, err := newCPUTokenStateForLegacyGenerate(m, prepared, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(st.output, prepared) {
		t.Fatalf("output=%v want %v", st.output, prepared)
	}
	if cap(st.output) != 5 {
		t.Fatalf("output cap=%d want 5", cap(st.output))
	}
	prepared[0] = 99
	if st.output[0] != 7 {
		t.Fatalf("output did not copy prompt: %v", st.output)
	}
	if st.maxSequence != 5 || st.position != 2 {
		t.Fatalf("maxSequence/position=%d/%d want 5/2", st.maxSequence, st.position)
	}
	if st.compressedKV != nil {
		t.Fatalf("compressedKV=%v want nil plain float KV", st.compressedKV)
	}
	if len(st.kvCacheK) != 2 || len(st.kvCacheV) != 2 {
		t.Fatalf("KV layers K/V=%d/%d want 2/2", len(st.kvCacheK), len(st.kvCacheV))
	}
	if cap(st.kvCacheK[0]) != 10 || cap(st.kvCacheV[0]) != 10 {
		t.Fatalf("layer0 KV caps K/V=%d/%d want 10/10", cap(st.kvCacheK[0]), cap(st.kvCacheV[0]))
	}
	if cap(st.kvCacheK[1]) != 20 || cap(st.kvCacheV[1]) != 20 {
		t.Fatalf("layer1 KV caps K/V=%d/%d want 20/20", cap(st.kvCacheK[1]), cap(st.kvCacheV[1]))
	}
	if len(st.attnScoresScratch) != 5 || len(st.attnOutScratch) != 8 {
		t.Fatalf("attention scratch lens scores/out=%d/%d want 5/8", len(st.attnScoresScratch), len(st.attnOutScratch))
	}
	if len(st.hidden) != 8 || len(st.scratchResidual) != 8 || len(st.scratchO) != 8 || len(st.scratchMlp) != 8 || len(st.scratchDown) != 8 {
		t.Fatalf("hidden/residual/o/mlp/down lens=%d/%d/%d/%d/%d want 8s", len(st.hidden), len(st.scratchResidual), len(st.scratchO), len(st.scratchMlp), len(st.scratchDown))
	}
	if len(st.scratchQ) != 8 || len(st.scratchK) != 4 || len(st.scratchV) != 4 {
		t.Fatalf("Q/K/V scratch lens=%d/%d/%d want 8/4/4", len(st.scratchQ), len(st.scratchK), len(st.scratchV))
	}
	if len(st.scratchGate) != 10 || len(st.scratchUp) != 10 {
		t.Fatalf("gate/up scratch lens=%d/%d want 10/10", len(st.scratchGate), len(st.scratchUp))
	}
	if len(st.scratchPLIGate) != 3 || len(st.scratchPLIProj) != 8 || len(st.pliProjBuf) != 6 || len(st.pliSlices) != 2 {
		t.Fatalf("PLI scratch gate/proj/buf/slices=%d/%d/%d/%d want 3/8/6/2", len(st.scratchPLIGate), len(st.scratchPLIProj), len(st.pliProjBuf), len(st.pliSlices))
	}
}

func TestNewCPUTokenStateForLegacyGenerateSharedKVUsesSourceOwnership(t *testing.T) {
	m := newGemma4CPUTokenStateTestModel()
	m.Layers[1].HasKV = false
	m.Layers[1].KVSourceLayer = 0
	st, err := newCPUTokenStateForLegacyGenerate(m, []int{1, 2}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if cap(st.kvCacheK[0]) != 10 || cap(st.kvCacheV[0]) != 10 {
		t.Fatalf("source KV caps K/V=%d/%d want 10/10", cap(st.kvCacheK[0]), cap(st.kvCacheV[0]))
	}
	if cap(st.kvCacheK[1]) != 0 || cap(st.kvCacheV[1]) != 0 {
		t.Fatalf("shared KV caps K/V=%d/%d want 0/0", cap(st.kvCacheK[1]), cap(st.kvCacheV[1]))
	}
	if len(st.scratchK) != 4 || len(st.scratchV) != 4 {
		t.Fatalf("shared-KV scratch lens K/V=%d/%d want 4/4", len(st.scratchK), len(st.scratchV))
	}
}

func TestNewCPUTokenStateForLegacyGenerateRejectsOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	cases := []struct {
		name string
		m    *LlamaModel
		prep []int
		max  int
	}{
		{
			name: "output capacity",
			m:    &LlamaModel{Config: LlamaConfig{HiddenSize: 1, NumLayers: 0, NumHeads: 1, NumKVHeads: 1, HeadDim: 1, Intermediate: 0}},
			prep: []int{1},
			max:  maxInt,
		},
		{
			name: "kv dimension",
			m:    &LlamaModel{Config: LlamaConfig{HiddenSize: 1, NumLayers: 1, NumHeads: 1, NumKVHeads: maxInt/2 + 1, HeadDim: 3, Intermediate: 1}, Layers: []LlamaLayer{{HasKV: true}}},
			prep: []int{1},
			max:  1,
		},
		{
			name: "attention scratch",
			m:    &LlamaModel{Config: LlamaConfig{HiddenSize: 1, NumLayers: 1, NumHeads: maxInt/2 + 1, NumKVHeads: 1, HeadDim: 3, Intermediate: 1}, Layers: []LlamaLayer{{HasKV: true}}},
			prep: []int{1},
			max:  1,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := newCPUTokenStateForLegacyGenerate(tc.m, tc.prep, tc.max); err == nil {
				t.Fatal("accepted overflowing allocation")
			}
		})
	}
}

func TestNewCPUTokenStateForLegacyGenerateRejectsSharedKVScratchOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	m := &LlamaModel{
		Config: LlamaConfig{HiddenSize: 1, NumLayers: 1, NumHeads: 1, NumKVHeads: maxInt/2 + 1, HeadDim: 3, Intermediate: 1},
		Layers: []LlamaLayer{{HasKV: false, KVSourceLayer: 0}},
	}
	if _, err := newCPUTokenStateForLegacyGenerate(m, []int{1}, 1); err == nil {
		t.Fatal("accepted shared-KV scratch overflow")
	}
}

func newGemma4CPUTokenStateTestModel() *LlamaModel {
	return &LlamaModel{
		Config: LlamaConfig{
			ModelType:        "gemma4_text",
			HiddenSize:       8,
			NumLayers:        2,
			NumHeads:         2,
			NumKVHeads:       1,
			NumGlobalKVHeads: 1,
			HeadDim:          2,
			GlobalHeadDim:    4,
			Intermediate:     6,
			HiddenPerLayer:   3,
			LayerTypes:       []string{"sliding_attention", "full_attention"},
		},
		Layers: []LlamaLayer{
			{HasKV: true, HeadDimLocal: 2},
			{HasKV: true, HeadDimLocal: 4, GateWm: &mlx.QuantWeight{OutDim: 10}},
		},
		PerLayerModelProj: make([]float32, 2*3*8),
	}
}
