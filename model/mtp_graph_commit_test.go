package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestMTPVerifierResultCommitGraphFloatKV(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2},
		Layers: []LlamaLayer{{HasKV: true}},
	}
	d := validProjectionOnlyDrafter()
	state, err := NewMTPDrafterState(0, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := NewMTPExecutionGraph(m, d, state, nil, []int{1, 2}, 3)
	if err != nil {
		t.Fatal(err)
	}
	result, err := NewMTPVerifierResultForModel(m, 0, []int{1, 2}, [][]float32{
		{0, 9, 0, 0}, // accepts draft 1
		{0, 0, 0, 8}, // rejects draft 2 and emits bonus 3
		{7, 0, 0, 0}, // unused unless all accepted
	}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	kvCacheK := [][]float32{{1, 2, 3, 4, 5, 6}}
	kvCacheV := [][]float32{{7, 8, 9, 10, 11, 12}}
	cp := kv.CheckpointFloatKV(kvCacheK, kvCacheV)
	kvCacheK[0] = append(kvCacheK[0], 100, 101, 200, 201, 300, 301)
	kvCacheV[0] = append(kvCacheV[0], 400, 401, 500, 501, 600, 601)
	commit, err := result.CommitGraphFloatKV(m, graph, kvCacheK, kvCacheV, cp)
	if err != nil {
		t.Fatal(err)
	}
	if commit.KeepTokens != 2 || !sameInts(commit.Positions, []int{3, 4}) || !sameInts(commit.OutputTokens, []int{1, 3}) {
		t.Fatalf("commit=%+v", commit)
	}
	wantK := []float32{1, 2, 3, 4, 5, 6, 100, 101, 200, 201}
	wantV := []float32{7, 8, 9, 10, 11, 12, 400, 401, 500, 501}
	if !sameFloat32s(kvCacheK[0], wantK) || !sameFloat32s(kvCacheV[0], wantV) {
		t.Fatalf("KV K=%v V=%v", kvCacheK[0], kvCacheV[0])
	}
}

func TestMTPVerifierResultCommitGraphRejectsMismatchedGraph(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	d := validProjectionOnlyDrafter()
	state, err := NewMTPDrafterState(0, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := NewMTPExecutionGraph(m, d, state, nil, []int{1, 2}, 3)
	if err != nil {
		t.Fatal(err)
	}
	result, err := NewMTPVerifierResultForModel(m, 0, []int{1}, [][]float32{{0, 9, 0, 0}, {7, 0, 0, 0}}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := result.CommitGraphFloatKV(m, graph, [][]float32{{}}, [][]float32{{}}, kv.FloatKVCheckpoint{}); err == nil {
		t.Fatal("accepted mismatched graph/result")
	}
	if _, err := result.CommitGraphCompressedKV(graph, nil, nil); err == nil {
		t.Fatal("compressed graph commit accepted mismatched graph/result")
	}
}
