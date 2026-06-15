package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestCPUDecodeStateCommitGraphAcceptedFloatKV(t *testing.T) {
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
	verifier, err := NewMTPVerifierResultForModel(m, 0, []int{1, 2}, [][]float32{
		{0, 9, 0, 0},
		{0, 0, 0, 8},
		{7, 0, 0, 0},
	}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	st := &CPUDecodeState{
		Model:    m,
		Output:   []int{10, 11, 12},
		KVCacheK: [][]float32{{1, 2, 3, 4, 5, 6}},
		KVCacheV: [][]float32{{7, 8, 9, 10, 11, 12}},
		KVDims:   []int{2},
	}
	cp := st.Checkpoint()
	st.KVCacheK[0] = append(st.KVCacheK[0], 100, 101, 200, 201, 300, 301)
	st.KVCacheV[0] = append(st.KVCacheV[0], 400, 401, 500, 501, 600, 601)
	commit, err := st.CommitGraphAccepted(cp, graph, verifier)
	if err != nil {
		t.Fatal(err)
	}
	if commit.KeepTokens != 2 || !sameInts(commit.OutputTokens, []int{1, 3}) || !sameInts(commit.Positions, []int{3, 4}) {
		t.Fatalf("commit=%+v", commit)
	}
	if !sameInts(st.Output, []int{10, 11, 12, 1, 3}) {
		t.Fatalf("output=%v", st.Output)
	}
	if !sameFloat32s(st.KVCacheK[0], []float32{1, 2, 3, 4, 5, 6, 100, 101, 200, 201}) || !sameFloat32s(st.KVCacheV[0], []float32{7, 8, 9, 10, 11, 12, 400, 401, 500, 501}) {
		t.Fatalf("KV K=%v V=%v", st.KVCacheK[0], st.KVCacheV[0])
	}
}

func TestCPUDecodeStateCommitGraphAcceptedRejectsCheckpointGraphCursorMismatch(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	d := validProjectionOnlyDrafter()
	state, err := NewMTPDrafterState(0, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := NewMTPExecutionGraph(m, d, state, nil, []int{1}, 3)
	if err != nil {
		t.Fatal(err)
	}
	verifier, err := NewMTPVerifierResultForModel(m, 0, []int{1}, [][]float32{{0, 9, 0, 0}, {7, 0, 0, 0}}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	st := &CPUDecodeState{Model: m, Output: []int{10, 11}, KVCacheK: [][]float32{{}}, KVCacheV: [][]float32{{}}, KVDims: []int{2}}
	cp := CPUDecodeCheckpoint{OutputLen: len(st.Output), FloatKV: kv.CheckpointFloatKV(st.KVCacheK, st.KVCacheV)}
	if _, err := st.CommitGraphAccepted(cp, graph, verifier); err == nil {
		t.Fatal("accepted graph whose start position does not match checkpoint cursor")
	}
	if !sameInts(st.Output, []int{10, 11}) {
		t.Fatalf("output mutated on failed cursor check: %v", st.Output)
	}
}

func TestCPUDecodeStateCommitGraphAcceptedRejectsMismatchWithoutOutputMutation(t *testing.T) {
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
	verifier, err := NewMTPVerifierResultForModel(m, 0, []int{1}, [][]float32{{0, 9, 0, 0}, {7, 0, 0, 0}}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	st := &CPUDecodeState{Model: m, Output: []int{10, 11}, KVCacheK: [][]float32{{}}, KVCacheV: [][]float32{{}}, KVDims: []int{2}}
	cp := CPUDecodeCheckpoint{OutputLen: len(st.Output), FloatKV: kv.CheckpointFloatKV(st.KVCacheK, st.KVCacheV)}
	if _, err := st.CommitGraphAccepted(cp, graph, verifier); err == nil {
		t.Fatal("accepted mismatched graph/result")
	}
	if !sameInts(st.Output, []int{10, 11}) {
		t.Fatalf("output mutated on failed graph commit: %v", st.Output)
	}
}

func TestCPUDecodeStateCommitGraphAcceptedCompressedKVRejectsModelDriftWithoutOutputMutation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	graph := MTPExecutionGraph{
		InputToken:    9,
		DraftedTokens: []int{1},
		StartPos:      3,
		DrafterSteps: []MTPDrafterGraphStep{{
			Index:           0,
			InputToken:      9,
			ActivationWidth: 2,
		}},
		Verifier: MTPVerifierPlan{
			InputToken:     9,
			DraftedTokens:  []int{1},
			VerifierTokens: []int{9, 1},
			StartPos:       3,
			Positions:      []int{3, 4},
		},
		MaxKVKeepTokens: 2,
	}
	verifier, err := NewMTPVerifierResult(9, []int{1}, [][]float32{
		{0, 8, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 7, 0, 0, 0, 0, 0, 0, 0},
	}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	st := &CPUDecodeState{
		Model:        m,
		Output:       []int{10, 11, 12},
		CompressedKV: []*kv.CompressedKVCache{},
		KVDims:       []int{2},
	}
	cp := st.Checkpoint()
	if _, err := st.CommitGraphAccepted(cp, graph, verifier); err == nil {
		t.Fatal("compressed graph commit accepted verifier rows outside owning model dims")
	}
	if !sameInts(st.Output, []int{10, 11, 12}) {
		t.Fatalf("output mutated on failed compressed graph commit: %v", st.Output)
	}
}
