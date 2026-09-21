package qev

import (
	"context"
	"testing"

	"github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestGemma4CPUScorerRestoresTrunkBetweenBranches(t *testing.T) {
	m := qevZeroLayerModel()
	scorer := &Gemma4CPUScorer{Model: m}
	prompt := []int{1}
	branches := []Branch{
		{Tokens: []int{0}, CandidateTokens: []int{0, 1}},
		{Tokens: []int{1}, CandidateTokens: []int{0, 1}},
	}
	got, err := scorer.ScoreContext(context.Background(), prompt, branches, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || len(got[0]) != 2 || len(got[1]) != 2 {
		t.Fatalf("scores=%v", got)
	}
	// Zero-layer fixture logits depend only on the appended token embedding. If
	// branch restore leaked state, the second row would not match this result.
	if got[0][0] != 0 || got[0][1] <= 1 || got[1][0] != got[1][1] || got[1][0] <= 1 {
		t.Fatalf("scores=%v", got)
	}
}

func TestGemma4SIMDBatchScorerMatchesCPUOracle(t *testing.T) {
	m := qevSingleLayerModel()
	prompt := []int{1, 2}
	branches := []Branch{
		{Tokens: []int{0}, CandidateTokens: []int{0, 1, 2}},
		{Tokens: []int{1}, CandidateTokens: []int{0, 2}},
		{Tokens: []int{2, 0}, CandidateTokens: []int{1, 2}},
		{Tokens: []int{2, 1}, CandidateTokens: []int{0, 1}},
	}
	want, err := (&Gemma4CPUScorer{Model: m}).ScoreContext(context.Background(), prompt, branches, false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := (&Gemma4SIMDBatchScorer{Model: m}).ScoreContext(context.Background(), prompt, branches, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("rows=%d want %d", len(got), len(want))
	}
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("row %d len=%d want %d", i, len(got[i]), len(want[i]))
		}
		for j := range got[i] {
			delta := got[i][j] - want[i][j]
			if delta < 0 {
				delta = -delta
			}
			if delta > 2e-5 {
				t.Fatalf("score[%d][%d]=%g want %g", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestGemma4SIMDBatchScorerChunksAtDecisionSequenceBound(t *testing.T) {
	m := qevSingleLayerModel()
	branches := make([]Branch, DecisionSequences+3)
	for i := range branches {
		branches[i] = Branch{Tokens: []int{i % m.Config.VocabSize}, CandidateTokens: []int{0, 1, 2}}
	}
	want, err := (&Gemma4CPUScorer{Model: m}).ScoreContext(context.Background(), []int{1}, branches, false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := (&Gemma4SIMDBatchScorer{Model: m}).ScoreContext(context.Background(), []int{1}, branches, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		for j := range got[i] {
			delta := got[i][j] - want[i][j]
			if delta < 0 {
				delta = -delta
			}
			if delta > 2e-5 {
				t.Fatalf("score[%d][%d]=%g want %g", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestGemma4CPUScorerValidation(t *testing.T) {
	m := qevZeroLayerModel()
	s := &Gemma4CPUScorer{Model: m}
	for _, tc := range []struct {
		prompt   []int
		branches []Branch
	}{
		{nil, []Branch{{Tokens: []int{0}, CandidateTokens: []int{1}}}},
		{[]int{1}, []Branch{{CandidateTokens: []int{1}}}},
		{[]int{1}, []Branch{{Tokens: []int{0}}}},
		{[]int{1}, []Branch{{Tokens: []int{3}, CandidateTokens: []int{1}}}},
	} {
		if _, err := s.ScoreContext(context.Background(), tc.prompt, tc.branches, false); err == nil {
			t.Fatalf("accepted prompt=%v branches=%v", tc.prompt, tc.branches)
		}
	}
}

func qevSingleLayerModel() *model.LlamaModel {
	m := qevZeroLayerModel()
	m.Config.NumLayers = 1
	m.Config.Intermediate = 2
	m.Config.RMSNormEps = 1e-6
	m.Config.HiddenAct = "gelu_pytorch_tanh"
	identity := tensor.FromFloat32([]float32{1, 0, 0, 1}, []int{2, 2})
	m.Layers = []model.LlamaLayer{{
		InputNorm: tensor.Ones([]int{2}), PostNorm: tensor.Ones([]int{2}), PostFFNNorm: tensor.Ones([]int{2}), LayerScalar: 1, HasKV: true,
		QW: identity, KW: identity, VW: identity, OW: identity,
		GateW: identity, UpW: identity, DownW: identity,
		QNorm: tensor.Ones([]int{2}), KNorm: tensor.Ones([]int{2}),
	}}
	return m
}

func qevZeroLayerModel() *model.LlamaModel {
	return &model.LlamaModel{
		Config: model.LlamaConfig{ModelType: "gemma4_text", VocabSize: 3, HiddenSize: 2, NumHeads: 1, NumKVHeads: 1, HeadDim: 2},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, []int{3, 2}),
		Norm: tensor.Ones([]int{2}),
		LMHead: tensor.FromFloat32([]float32{
			0, 1,
			1, 1,
			1, 0,
		}, []int{3, 2}),
	}
}
