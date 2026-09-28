package model

import (
	"context"
	"reflect"
	"testing"
)

func TestGemma4IndependentBranchBatchMatchesCheckpointOracle(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_PURE_FLASH", "0")
	m := newGemma4SingleLayerDecodeSessionTestModel()
	prompt := []int{1, 2}
	branches := [][]int{{0}, {1}, {2, 0}, {2, 1}, {0, 2, 1}}

	batchedSession := newGemma4DecodeSessionForTest(t, m, 3, Gemma4PromptCacheConfig{})
	if err := batchedSession.BeginPreparedPrefill(prompt); err != nil {
		t.Fatal(err)
	}
	if _, err := batchedSession.PrefillNext(batchedSession.RemainingPrefill()); err != nil {
		t.Fatal(err)
	}
	trunkK := cloneFloat32Matrix(batchedSession.state.kvCacheK)
	trunkV := cloneFloat32Matrix(batchedSession.state.kvCacheV)
	output := batchedSession.OutputTokens()
	got, err := batchedSession.ScoreIndependentBranches(context.Background(), branches)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(batchedSession.OutputTokens(), output) || !reflect.DeepEqual(batchedSession.state.kvCacheK, trunkK) || !reflect.DeepEqual(batchedSession.state.kvCacheV, trunkV) {
		t.Fatal("independent branch batch mutated the session trunk")
	}

	oracle := newGemma4DecodeSessionForTest(t, m, 3, Gemma4PromptCacheConfig{})
	if err := oracle.BeginPreparedPrefill(prompt); err != nil {
		t.Fatal(err)
	}
	if _, err := oracle.PrefillNext(oracle.RemainingPrefill()); err != nil {
		t.Fatal(err)
	}
	cp, err := oracle.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	for i, branch := range branches {
		if err := oracle.Restore(cp); err != nil {
			t.Fatal(err)
		}
		var want []float32
		for _, token := range branch {
			step, err := oracle.AppendToken(token)
			if err != nil {
				t.Fatalf("oracle branch %d: %v", i, err)
			}
			want = step.Logits
		}
		assertFloat32RowsClose(t, got.Logits[i], want, 2e-5)
	}
}

func TestGemma4IndependentBranchBatchDoesNotCrossAttend(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_PURE_FLASH", "0")
	m := newGemma4SingleLayerDecodeSessionTestModel()
	s := newGemma4DecodeSessionForTest(t, m, 2, Gemma4PromptCacheConfig{})
	if err := s.BeginPreparedPrefill([]int{1}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PrefillNext(s.RemainingPrefill()); err != nil {
		t.Fatal(err)
	}
	alone, err := s.ScoreIndependentBranches(context.Background(), [][]int{{2, 1}})
	if err != nil {
		t.Fatal(err)
	}
	withSibling, err := s.ScoreIndependentBranches(context.Background(), [][]int{{0, 0}, {2, 1}, {1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32RowsClose(t, withSibling.Logits[1], alone.Logits[0], 0)
}

func TestGemma4IndependentBranchBatchSlidingWindowMatchesCheckpointOracle(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_PURE_FLASH", "0")
	m := newGemma4SingleLayerDecodeSessionTestModel()
	m.Config.SlidingWindow = 2
	m.Config.LayerTypes = []string{"sliding_attention"}
	prompt := []int{1, 2, 0}
	branches := [][]int{{0, 1}, {1, 2}, {2, 0}}
	batchSession := newGemma4DecodeSessionForTest(t, m, 2, Gemma4PromptCacheConfig{})
	if err := batchSession.BeginPreparedPrefill(prompt); err != nil {
		t.Fatal(err)
	}
	if _, err := batchSession.PrefillNext(batchSession.RemainingPrefill()); err != nil {
		t.Fatal(err)
	}
	got, err := batchSession.ScoreIndependentBranches(context.Background(), branches)
	if err != nil {
		t.Fatal(err)
	}
	oracle := newGemma4DecodeSessionForTest(t, m, 2, Gemma4PromptCacheConfig{})
	if err := oracle.BeginPreparedPrefill(prompt); err != nil {
		t.Fatal(err)
	}
	if _, err := oracle.PrefillNext(oracle.RemainingPrefill()); err != nil {
		t.Fatal(err)
	}
	cp, err := oracle.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	for i, branch := range branches {
		if err := oracle.Restore(cp); err != nil {
			t.Fatal(err)
		}
		var want []float32
		for _, token := range branch {
			step, err := oracle.AppendToken(token)
			if err != nil {
				t.Fatal(err)
			}
			want = step.Logits
		}
		assertFloat32RowsClose(t, got.Logits[i], want, 2e-5)
	}
}

func TestGemma4IndependentBranchBatchCancellationAndValidation(t *testing.T) {
	m := newGemma4SingleLayerDecodeSessionTestModel()
	s := newGemma4DecodeSessionForTest(t, m, 2, Gemma4PromptCacheConfig{})
	if _, err := s.ScoreIndependentBranches(context.Background(), [][]int{{0}}); err == nil {
		t.Fatal("scored before prefill")
	}
	if err := s.BeginPreparedPrefill([]int{1}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PrefillNext(s.RemainingPrefill()); err != nil {
		t.Fatal(err)
	}
	for _, branches := range [][][]int{nil, {{}}, {{-1}}, {{m.Config.VocabSize}}, {{0, 1, 2}}} {
		if _, err := s.ScoreIndependentBranches(context.Background(), branches); err == nil {
			t.Fatalf("accepted branches=%v", branches)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.ScoreIndependentBranches(ctx, [][]int{{0}, {1}}); err == nil {
		t.Fatal("accepted cancelled branch batch")
	}
}

func TestGatherGemma4BranchAttentionUsesOnlyTrunkAndOwnSuffix(t *testing.T) {
	trunkK := []float32{10, 11, 20, 21}
	trunkV := []float32{12, 13, 22, 23}
	suffixK := []float32{30, 31, 40, 41}
	suffixV := []float32{32, 33, 42, 43}
	gotK, gotV, err := gatherGemma4BranchAttention(2, 1, 4, 2, trunkK, trunkV, suffixK, suffixV)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotK, []float32{20, 21, 30, 31, 40, 41}) || !reflect.DeepEqual(gotV, []float32{22, 23, 32, 33, 42, 43}) {
		t.Fatalf("gather K/V=%v/%v", gotK, gotV)
	}
	gotK[0] = -1
	if trunkK[2] != 20 || suffixK[0] != 30 {
		t.Fatal("gather output aliases trunk or suffix")
	}
	if _, _, err := gatherGemma4BranchAttention(2, 0, 5, 2, trunkK, trunkV, suffixK, suffixV); err == nil {
		t.Fatal("accepted range beyond branch suffix")
	}
}

func assertFloat32RowsClose(t *testing.T, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("row len=%d want %d", len(got), len(want))
	}
	for i := range got {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d > tolerance {
			t.Fatalf("row[%d]=%g want %g tolerance=%g", i, got[i], want[i], tolerance)
		}
	}
}
