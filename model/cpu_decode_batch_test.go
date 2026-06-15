package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestFinishCPUDecodeBatchMatchesRows(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 2, RMSNormEps: 0},
		Norm:   tensor.Ones([]int{2}),
		LMHead: tensor.FromFloat32([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, []int{3, 2}),
	}
	hiddenRows := [][]float32{{3, 4}, {5, 12}}
	acts, logits, tokens, err := m.FinishCPUDecodeBatch(hiddenRows)
	if err != nil {
		t.Fatal(err)
	}
	if len(acts) != 2 || len(logits) != 2 || len(tokens) != 2 {
		t.Fatalf("batch shapes acts=%d logits=%d tokens=%d", len(acts), len(logits), len(tokens))
	}
	for i, row := range hiddenRows {
		rowCopy := append([]float32(nil), row...)
		act, logit, tok, err := m.FinishCPUDecodeStep(rowCopy)
		if err != nil {
			t.Fatal(err)
		}
		if tokens[i] != tok || !sameFloat32s(acts[i], act) || !sameFloat32s(logits[i], logit) {
			t.Fatalf("row %d batch token=%d act=%v logits=%v; row token=%d act=%v logits=%v", i, tokens[i], acts[i], logits[i], tok, act, logit)
		}
	}
	if hiddenRows[0][0] != 3 {
		t.Fatalf("FinishCPUDecodeBatch mutated input rows: %v", hiddenRows)
	}
}

func TestFinishCPUDecodeBatchValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 2, HiddenSize: 2}, Norm: tensor.Ones([]int{2}), LMHead: tensor.Ones([]int{2, 2})}
	if _, _, _, err := (*LlamaModel)(nil).FinishCPUDecodeBatch([][]float32{{1, 2}}); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, _, _, err := m.FinishCPUDecodeBatch(nil); err == nil {
		t.Fatal("accepted empty batch")
	}
	if _, _, _, err := m.FinishCPUDecodeBatch([][]float32{{1}}); err == nil {
		t.Fatal("accepted wrong hidden width")
	}
	bad := &LlamaModel{Config: LlamaConfig{VocabSize: 2, HiddenSize: 2}, Norm: tensor.Ones([]int{2})}
	if _, _, _, err := bad.FinishCPUDecodeBatch([][]float32{{1, 2}}); err == nil {
		t.Fatal("accepted missing LM head")
	}
}
