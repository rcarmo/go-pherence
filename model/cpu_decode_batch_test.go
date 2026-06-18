package model

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestFinishCPUDecodeBatchMatchesRows(t *testing.T) {
	m := &LlamaModel{
		Config:         LlamaConfig{VocabSize: 3, HiddenSize: 2, RMSNormEps: 0, FinalLogitSoftcapping: 2},
		SuppressTokens: []int{1},
		Norm:           tensor.Ones([]int{2}),
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

func TestFinishCPUDecodeBatchMatchesRowsWithGGUFLMHead(t *testing.T) {
	lm := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	raw := make([]byte, len(lm)*4)
	for i, v := range lm {
		binary.LittleEndian.PutUint32(raw[i*4:(i+1)*4], math.Float32bits(v))
	}
	m := &LlamaModel{
		Config:         LlamaConfig{VocabSize: 3, HiddenSize: 2, RMSNormEps: 0, FinalLogitSoftcapping: 2},
		SuppressTokens: []int{1},
		Norm:           tensor.Ones([]int{2}),
		LMHeadGGUF:     &gguf.QuantMatrix{Name: "synthetic.lm_head", QType: gguf.QuantF32, Raw: raw, InDim: 2, OutDim: 3},
	}
	hiddenRows := [][]float32{{3, 4}, {5, 12}}
	acts, logits, tokens, err := m.FinishCPUDecodeBatch(hiddenRows)
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range hiddenRows {
		rowCopy := append([]float32(nil), row...)
		act, logit, tok, err := m.FinishCPUDecodeStep(rowCopy)
		if err != nil {
			t.Fatal(err)
		}
		if tokens[i] != tok || !sameFloat32s(acts[i], act) || !sameFloat32s(logits[i], logit) {
			t.Fatalf("GGUF row %d batch token=%d act=%v logits=%v; row token=%d act=%v logits=%v", i, tokens[i], acts[i], logits[i], tok, act, logit)
		}
	}
}

func TestFinishCPUDecodeBatchMatchesRowsWithRealGemma4GGUF(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	h := m.Config.HiddenSize
	hiddenRows := make([][]float32, 3)
	for r := range hiddenRows {
		row := make([]float32, h)
		for i := range row {
			row[i] = float32(math.Sin(float64(i+r*17)*0.031)) * 0.75
		}
		hiddenRows[r] = row
	}
	acts, logits, tokens, err := m.FinishCPUDecodeBatch(hiddenRows)
	if err != nil {
		t.Fatal(err)
	}
	if len(acts) != len(hiddenRows) || len(logits) != len(hiddenRows) || len(tokens) != len(hiddenRows) {
		t.Fatalf("batch shapes acts/logits/tokens=%d/%d/%d want %d", len(acts), len(logits), len(tokens), len(hiddenRows))
	}
	probeIDs := []int{564, 236751, 236757, 236789}
	for i, row := range hiddenRows {
		rowCopy := append([]float32(nil), row...)
		act, logit, tok, err := m.FinishCPUDecodeStep(rowCopy)
		if err != nil {
			t.Fatal(err)
		}
		if tokens[i] != tok || !sameFloat32s(acts[i], act) {
			t.Fatalf("real GGUF row %d batch token/act mismatch token=%d/%d", i, tokens[i], tok)
		}
		for _, id := range probeIDs {
			if id < 0 || id >= len(logits[i]) || id >= len(logit) {
				t.Fatalf("probe id %d outside logits", id)
			}
			if logits[i][id] != logit[id] {
				t.Fatalf("real GGUF row %d token %d batch logit=%g row logit=%g", i, id, logits[i][id], logit[id])
			}
		}
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
