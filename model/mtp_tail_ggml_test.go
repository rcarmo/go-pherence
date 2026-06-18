//go:build ggml && cgo && linux

package model

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/backends/ggmlcompute"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func cloneMTPFloatCacheForTest(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}

func TestGemma4MTPStrictFixtureTailNormGGMLOracle(t *testing.T) {
	fixturePath := os.Getenv("GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE")
	if fixturePath == "" {
		t.Skip("GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE not set")
	}
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx mtpLlamaCPPParityFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if envMain := os.Getenv("GO_PHERENCE_GEMMA4_MAIN"); envMain != "" {
		fx.MainModel = envMain
	}
	fx.MainModel = resolveMTPParityPath(fixturePath, fx.MainModel)
	if !mtpParityFileExists(fx.MainModel) {
		t.Skipf("main model not available at %s", fx.MainModel)
	}
	oldForceOnTheFly := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForceOnTheFly }()
	m, err := LoadGemma4GGUFAsLlama(fx.MainModel)
	if err != nil {
		t.Fatal(err)
	}
	promptForContext := append([]int(nil), fx.Prompt...)
	inputToken := fx.Cycle.InputToken
	if inputToken < 0 {
		inputToken = promptForContext[len(promptForContext)-1]
	}
	if len(promptForContext) > 1 && promptForContext[len(promptForContext)-1] == inputToken {
		promptForContext = promptForContext[:len(promptForContext)-1]
	}
	ctx, err := m.BuildMTPPromptContext(promptForContext)
	if err != nil {
		t.Fatal(err)
	}
	plan := MTPVerifierPlan{
		InputToken:     inputToken,
		DraftedTokens:  append([]int(nil), fx.Cycle.DraftedTokens...),
		VerifierTokens: append([]int(nil), fx.Cycle.VerifierTokens...),
		StartPos:       ctx.SeqLen,
	}
	plan.Positions, err = mtpVerifierPositions(plan.StartPos, len(plan.VerifierTokens))
	if err != nil {
		t.Fatal(err)
	}
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	res, err := m.RunMTPVerifierBatchForward(batch, cloneMTPFloatCacheForTest(ctx.KVCacheK), cloneMTPFloatCacheForTest(ctx.KVCacheV))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ActivationRows) != len(plan.VerifierTokens) {
		t.Fatalf("activation rows=%d want %d", len(res.ActivationRows), len(plan.VerifierTokens))
	}
	g, err := gguf.Open(fx.MainModel)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	lmTensor, ok := g.TensorByName("output.weight")
	if !ok {
		lmTensor, ok = g.TensorByName("token_embd.weight")
	}
	if !ok {
		t.Fatal("output.weight/token_embd.weight not found")
	}
	lmRaw, err := g.Raw(lmTensor)
	if err != nil {
		t.Fatal(err)
	}
	probes := []int{564, 236751, 236757, 236789}
	for row, hidden := range res.ActivationRows {
		goNorm := append([]float32(nil), hidden...)
		rmsNormInPlace(goNorm, m.Norm.Data(), float32(m.Config.RMSNormEps))
		ggNorm := make([]float32, len(hidden))
		if err := ggmlcompute.RMSNormMulF32(ggNorm, hidden, m.Norm.Data(), float32(m.Config.RMSNormEps)); err != nil {
			t.Fatal(err)
		}
		maxDiff, meanDiff := maxMeanAbsDiff(ggNorm, goNorm)
		t.Logf("strict fixture verifier row=%d final norm ggml-vs-go max=%g mean=%g", row, maxDiff, meanDiff)
		if maxDiff > 1e-5 {
			t.Fatalf("strict fixture verifier row=%d final norm max=%g mean=%g", row, maxDiff, meanDiff)
		}
		goLogits := make([]float32, m.Config.VocabSize)
		if err := m.LMHeadLogitsInto(goLogits, goNorm); err != nil {
			t.Fatal(err)
		}
		ggLogits := make([]float32, m.Config.VocabSize)
		if err := ggmlcompute.MulMatQuantF32(int(lmTensor.QType), ggLogits, lmRaw, ggNorm, m.Config.HiddenSize, m.Config.VocabSize); err != nil {
			t.Fatal(err)
		}
		applyLlamaFinalLogitSoftcap(ggLogits, m.Config.FinalLogitSoftcapping)
		applyLlamaSuppressTokens(ggLogits, m.SuppressTokens)
		for _, id := range probes {
			if id >= len(goLogits) || id >= len(ggLogits) {
				t.Fatalf("probe id %d outside logits", id)
			}
			diff := goLogits[id] - ggLogits[id]
			if diff < 0 {
				diff = -diff
			}
			t.Logf("strict fixture verifier row=%d token=%d LM-head ggml=%g go=%g diff=%g", row, id, ggLogits[id], goLogits[id], diff)
			if diff > 1e-5 {
				t.Fatalf("strict fixture verifier row=%d token=%d LM-head diff=%g", row, id, diff)
			}
		}
	}
}
