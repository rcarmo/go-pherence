package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type mtpLlamaCPPParityFixture struct {
	MainModel  string                  `json:"main_model"`
	Drafter    string                  `json:"drafter"`
	Prompt     []int                   `json:"prompt_tokens"`
	MaxTokens  int                     `json:"max_tokens"`
	DraftCount int                     `json:"draft_count"`
	Compressed bool                    `json:"compressed_kv"`
	Tolerance  float64                 `json:"logit_tolerance"`
	Cycle      mtpLlamaCPPCycleFixture `json:"cycle"`
}

type mtpLlamaCPPCycleFixture struct {
	InputToken           int                  `json:"input_token"`
	DraftedTokens        []int                `json:"drafted_tokens"`
	VerifierTokens       []int                `json:"verifier_tokens"`
	VerifierOutputTokens []int                `json:"verifier_output_tokens"`
	AcceptedPrefixLen    int                  `json:"accepted_prefix_len"`
	BonusToken           int                  `json:"bonus_token"`
	OutputTokens         []int                `json:"output_tokens"`
	AllDraftsAccepted    bool                 `json:"all_drafts_accepted"`
	DrafterLogits        []map[string]float64 `json:"drafter_logits"`
	VerifierLogits       []map[string]float64 `json:"verifier_logits"`
}

func resolveMTPParityPath(fixturePath, path string) string {
	if path == "" || filepath.IsAbs(path) {
		return path
	}
	if _, err := os.Stat(path); err == nil {
		return path
	}
	fixtureRelative := filepath.Join(filepath.Dir(fixturePath), path)
	if _, err := os.Stat(fixtureRelative); err == nil {
		return fixtureRelative
	}
	if root := findMTPParityRepoRoot(); root != "" {
		repoRelative := filepath.Join(root, path)
		if _, err := os.Stat(repoRelative); err == nil {
			return repoRelative
		}
	}
	return path
}

func findMTPParityRepoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

func mtpParityFileExists(path string) bool {
	if path == "" {
		return false
	}
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func assertMTPTrimmedParityFixture(t *testing.T, fx mtpLlamaCPPParityFixture) {
	t.Helper()
	if len(fx.Prompt) == 0 {
		t.Fatalf("trimmed fixture prompt_tokens is empty")
	}
	if fx.DraftCount <= 0 {
		fx.DraftCount = len(fx.Cycle.DraftedTokens)
	}
	if fx.MaxTokens <= 1 {
		fx.MaxTokens = len(fx.Cycle.OutputTokens)
	}
	if fx.Cycle.InputToken < 0 || len(fx.Cycle.DraftedTokens) == 0 || len(fx.Cycle.VerifierTokens) != len(fx.Cycle.DraftedTokens)+1 {
		t.Fatalf("invalid trimmed MTP cycle: %+v", fx.Cycle)
	}
	wantVerifierTokens := append([]int{fx.Cycle.InputToken}, fx.Cycle.DraftedTokens...)
	if !sameInts(fx.Cycle.VerifierTokens, wantVerifierTokens) {
		t.Fatalf("trimmed verifier tokens=%v, want [input]+drafted %v", fx.Cycle.VerifierTokens, wantVerifierTokens)
	}
	if len(fx.Cycle.VerifierOutputTokens) != len(fx.Cycle.VerifierTokens) {
		t.Fatalf("trimmed verifier outputs=%d, want %d", len(fx.Cycle.VerifierOutputTokens), len(fx.Cycle.VerifierTokens))
	}
	accepted := 0
	for accepted < len(fx.Cycle.DraftedTokens) && fx.Cycle.VerifierOutputTokens[accepted] == fx.Cycle.DraftedTokens[accepted] {
		accepted++
	}
	if accepted != fx.Cycle.AcceptedPrefixLen {
		t.Fatalf("trimmed accepted prefix=%d, want %d", fx.Cycle.AcceptedPrefixLen, accepted)
	}
	wantBonus := fx.Cycle.VerifierOutputTokens[accepted]
	if fx.Cycle.BonusToken != wantBonus {
		t.Fatalf("trimmed bonus=%d, want verifier output %d", fx.Cycle.BonusToken, wantBonus)
	}
	wantOutput := append([]int(nil), fx.Cycle.DraftedTokens[:accepted]...)
	wantOutput = append(wantOutput, wantBonus)
	if !sameInts(fx.Cycle.OutputTokens, wantOutput) {
		t.Fatalf("trimmed output tokens=%v, want %v", fx.Cycle.OutputTokens, wantOutput)
	}
	if fx.Cycle.AllDraftsAccepted != (accepted == len(fx.Cycle.DraftedTokens)) {
		t.Fatalf("trimmed all_drafts_accepted=%v inconsistent with accepted=%d drafted=%d", fx.Cycle.AllDraftsAccepted, accepted, len(fx.Cycle.DraftedTokens))
	}
}

func TestGemma4MTPLlamaCPPParityFixture(t *testing.T) {
	fixturePath := os.Getenv("GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE")
	usingDefaultFixture := fixturePath == ""
	if usingDefaultFixture {
		fixturePath = filepath.Join("testdata", "gemma4-mtp-llamacpp-fixture.json")
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
	if envDrafter := os.Getenv("GO_PHERENCE_GEMMA4_MTP_DRAFTER"); envDrafter != "" {
		fx.Drafter = envDrafter
	}
	if fx.MainModel == "" || fx.Drafter == "" {
		t.Fatalf("fixture or env must supply main_model/drafter paths")
	}
	fx.MainModel = resolveMTPParityPath(fixturePath, fx.MainModel)
	fx.Drafter = resolveMTPParityPath(fixturePath, fx.Drafter)
	if usingDefaultFixture && (!mtpParityFileExists(fx.MainModel) || !mtpParityFileExists(fx.Drafter)) {
		assertMTPTrimmedParityFixture(t, fx)
		return
	}
	if len(fx.Prompt) == 0 {
		t.Fatalf("fixture prompt_tokens is empty")
	}
	if fx.MaxTokens <= 1 {
		fx.MaxTokens = len(fx.Cycle.DraftedTokens) + 1
	}
	if fx.DraftCount <= 0 {
		fx.DraftCount = len(fx.Cycle.DraftedTokens)
	}
	if fx.DraftCount <= 0 {
		t.Fatalf("fixture draft_count=%d and drafted_tokens=%d", fx.DraftCount, len(fx.Cycle.DraftedTokens))
	}
	if fx.Tolerance == 0 {
		fx.Tolerance = 1e-3
	}

	ForceOnTheFly = true
	var m *LlamaModel
	if strings.HasSuffix(strings.ToLower(fx.MainModel), ".gguf") {
		m, err = LoadGemma4GGUFAsLlama(fx.MainModel)
	} else {
		m, err = LoadLlama(fx.MainModel)
	}
	if err != nil {
		t.Fatalf("load main model: %v", err)
	}
	var d *Gemma4MTPDrafter
	if strings.HasSuffix(strings.ToLower(fx.Drafter), ".gguf") {
		d, err = LoadGemma4MTPDrafterGGUF(fx.Drafter)
	} else {
		d, err = LoadGemma4MTPDrafter(fx.Drafter)
	}
	if err != nil {
		t.Fatalf("load MTP drafter: %v", err)
	}
	if m.Config.HiddenSize != d.BackboneHiddenSize || m.Config.VocabSize != d.Config.VocabSize {
		t.Fatalf("model/drafter mismatch h/vocab=%d/%d backbone/vocab=%d/%d", m.Config.HiddenSize, m.Config.VocabSize, d.BackboneHiddenSize, d.Config.VocabSize)
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
		t.Fatalf("prompt context: %v", err)
	}
	externalKV, err := NewMTPDrafterExternalKVFromPromptContext(m, d, ctx)
	if err != nil {
		t.Fatalf("external KV: %v", err)
	}
	decode, err := NewCPUDecodeStateFromMTPPromptContext(m, ctx, fx.MaxTokens)
	if err != nil {
		t.Fatalf("decode state: %v", err)
	}
	if fx.Compressed || m.EnableTurboQuant || m.TurboQuantConfig != nil {
		if err := m.SeedCompressedKVFromPromptContext(decode, ctx); err != nil {
			t.Fatalf("compressed prompt seed: %v", err)
		}
	}
	state, err := NewMTPDrafterState(inputToken, ctx.Activation, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("drafter state: %v", err)
	}
	step, err := decode.RunMTPGraphDecodeStep(d, state, externalKV, MTPGraphDecodeStepOptions{RemainingTokens: fx.MaxTokens, DraftCount: fx.DraftCount}, MTPSpeculationStats{})
	if err != nil {
		t.Fatalf("MTP graph decode step: %v", err)
	}
	summary := newMTPGraphGenerationStepSummary(step)
	assertMTPParityCycle(t, summary, fx.Cycle)
	assertMTPParityLogits(t, "drafter", step.Step.Drafts.Logits, fx.Cycle.DrafterLogits, fx.Tolerance)
	assertMTPParityLogits(t, "verifier", step.Step.Verifier.Logits, fx.Cycle.VerifierLogits, fx.Tolerance)
}

func assertMTPParityCycle(t *testing.T, got MTPGraphGenerationStepSummary, want mtpLlamaCPPCycleFixture) {
	t.Helper()
	if got.InputToken != want.InputToken || got.AcceptedPrefixLen != want.AcceptedPrefixLen || got.BonusToken != want.BonusToken || got.AllDraftsAccepted != want.AllDraftsAccepted ||
		!sameInts(got.DraftedTokens, want.DraftedTokens) || !sameInts(got.VerifierTokens, want.VerifierTokens) || !sameInts(got.VerifierOutputTokens, want.VerifierOutputTokens) || !sameInts(got.OutputTokens, want.OutputTokens) {
		t.Fatalf("MTP llama.cpp parity mismatch\n got=%+v\nwant=%+v", got, want)
	}
}

func assertMTPParityLogits(t *testing.T, name string, got [][]float32, want []map[string]float64, tol float64) {
	t.Helper()
	if len(want) == 0 {
		return
	}
	if len(got) < len(want) {
		t.Fatalf("%s logits rows=%d, want at least %d", name, len(got), len(want))
	}
	for row, probes := range want {
		for key, wantLogit := range probes {
			id, err := strconv.Atoi(key)
			if err != nil {
				t.Fatalf("%s logits row %d token key %q: %v", name, row, key, err)
			}
			if id < 0 || id >= len(got[row]) {
				t.Fatalf("%s logits row %d token=%d outside width=%d", name, row, id, len(got[row]))
			}
			gotLogit := float64(got[row][id])
			if math.Abs(gotLogit-wantLogit) > tol {
				t.Fatalf("%s logits row %d token %d=%g, want %g (tol %g)", name, row, id, gotLogit, wantLogit, tol)
			}
		}
	}
}

func TestGemma4MTPLlamaCPPDefaultFixtureTrimmedValidation(t *testing.T) {
	fixturePath := filepath.Join("testdata", "gemma4-mtp-llamacpp-fixture.json")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read default fixture: %v", err)
	}
	var fx mtpLlamaCPPParityFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("parse default fixture: %v", err)
	}
	assertMTPTrimmedParityFixture(t, fx)
	if len(fx.Cycle.VerifierLogits) != 0 || len(fx.Cycle.DrafterLogits) != 0 {
		t.Fatalf("default fixture must remain a trimmed token/acceptance gate; got drafter probes=%d verifier probes=%d", len(fx.Cycle.DrafterLogits), len(fx.Cycle.VerifierLogits))
	}
}

func TestGemma4MTPLlamaCPPParityFixtureSchema(t *testing.T) {
	fixture := mtpLlamaCPPParityFixture{Prompt: []int{1}, MaxTokens: 3, DraftCount: 2, Cycle: mtpLlamaCPPCycleFixture{InputToken: 1, DraftedTokens: []int{2, 3}, VerifierTokens: []int{1, 2, 3}, VerifierOutputTokens: []int{2, 4, 5}, AcceptedPrefixLen: 1, BonusToken: 4, OutputTokens: []int{2, 4}, DrafterLogits: []map[string]float64{{"2": 1.25}}, VerifierLogits: []map[string]float64{{"2": 2.5}}}}
	data, err := json.Marshal(fixture)
	if err != nil {
		t.Fatal(err)
	}
	var decoded mtpLlamaCPPParityFixture
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Prompt) != 1 || decoded.Cycle.BonusToken != 4 || decoded.Cycle.DrafterLogits[0]["2"] != 1.25 {
		t.Fatalf("decoded fixture=%+v", decoded)
	}
	if fmt.Sprint(decoded.Cycle.OutputTokens) != "[2 4]" {
		t.Fatalf("decoded output tokens=%v", decoded.Cycle.OutputTokens)
	}
}
