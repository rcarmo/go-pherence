package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/model"
)

type parityFixture struct {
	MainModel  string      `json:"main_model"`
	Drafter    string      `json:"drafter"`
	Prompt     []int       `json:"prompt_tokens"`
	MaxTokens  int         `json:"max_tokens"`
	DraftCount int         `json:"draft_count"`
	Compressed bool        `json:"compressed_kv"`
	Tolerance  float64     `json:"logit_tolerance"`
	Cycle      parityCycle `json:"cycle"`
}

type parityCycle struct {
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

type parityReport struct {
	Fixture          string                              `json:"fixture"`
	MainModel        string                              `json:"main_model"`
	Drafter          string                              `json:"drafter"`
	CompressedKV     bool                                `json:"compressed_kv"`
	Matched          bool                                `json:"matched"`
	Got              model.MTPGraphGenerationStepSummary `json:"got"`
	Want             parityCycle                         `json:"want"`
	LogitMismatches  []string                            `json:"logit_mismatches,omitempty"`
	Capabilities     model.MTPGraphCapabilities          `json:"capabilities"`
	MissingForPublic []string                            `json:"missing_for_public_generation,omitempty"`
}

func main() {
	fixturePath := flag.String("fixture", "", "llama.cpp/LiteRT Gemma4 MTP parity JSON fixture")
	mainModel := flag.String("model", "", "override main model path from fixture")
	drafterPath := flag.String("drafter", "", "override MTP drafter path from fixture")
	pretty := flag.Bool("pretty", true, "pretty-print JSON report")
	flag.Parse()
	if *fixturePath == "" {
		fmt.Fprintln(os.Stderr, "usage: gemma4mtpparity -fixture <fixture.json> [-model main] [-drafter assistant]")
		os.Exit(2)
	}
	fx, err := loadParityFixture(*fixturePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fixture: %v\n", err)
		os.Exit(1)
	}
	if *mainModel != "" {
		fx.MainModel = *mainModel
	}
	if *drafterPath != "" {
		fx.Drafter = *drafterPath
	}
	report, err := runParity(*fixturePath, fx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parity: %v\n", err)
		os.Exit(1)
	}
	var out []byte
	if *pretty {
		out, err = json.MarshalIndent(report, "", "  ")
	} else {
		out, err = json.Marshal(report)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "report: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(out))
	if !report.Matched {
		os.Exit(1)
	}
}

func loadParityFixture(path string) (parityFixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return parityFixture{}, err
	}
	var fx parityFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		return parityFixture{}, err
	}
	if fx.MainModel == "" {
		fx.MainModel = os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	}
	if fx.Drafter == "" {
		fx.Drafter = os.Getenv("GO_PHERENCE_GEMMA4_MTP_DRAFTER")
	}
	if fx.MainModel == "" || fx.Drafter == "" {
		return parityFixture{}, fmt.Errorf("fixture or env must supply main_model/drafter paths")
	}
	if len(fx.Prompt) == 0 {
		return parityFixture{}, fmt.Errorf("prompt_tokens is empty")
	}
	if fx.DraftCount <= 0 {
		fx.DraftCount = len(fx.Cycle.DraftedTokens)
	}
	if fx.MaxTokens <= 1 {
		fx.MaxTokens = fx.DraftCount + 1
	}
	if fx.DraftCount <= 0 || fx.MaxTokens <= fx.DraftCount {
		return parityFixture{}, fmt.Errorf("invalid draft/max tokens draft=%d max=%d", fx.DraftCount, fx.MaxTokens)
	}
	if fx.Tolerance == 0 {
		fx.Tolerance = 1e-3
	}
	return fx, nil
}

func resolveParityPath(fixturePath, path string) string {
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
	if root := findParityRepoRoot(); root != "" {
		repoRelative := filepath.Join(root, path)
		if _, err := os.Stat(repoRelative); err == nil {
			return repoRelative
		}
	}
	return path
}

func findParityRepoRoot() string {
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

func runParity(path string, fx parityFixture) (parityReport, error) {
	oldForceOnTheFly := model.ForceOnTheFly
	model.ForceOnTheFly = true
	defer func() { model.ForceOnTheFly = oldForceOnTheFly }()
	fx.MainModel = resolveParityPath(path, fx.MainModel)
	fx.Drafter = resolveParityPath(path, fx.Drafter)
	m, err := loadMainModelForParity(fx.MainModel)
	if err != nil {
		return parityReport{}, fmt.Errorf("load main model: %w", err)
	}
	d, err := loadDrafterForParity(fx.Drafter)
	if err != nil {
		return parityReport{}, fmt.Errorf("load drafter: %w", err)
	}
	promptForContext := append([]int(nil), fx.Prompt...)
	inputToken := fx.Cycle.InputToken
	if inputToken < 0 {
		inputToken = promptForContext[len(promptForContext)-1]
	}
	// llama.cpp speculative MTP uses target prompt K/V and the target hidden row
	// that produced id_last, then feeds id_last as the assistant token at
	// dp.n_past. Some historical fixtures stored id_last as the last
	// prompt_tokens entry; keep that token out of the prompt-context KV.
	if len(promptForContext) > 1 && promptForContext[len(promptForContext)-1] == inputToken {
		promptForContext = promptForContext[:len(promptForContext)-1]
	}
	ctx, err := m.BuildMTPPromptContext(promptForContext)
	if err != nil {
		return parityReport{}, fmt.Errorf("prompt context: %w", err)
	}
	ext, err := model.NewMTPDrafterExternalKVFromPromptContext(m, d, ctx)
	if err != nil {
		return parityReport{}, fmt.Errorf("external KV: %w", err)
	}
	decode, err := model.NewCPUDecodeStateFromMTPPromptContext(m, ctx, fx.MaxTokens)
	if err != nil {
		return parityReport{}, fmt.Errorf("decode state: %w", err)
	}
	if fx.Compressed {
		m.EnableTurboQuant = true
		if err := m.SeedCompressedKVFromPromptContext(decode, ctx); err != nil {
			return parityReport{}, fmt.Errorf("compressed prompt seed: %w", err)
		}
	}
	state, err := model.NewMTPDrafterState(inputToken, ctx.Activation, d.BackboneHiddenSize)
	if err != nil {
		return parityReport{}, fmt.Errorf("drafter state: %w", err)
	}
	step, err := decode.RunMTPGraphDecodeStep(d, state, ext, model.MTPGraphDecodeStepOptions{RemainingTokens: fx.MaxTokens, DraftCount: fx.DraftCount}, model.MTPSpeculationStats{})
	if err != nil {
		return parityReport{}, fmt.Errorf("MTP graph decode step: %w", err)
	}
	got := newStepSummary(step)
	mismatches := compareSelectedLogits("drafter", step.Step.Drafts.Logits, fx.Cycle.DrafterLogits, fx.Tolerance)
	mismatches = append(mismatches, compareSelectedLogits("verifier", step.Step.Verifier.Logits, fx.Cycle.VerifierLogits, fx.Tolerance)...)
	matched := len(mismatches) == 0 && sameInts(got.DraftedTokens, fx.Cycle.DraftedTokens) &&
		sameInts(got.VerifierTokens, fx.Cycle.VerifierTokens) &&
		sameInts(got.VerifierOutputTokens, fx.Cycle.VerifierOutputTokens) &&
		sameInts(got.OutputTokens, fx.Cycle.OutputTokens) &&
		got.InputToken == fx.Cycle.InputToken &&
		got.AcceptedPrefixLen == fx.Cycle.AcceptedPrefixLen &&
		got.BonusToken == fx.Cycle.BonusToken &&
		got.AllDraftsAccepted == fx.Cycle.AllDraftsAccepted
	caps := model.Gemma4MTPGraphCapabilities()
	return parityReport{Fixture: path, MainModel: fx.MainModel, Drafter: fx.Drafter, CompressedKV: fx.Compressed, Matched: matched, Got: got, Want: fx.Cycle, LogitMismatches: mismatches, Capabilities: caps, MissingForPublic: caps.MissingForPublicGeneration()}, nil
}

func loadMainModelForParity(path string) (*model.LlamaModel, error) {
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return model.LoadGemma4GGUFAsLlama(path)
	}
	return model.LoadLlama(path)
}

func loadDrafterForParity(path string) (*model.Gemma4MTPDrafter, error) {
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return model.LoadGemma4MTPDrafterGGUF(path)
	}
	return model.LoadGemma4MTPDrafter(path)
}

func newStepSummary(step model.MTPGraphDecodeStepResult) model.MTPGraphGenerationStepSummary {
	verifierOutputs := make([]int, 0, len(step.Step.Verifier.Logits))
	for _, logits := range step.Step.Verifier.Logits {
		id, _, err := model.ArgmaxLogits(logits)
		if err != nil {
			id = -1
		}
		verifierOutputs = append(verifierOutputs, id)
	}
	return model.MTPGraphGenerationStepSummary{
		InputToken:           step.Step.Graph.InputToken,
		DraftedTokens:        append([]int(nil), step.Step.Drafts.Tokens...),
		VerifierTokens:       append([]int(nil), step.Step.Plan.VerifierTokens...),
		VerifierOutputTokens: verifierOutputs,
		VerifierPositions:    append([]int(nil), step.Step.Plan.Positions...),
		Positions:            append([]int(nil), step.Commit.Positions...),
		AcceptedPrefixLen:    step.Step.Verifier.Acceptance.AcceptedPrefixLen,
		BonusToken:           step.Step.Verifier.Acceptance.BonusToken,
		OutputTokens:         append([]int(nil), step.Commit.OutputTokens...),
		AllDraftsAccepted:    step.Step.Verifier.Acceptance.AllDraftsAccepted,
	}
}

func compareSelectedLogits(name string, got [][]float32, want []map[string]float64, tol float64) []string {
	var mismatches []string
	if len(want) == 0 {
		return mismatches
	}
	if len(got) < len(want) {
		return append(mismatches, fmt.Sprintf("%s logits rows=%d want_at_least=%d", name, len(got), len(want)))
	}
	for row, probes := range want {
		for key, wantLogit := range probes {
			id, err := strconv.Atoi(key)
			if err != nil {
				mismatches = append(mismatches, fmt.Sprintf("%s row=%d invalid_token_key=%q", name, row, key))
				continue
			}
			if id < 0 || id >= len(got[row]) {
				mismatches = append(mismatches, fmt.Sprintf("%s row=%d token=%d outside_width=%d", name, row, id, len(got[row])))
				continue
			}
			gotLogit := float64(got[row][id])
			if math.Abs(gotLogit-wantLogit) > tol {
				mismatches = append(mismatches, fmt.Sprintf("%s row=%d token=%d got=%g want=%g tol=%g", name, row, id, gotLogit, wantLogit, tol))
			}
		}
	}
	return mismatches
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
