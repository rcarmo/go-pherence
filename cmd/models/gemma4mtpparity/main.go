package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/model"
)

type parityFixture struct {
	MainModel  string      `json:"main_model"`
	Drafter    string      `json:"drafter"`
	Prompt     []int       `json:"prompt_tokens"`
	MaxTokens  int         `json:"max_tokens"`
	DraftCount int         `json:"draft_count"`
	Compressed bool        `json:"compressed_kv"`
	Cycle      parityCycle `json:"cycle"`
}

type parityCycle struct {
	InputToken           int   `json:"input_token"`
	DraftedTokens        []int `json:"drafted_tokens"`
	VerifierTokens       []int `json:"verifier_tokens"`
	VerifierOutputTokens []int `json:"verifier_output_tokens"`
	AcceptedPrefixLen    int   `json:"accepted_prefix_len"`
	BonusToken           int   `json:"bonus_token"`
	OutputTokens         []int `json:"output_tokens"`
	AllDraftsAccepted    bool  `json:"all_drafts_accepted"`
}

type parityReport struct {
	Fixture          string                              `json:"fixture"`
	MainModel        string                              `json:"main_model"`
	Drafter          string                              `json:"drafter"`
	CompressedKV     bool                                `json:"compressed_kv"`
	Matched          bool                                `json:"matched"`
	Got              model.MTPGraphGenerationStepSummary `json:"got"`
	Want             parityCycle                         `json:"want"`
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
	return fx, nil
}

func runParity(path string, fx parityFixture) (parityReport, error) {
	model.ForceOnTheFly = true
	m, err := model.LoadLlama(fx.MainModel)
	if err != nil {
		return parityReport{}, fmt.Errorf("load main model: %w", err)
	}
	d, err := model.LoadGemma4MTPDrafter(fx.Drafter)
	if err != nil {
		return parityReport{}, fmt.Errorf("load drafter: %w", err)
	}
	ctx, err := m.BuildMTPPromptContext(fx.Prompt)
	if err != nil {
		return parityReport{}, fmt.Errorf("prompt context: %w", err)
	}
	ext, err := model.NewMTPDrafterExternalKVFromPromptContext(m, d, ctx)
	if err != nil {
		return parityReport{}, fmt.Errorf("external KV: %w", err)
	}
	res, err := m.GenerateMTPGraphFromPromptContext(d, ctx, ext, model.MTPGraphGenerationOptions{
		MaxTokens:       fx.MaxTokens,
		UseCompressedKV: fx.Compressed,
		Policy: model.MTPAdaptiveDraftPolicy{
			MinDrafts:     fx.DraftCount,
			InitialDrafts: fx.DraftCount,
			MaxDrafts:     fx.DraftCount,
		},
	})
	if err != nil {
		return parityReport{}, fmt.Errorf("generate MTP graph: %w", err)
	}
	if len(res.StepSummaries) == 0 {
		return parityReport{}, fmt.Errorf("no MTP graph cycle produced")
	}
	got := res.StepSummaries[0]
	matched := sameInts(got.DraftedTokens, fx.Cycle.DraftedTokens) &&
		sameInts(got.VerifierTokens, fx.Cycle.VerifierTokens) &&
		sameInts(got.VerifierOutputTokens, fx.Cycle.VerifierOutputTokens) &&
		sameInts(got.OutputTokens, fx.Cycle.OutputTokens) &&
		got.InputToken == fx.Cycle.InputToken &&
		got.AcceptedPrefixLen == fx.Cycle.AcceptedPrefixLen &&
		got.BonusToken == fx.Cycle.BonusToken &&
		got.AllDraftsAccepted == fx.Cycle.AllDraftsAccepted
	return parityReport{Fixture: path, MainModel: fx.MainModel, Drafter: fx.Drafter, CompressedKV: res.UsedCompressedKV, Matched: matched, Got: got, Want: fx.Cycle, Capabilities: res.Capabilities, MissingForPublic: res.MissingForPublicGeneration}, nil
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
