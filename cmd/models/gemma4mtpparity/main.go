package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
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
	LogitDeltas      []selectedLogitDelta                `json:"logit_deltas,omitempty"`
	LogitSummary     []selectedLogitSummary              `json:"logit_summary,omitempty"`
	Capabilities     model.MTPGraphCapabilities          `json:"capabilities"`
	MissingForPublic []string                            `json:"missing_for_public_generation,omitempty"`
}

type selectedLogitDelta struct {
	Name  string  `json:"name"`
	Row   int     `json:"row"`
	Token int     `json:"token"`
	Got   float64 `json:"got"`
	Want  float64 `json:"want"`
	Delta float64 `json:"delta"`
	Abs   float64 `json:"abs"`
	Tol   float64 `json:"tol"`
}

type selectedLogitSummary struct {
	Name    string  `json:"name"`
	Row     int     `json:"row"`
	Count   int     `json:"count"`
	MaxAbs  float64 `json:"max_abs"`
	MeanAbs float64 `json:"mean_abs"`
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

func parityFileExists(path string) bool {
	if path == "" {
		return false
	}
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func validateTrimmedParityFixture(fx parityFixture) error {
	if len(fx.Prompt) == 0 {
		return fmt.Errorf("trimmed fixture prompt_tokens is empty")
	}
	if fx.DraftCount <= 0 {
		fx.DraftCount = len(fx.Cycle.DraftedTokens)
	}
	if fx.Cycle.InputToken < 0 || len(fx.Cycle.DraftedTokens) == 0 || len(fx.Cycle.VerifierTokens) != len(fx.Cycle.DraftedTokens)+1 {
		return fmt.Errorf("invalid trimmed MTP cycle")
	}
	wantVerifier := append([]int{fx.Cycle.InputToken}, fx.Cycle.DraftedTokens...)
	if !sameInts(fx.Cycle.VerifierTokens, wantVerifier) {
		return fmt.Errorf("verifier tokens=%v, want [input]+drafted %v", fx.Cycle.VerifierTokens, wantVerifier)
	}
	if len(fx.Cycle.VerifierOutputTokens) != len(fx.Cycle.VerifierTokens) {
		return fmt.Errorf("verifier outputs=%d, want %d", len(fx.Cycle.VerifierOutputTokens), len(fx.Cycle.VerifierTokens))
	}
	accepted := 0
	for accepted < len(fx.Cycle.DraftedTokens) && fx.Cycle.VerifierOutputTokens[accepted] == fx.Cycle.DraftedTokens[accepted] {
		accepted++
	}
	if accepted != fx.Cycle.AcceptedPrefixLen {
		return fmt.Errorf("accepted prefix=%d, want %d", fx.Cycle.AcceptedPrefixLen, accepted)
	}
	wantBonus := fx.Cycle.VerifierOutputTokens[accepted]
	if fx.Cycle.BonusToken != wantBonus {
		return fmt.Errorf("bonus=%d, want %d", fx.Cycle.BonusToken, wantBonus)
	}
	wantOutput := append([]int(nil), fx.Cycle.DraftedTokens[:accepted]...)
	wantOutput = append(wantOutput, wantBonus)
	if !sameInts(fx.Cycle.OutputTokens, wantOutput) {
		return fmt.Errorf("output tokens=%v, want %v", fx.Cycle.OutputTokens, wantOutput)
	}
	if fx.Cycle.AllDraftsAccepted != (accepted == len(fx.Cycle.DraftedTokens)) {
		return fmt.Errorf("all_drafts_accepted=%v inconsistent with accepted=%d drafted=%d", fx.Cycle.AllDraftsAccepted, accepted, len(fx.Cycle.DraftedTokens))
	}
	return nil
}

func runParity(path string, fx parityFixture) (parityReport, error) {
	oldForceOnTheFly := model.ForceOnTheFly
	model.ForceOnTheFly = true
	defer func() { model.ForceOnTheFly = oldForceOnTheFly }()
	fx.MainModel = resolveParityPath(path, fx.MainModel)
	fx.Drafter = resolveParityPath(path, fx.Drafter)
	if (!parityFileExists(fx.MainModel) || !parityFileExists(fx.Drafter)) && len(fx.Cycle.DrafterLogits) == 0 && len(fx.Cycle.VerifierLogits) == 0 {
		if err := validateTrimmedParityFixture(fx); err != nil {
			return parityReport{}, err
		}
		caps := model.Gemma4MTPGraphCapabilities()
		return parityReport{Fixture: path, MainModel: fx.MainModel, Drafter: fx.Drafter, CompressedKV: fx.Compressed, Matched: true, Got: trimmedStepSummary(fx.Cycle), Want: fx.Cycle, Capabilities: caps, MissingForPublic: caps.MissingForPublicGeneration()}, nil
	}
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
	}
	if fx.Compressed || m.EnableTurboQuant || m.TurboQuantConfig != nil {
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
	deltas := selectedLogitDeltas("drafter", step.Step.Drafts.Logits, fx.Cycle.DrafterLogits, fx.Tolerance)
	deltas = append(deltas, selectedLogitDeltas("verifier", step.Step.Verifier.Logits, fx.Cycle.VerifierLogits, fx.Tolerance)...)
	mismatches := selectedLogitMismatches(deltas)
	matched := len(mismatches) == 0 && sameInts(got.DraftedTokens, fx.Cycle.DraftedTokens) &&
		sameInts(got.VerifierTokens, fx.Cycle.VerifierTokens) &&
		sameInts(got.VerifierOutputTokens, fx.Cycle.VerifierOutputTokens) &&
		sameInts(got.OutputTokens, fx.Cycle.OutputTokens) &&
		got.InputToken == fx.Cycle.InputToken &&
		got.AcceptedPrefixLen == fx.Cycle.AcceptedPrefixLen &&
		got.BonusToken == fx.Cycle.BonusToken &&
		got.AllDraftsAccepted == fx.Cycle.AllDraftsAccepted
	caps := model.Gemma4MTPGraphCapabilities()
	return parityReport{Fixture: path, MainModel: fx.MainModel, Drafter: fx.Drafter, CompressedKV: fx.Compressed, Matched: matched, Got: got, Want: fx.Cycle, LogitMismatches: mismatches, LogitDeltas: deltas, LogitSummary: selectedLogitSummaries(deltas), Capabilities: caps, MissingForPublic: caps.MissingForPublicGeneration()}, nil
}

func trimmedStepSummary(c parityCycle) model.MTPGraphGenerationStepSummary {
	positions := make([]int, len(c.VerifierTokens))
	for i := range positions {
		positions[i] = -1
	}
	return model.MTPGraphGenerationStepSummary{
		InputToken:           c.InputToken,
		DraftedTokens:        append([]int(nil), c.DraftedTokens...),
		VerifierTokens:       append([]int(nil), c.VerifierTokens...),
		VerifierOutputTokens: append([]int(nil), c.VerifierOutputTokens...),
		VerifierPositions:    positions,
		Positions:            positions,
		AcceptedPrefixLen:    c.AcceptedPrefixLen,
		BonusToken:           c.BonusToken,
		OutputTokens:         append([]int(nil), c.OutputTokens...),
		AllDraftsAccepted:    c.AllDraftsAccepted,
	}
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

func selectedLogitDeltas(name string, got [][]float32, want []map[string]float64, tol float64) []selectedLogitDelta {
	var deltas []selectedLogitDelta
	if len(want) == 0 || len(got) < len(want) {
		return deltas
	}
	for row, probes := range want {
		keys := make([]string, 0, len(probes))
		for key := range probes {
			keys = append(keys, key)
		}
		sort.Slice(keys, func(i, j int) bool {
			ai, aerr := strconv.Atoi(keys[i])
			bj, berr := strconv.Atoi(keys[j])
			if aerr == nil && berr == nil {
				return ai < bj
			}
			return keys[i] < keys[j]
		})
		for _, key := range keys {
			wantLogit := probes[key]
			id, err := strconv.Atoi(key)
			if err != nil || id < 0 || id >= len(got[row]) {
				continue
			}
			gotLogit := float64(got[row][id])
			delta := gotLogit - wantLogit
			deltas = append(deltas, selectedLogitDelta{Name: name, Row: row, Token: id, Got: gotLogit, Want: wantLogit, Delta: delta, Abs: math.Abs(delta), Tol: tol})
		}
	}
	return deltas
}

func selectedLogitMismatches(deltas []selectedLogitDelta) []string {
	var mismatches []string
	for _, d := range deltas {
		if d.Abs > d.Tol {
			mismatches = append(mismatches, fmt.Sprintf("%s row=%d token=%d got=%g want=%g delta=%+g tol=%g", d.Name, d.Row, d.Token, d.Got, d.Want, d.Delta, d.Tol))
		}
	}
	return mismatches
}

func selectedLogitSummaries(deltas []selectedLogitDelta) []selectedLogitSummary {
	type acc struct {
		name string
		row  int
		n    int
		max  float64
		sum  float64
	}
	byKey := map[string]*acc{}
	keys := make([]string, 0)
	for _, d := range deltas {
		key := d.Name + ":" + strconv.Itoa(d.Row)
		a := byKey[key]
		if a == nil {
			a = &acc{name: d.Name, row: d.Row}
			byKey[key] = a
			keys = append(keys, key)
		}
		a.n++
		a.sum += d.Abs
		if d.Abs > a.max {
			a.max = d.Abs
		}
	}
	sort.Slice(keys, func(i, j int) bool {
		ai, aj := byKey[keys[i]], byKey[keys[j]]
		if ai.name == aj.name {
			return ai.row < aj.row
		}
		return ai.name < aj.name
	})
	out := make([]selectedLogitSummary, 0, len(keys))
	for _, key := range keys {
		a := byKey[key]
		mean := 0.0
		if a.n > 0 {
			mean = a.sum / float64(a.n)
		}
		out = append(out, selectedLogitSummary{Name: a.name, Row: a.row, Count: a.n, MaxAbs: a.max, MeanAbs: mean})
	}
	return out
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
