package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/rcarmo/go-pherence/backends/spacemit/board"
)

const gemmaFixtureLlamaCPPRevision = "4a6735f1c"

type gemmaLlamaCPPFixture struct {
	LlamaCPPRevision string                     `json:"llama_cpp_revision"`
	Tolerance        float64                    `json:"logit_tolerance"`
	Cases            []gemmaLlamaCPPFixtureCase `json:"cases"`
}

type gemmaLlamaCPPFixtureCase struct {
	Name         string                     `json:"name"`
	Architecture string                     `json:"architecture"`
	Model        string                     `json:"model"`
	Prompt       []int                      `json:"prompt_tokens"`
	Steps        []gemmaLlamaCPPFixtureStep `json:"steps"`
}

type gemmaLlamaCPPFixtureStep struct {
	TopToken       int                `json:"top_token"`
	SelectedLogits map[string]float64 `json:"selected_logits"`
}

func validateGemmaLlamaCPPFixture(fx gemmaLlamaCPPFixture) error {
	if fx.LlamaCPPRevision != gemmaFixtureLlamaCPPRevision {
		return fmt.Errorf("llama.cpp revision %q, want %q", fx.LlamaCPPRevision, gemmaFixtureLlamaCPPRevision)
	}
	if fx.Tolerance <= 0 {
		return fmt.Errorf("logit_tolerance must be positive")
	}
	if len(fx.Cases) != 2 {
		return fmt.Errorf("fixture must contain exactly Gemma2 and Gemma3 cases")
	}
	seen := map[string]bool{}
	for i, tc := range fx.Cases {
		if tc.Architecture != "gemma2" && tc.Architecture != "gemma3" {
			return fmt.Errorf("case %d architecture %q is not Gemma2/Gemma3", i, tc.Architecture)
		}
		if seen[tc.Architecture] {
			return fmt.Errorf("duplicate %s case", tc.Architecture)
		}
		seen[tc.Architecture] = true
		if tc.Model == "" || len(tc.Prompt) == 0 || len(tc.Steps) != len(tc.Prompt) {
			return fmt.Errorf("case %q requires model and one step per prompt token", tc.Name)
		}
		for step, want := range tc.Steps {
			if want.TopToken < 0 || len(want.SelectedLogits) == 0 {
				return fmt.Errorf("case %q step %d has no strict reference output", tc.Name, step)
			}
			if _, ok := want.SelectedLogits[strconv.Itoa(want.TopToken)]; !ok {
				return fmt.Errorf("case %q step %d selected_logits omits top token %d", tc.Name, step, want.TopToken)
			}
		}
	}
	return nil
}

func TestGemmaRealGGUFLlamaCPPFixture(t *testing.T) {
	fixturePath := os.Getenv("GO_PHERENCE_GEMMA_LLAMA_CPP_FIXTURE")
	if fixturePath == "" {
		t.Skip("set GO_PHERENCE_GEMMA_LLAMA_CPP_FIXTURE for the opt-in real-GGUF parity gate")
	}
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fx gemmaLlamaCPPFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatal(err)
	}
	if err := validateGemmaLlamaCPPFixture(fx); err != nil {
		t.Fatal(err)
	}

	for _, tc := range fx.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			modelPath := tc.Model
			if !filepath.IsAbs(modelPath) {
				modelPath = filepath.Join(filepath.Dir(fixturePath), modelPath)
			}
			m, err := LoadGGUFLlama(modelPath, board.SIMDBackend{})
			if err != nil {
				t.Fatal(err)
			}
			if m.Config.Architecture != tc.Architecture {
				t.Fatalf("loaded architecture=%q want=%q", m.Config.Architecture, tc.Architecture)
			}
			if len(tc.Prompt) > m.Config.MaxSeqLen {
				t.Fatalf("prompt length=%d exceeds model context=%d", len(tc.Prompt), m.Config.MaxSeqLen)
			}
			for step, token := range tc.Prompt {
				if token < 0 || token >= m.Config.VocabSize {
					t.Fatalf("prompt step %d token=%d outside vocabulary [0,%d)", step, token, m.Config.VocabSize)
				}
			}
			kvDim := m.Config.NumKVHeads * m.Config.HeadDim
			kvK := make([][]float32, len(m.Layers))
			kvV := make([][]float32, len(m.Layers))
			for l := range m.Layers {
				kvK[l] = make([]float32, len(tc.Prompt)*kvDim)
				kvV[l] = make([]float32, len(tc.Prompt)*kvDim)
			}
			state := m.NewForwardState()
			for step, token := range tc.Prompt {
				logits := m.ForwardState(state, token, step, kvK, kvV)
				want := tc.Steps[step]
				if got := argmaxF32(logits); got != want.TopToken {
					t.Fatalf("step %d top token=%d want=%d", step, got, want.TopToken)
				}
				for tokenText, wantLogit := range want.SelectedLogits {
					tokenID, err := strconv.Atoi(tokenText)
					if err != nil || tokenID < 0 || tokenID >= len(logits) {
						t.Fatalf("step %d invalid selected token %q", step, tokenText)
					}
					if diff := math.Abs(float64(logits[tokenID]) - wantLogit); diff > fx.Tolerance {
						t.Fatalf("step %d token %d logit=%g want=%g diff=%g tolerance=%g", step, tokenID, logits[tokenID], wantLogit, diff, fx.Tolerance)
					}
				}
			}
		})
	}
}

func TestValidateGemmaLlamaCPPFixture(t *testing.T) {
	valid := gemmaLlamaCPPFixture{
		LlamaCPPRevision: gemmaFixtureLlamaCPPRevision,
		Tolerance:        1e-3,
		Cases: []gemmaLlamaCPPFixtureCase{
			{Name: "g2", Architecture: "gemma2", Model: "g2.gguf", Prompt: []int{1}, Steps: []gemmaLlamaCPPFixtureStep{{TopToken: 2, SelectedLogits: map[string]float64{"2": 1}}}},
			{Name: "g3", Architecture: "gemma3", Model: "g3.gguf", Prompt: []int{1}, Steps: []gemmaLlamaCPPFixtureStep{{TopToken: 3, SelectedLogits: map[string]float64{"3": 1}}}},
		},
	}
	if err := validateGemmaLlamaCPPFixture(valid); err != nil {
		t.Fatalf("valid fixture: %v", err)
	}
	valid.LlamaCPPRevision = "unreviewed"
	if err := validateGemmaLlamaCPPFixture(valid); err == nil {
		t.Fatal("accepted fixture from unpinned llama.cpp revision")
	}
}
