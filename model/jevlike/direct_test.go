package jevlike

import (
	"math"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

type directFake struct {
	calls  int
	logits []float32
}

func (f *directFake) PrefillSelectedLogits(ids, selected []int) ([]float32, error) {
	f.calls++
	return f.logits, nil
}
func directTestPrompt() *Qwen3ChoicePrompt {
	t := &tokenizer.Tokenizer{Vocab: map[string]int{}, InvVocab: map[int]string{}, AddedSpecial: map[string]int{}}
	for c := 32; c < 127; c++ {
		t.Vocab[string(rune(c))] = c
		t.InvVocab[c] = string(rune(c))
	}
	return &Qwen3ChoicePrompt{Tokenizer: t, MaxTokens: 512, TemplateSHA256: supportedQwen3ChoiceTemplateSHA256}
}
func directTestRequest() DirectChoiceRequest {
	return DirectChoiceRequest{Evidence: "Ada lives in Lisbon.", Question: "Where does Ada live?", Candidates: []ChoiceCandidate{{"lisbon", "Lisbon"}, {"paris", "Paris"}}, Temperature: 1}
}
func TestDirectSelectedLogitContract(t *testing.T) {
	p := directTestPrompt()
	f := &directFake{logits: []float32{2, 1}}
	r := directTestRequest()
	got, err := ScoreChoices(f, p, r)
	if err != nil {
		t.Fatal(err)
	}
	if f.calls != 1 || got.SelectedID != "lisbon" || got.IDs[1] != "paris" || got.TokenIDs[0] == got.TokenIDs[1] {
		t.Fatal(got, f.calls)
	}
	if math.Abs(got.Probabilities[0]-1/(1+math.Exp(-1))) > 1e-12 {
		t.Fatal(got)
	}
	if strings.Contains(got.SelectedID, "A") {
		t.Fatal("selected code instead of stable ID")
	}
	p.MaxTokens = 1
	f.calls = 0
	if _, err = ScoreChoices(f, p, r); err == nil || f.calls != 0 {
		t.Fatal("overlength was executed")
	}
}
func TestDirectRejectsInvalidChoicesTemperatureAndControlTokens(t *testing.T) {
	for _, change := range []func(*DirectChoiceRequest){func(r *DirectChoiceRequest) { r.Candidates[1].ID = r.Candidates[0].ID }, func(r *DirectChoiceRequest) { r.Candidates[1].Text = r.Candidates[0].Text }, func(r *DirectChoiceRequest) { r.Temperature = 0 }, func(r *DirectChoiceRequest) { r.Temperature = math.NaN() }, func(r *DirectChoiceRequest) { r.Question = "" }, func(r *DirectChoiceRequest) { r.Evidence = "reserved" }} {
		r := directTestRequest()
		change(&r)
		p := directTestPrompt()
		p.Tokenizer.AddedSpecial["reserved"] = 1000
		f := &directFake{logits: []float32{1, 0}}
		if _, err := ScoreChoices(f, p, r); err == nil || f.calls != 0 {
			t.Fatal("invalid request executed", r)
		}
	}
	p := directTestPrompt()
	p.Tokenizer.AddedSpecial["A"] = 65
	if _, _, _, err := p.Prepare(directTestRequest()); err == nil {
		t.Fatal("special answer token accepted")
	}
}
func TestDirectRejectsBoundaryRetokenisation(t *testing.T) {
	p := directTestPrompt()
	r := directTestRequest()
	prompt, _, _, err := p.Prepare(r)
	if err != nil {
		t.Fatal(err)
	}
	// A synthetic added token consumes the complete prompt+answer, proving
	// isolated A encoding is insufficient and boundary identity is checked.
	p.Tokenizer.AddedSpecial[prompt+"A"] = 9999
	if _, _, _, err = p.Prepare(r); err == nil {
		t.Fatal("boundary merge accepted")
	}
}

func TestDirectSoftmaxTinyTemperatureAndTie(t *testing.T) {
	for _, logits := range [][]float32{{2, 1}, {2, 2}} {
		r := directTestRequest()
		r.Temperature = math.SmallestNonzeroFloat64
		got, err := ScoreChoices(&directFake{logits: logits}, directTestPrompt(), r)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range got.Probabilities {
			if math.IsNaN(p) || math.IsInf(p, 0) {
				t.Fatal("unstable softmax", got)
			}
		}
		if got.SelectedID != "lisbon" {
			t.Fatal("tie policy")
		}
	}
}

func TestDirectNearTieAndStableStrictTiePolicy(t *testing.T) {
	for _, identity := range []string{supportedQwen3ChoiceTemplateSHA256, instructionQwen3ChoiceTemplateSHA256} {
		p := directTestPrompt()
		p.TemplateSHA256 = identity
		r := directTestRequest()
		next := math.Nextafter32(1, 2)
		got, err := ScoreChoices(&directFake{logits: []float32{1, next}}, p, r)
		if err != nil || got.SelectedID != "paris" {
			t.Fatal("one ULP advantage is not a tie", got, err)
		}
		r.Candidates[0], r.Candidates[1] = r.Candidates[1], r.Candidates[0]
		got, err = ScoreChoices(&directFake{logits: []float32{1, 1}}, p, r)
		if err != nil || got.SelectedID != "paris" {
			t.Fatal("strict ties select first supplied candidate", got, err)
		}
	}
}
