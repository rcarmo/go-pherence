package nimble

import (
	"fmt"
	"math"
)

type FieldResult struct {
	Value             any                `json:"value"`
	Scores            map[string]float32 `json:"scores"`
	Logits            map[string]float32 `json:"logits"`
	CandidateTokenIDs []int              `json:"candidate_token_ids"`
	CodeToChoice      map[string]any     `json:"code_to_choice"`
}
type Result struct {
	Model             string                 `json:"model"`
	Revision          string                 `json:"revision"`
	Backend           string                 `json:"backend"`
	Temperature       float32                `json:"temperature"`
	TemperatureFitted bool                   `json:"temperature_fitted"`
	Context           string                 `json:"context"`
	Output            map[string]any         `json:"output"`
	Fields            map[string]FieldResult `json:"fields"`
}

func Summarize(context, model, revision string, fields []Field, candidateIDs [][]int, logits [][]float32, temperature float32) (Result, error) {
	if err := ValidateSchema(fields); err != nil {
		return Result{}, err
	}
	if !isFinite(temperature) || temperature <= 0 {
		return Result{}, fmt.Errorf("nimble: temperature must be positive and finite")
	}
	if len(candidateIDs) != len(fields) || len(logits) != len(fields) {
		return Result{}, fmt.Errorf("nimble: result row count mismatch")
	}
	r := Result{Model: model, Revision: revision, Backend: "native-go", Temperature: temperature, Context: context, Output: map[string]any{}, Fields: map[string]FieldResult{}}
	for i, f := range fields {
		choices := choicesFor(f)
		if len(candidateIDs[i]) != len(choices) || len(logits[i]) != len(choices) {
			return Result{}, fmt.Errorf("nimble: %s candidate count mismatch", f.Name)
		}
		peak := float32(-math.MaxFloat32)
		for _, v := range logits[i] {
			if !isFinite(v) {
				return Result{}, fmt.Errorf("nimble: %s non-finite logit", f.Name)
			}
			if v > peak {
				peak = v
			}
		}
		probs := make([]float32, len(choices))
		var sum float64
		for j, v := range logits[i] {
			probs[j] = float32(math.Exp(float64((v - peak) / temperature)))
			sum += float64(probs[j])
		}
		best := 0
		fr := FieldResult{Scores: map[string]float32{}, Logits: map[string]float32{}, CandidateTokenIDs: append([]int(nil), candidateIDs[i]...), CodeToChoice: map[string]any{}}
		for j, v := range choices {
			p := float32(float64(probs[j]) / sum)
			key := choiceKey(v)
			fr.Scores[key] = p
			fr.Logits[key] = logits[i][j]
			fr.CodeToChoice[string(rune('A'+j))] = v
			if logits[i][j] > logits[i][best] {
				best = j
			}
		}
		fr.Value = choices[best]
		r.Output[f.Name] = fr.Value
		r.Fields[f.Name] = fr
	}
	return r, nil
}
func isFinite(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
