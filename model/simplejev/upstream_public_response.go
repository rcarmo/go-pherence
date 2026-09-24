package simplejev

import (
	"fmt"
	"math"
	"strconv"
)

// TextStateLogits is an injected, ordered selected-logit source. The caller
// owns tokenizer/template validation; this model-free assembler does not run
// inference. Rows correspond to request question order and their allowed
// choice/score/Noul labels (2..50, 2..50 and exactly nine respectively).
type TextStateLogits interface {
	Rows(request TextStateRequest) ([][]float32, error)
}

type PublicResponse struct {
	Model   string                  `json:"model"`
	Answers map[string]PublicAnswer `json:"answers"`
	Usage   PublicUsage             `json:"usage"`
}

type PublicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type PublicAnswer struct {
	Type          string             `json:"type"`
	Confidence    *float64           `json:"confidence,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Choice        *string            `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]*string `json:"legend,omitempty"`
	Noul          *float64           `json:"noul,omitempty"`
}

// AssembleTextStateResponse returns public, non-advanced answers only. Invalid
// rows fail transactionally; no partial answer or backend buffer escapes.
// Advanced raw logits, diagnostics and prompt/model execution are unsupported.
func AssembleTextStateResponse(request TextStateRequest, provider TextStateLogits, inputTokens int) (PublicResponse, error) {
	var empty PublicResponse
	if request.RawLogits {
		return empty, fmt.Errorf("simplejev: raw logits require unsupported advanced mode")
	}
	if request.Model == "" || len(request.Questions) == 0 || len(request.Questions) > MaxQuestions || len(request.State) > MaxStateBytes || provider == nil || inputTokens < 0 {
		return empty, fmt.Errorf("simplejev: invalid text-state response inputs")
	}
	// Copy the validated envelope before invoking any callback. The provider
	// must not change IDs, criteria, or row order while answers are assembled.
	questions := make([]TextStateQuestion, len(request.Questions))
	for i, q := range request.Questions {
		questions[i] = q
		questions[i].Choices = append([]TextCriterion(nil), q.Choices...)
		questions[i].Levels = append([]*string(nil), q.Levels...)
		questions[i].NoulCriteria = append([]TextCriterion(nil), q.NoulCriteria...)
		if q.Instructions != nil {
			v := *q.Instructions
			questions[i].Instructions = &v
		}
		for j := range questions[i].Choices {
			if v := questions[i].Choices[j].Description; v != nil {
				copied := *v
				questions[i].Choices[j].Description = &copied
			}
		}
		for j := range questions[i].Levels {
			if v := questions[i].Levels[j]; v != nil {
				copied := *v
				questions[i].Levels[j] = &copied
			}
		}
		for j := range questions[i].NoulCriteria {
			if v := questions[i].NoulCriteria[j].Description; v != nil {
				copied := *v
				questions[i].NoulCriteria[j].Description = &copied
			}
		}
	}
	owned := request
	owned.Questions = questions
	rows, err := provider.Rows(owned)
	if err != nil {
		return empty, err
	}
	if len(rows) != len(questions) {
		return empty, fmt.Errorf("simplejev: expected one row per question")
	}
	// A provider may mutate its own request copy. Restore trusted question
	// metadata from the original before applying untrusted numerical rows.
	questions = request.Questions
	answers := make(map[string]PublicAnswer, len(questions))
	for i, q := range questions {
		if q.ID == "" {
			return empty, fmt.Errorf("simplejev: empty question ID")
		}
		if _, exists := answers[q.ID]; exists {
			return empty, fmt.Errorf("simplejev: duplicate question ID")
		}
		var labels []Label
		switch q.Type {
		case "choice":
			if len(q.Choices) < 2 || len(q.Choices) > 50 || len(rows[i]) != len(q.Choices) {
				return empty, fmt.Errorf("simplejev: invalid choice row")
			}
			labels = make([]Label, len(q.Choices))
			for j, c := range q.Choices {
				labels[j] = Label{ID: c.ID, TokenID: j, Logit: rows[i][j]}
			}
		case "score":
			if len(q.Levels) < 2 || len(q.Levels) > 50 || len(rows[i]) != len(q.Levels) {
				return empty, fmt.Errorf("simplejev: invalid score row")
			}
			labels = make([]Label, len(q.Levels))
			for j := range labels {
				labels[j] = Label{ID: strconv.Itoa(j), TokenID: j, Logit: rows[i][j], Value: float64(j)}
			}
		case "noul":
			if len(rows[i]) != 9 {
				return empty, fmt.Errorf("simplejev: invalid Noul row")
			}
			labels = make([]Label, 9)
			for j := range labels {
				labels[j] = Label{ID: strconv.Itoa(j + 1), TokenID: j, Logit: rows[i][j], Value: float64(j + 1)}
			}
		default:
			return empty, fmt.Errorf("simplejev: unsupported question type")
		}
		probs, err := Distribution(labels)
		if err != nil {
			return empty, fmt.Errorf("simplejev: question %q: %w", q.ID, err)
		}
		answer := PublicAnswer{Type: q.Type}
		if q.Type == "noul" {
			var rating float64
			for j, p := range probs {
				rating += float64(p) * float64(j+1)
			}
			value := math.Min(.99, math.Max(.01, .01+(rating-1)*(.98/8)))
			answer.Noul = &value
		} else {
			answer.Probabilities = make(map[string]float64, len(labels))
			winner := 0
			for j, label := range labels {
				answer.Probabilities[label.ID] = float64(probs[j])
				if label.Logit > labels[winner].Logit {
					winner = j
				}
			}
			confidence := float64(probs[winner])
			answer.Confidence = &confidence
			if q.Type == "choice" {
				choice := labels[winner].ID
				answer.Choice = &choice
			} else {
				var score float64
				answer.Legend = make(map[string]*string, len(q.Levels))
				for j, p := range probs {
					score += float64(p) * float64(j)
					answer.Legend[strconv.Itoa(j)] = q.Levels[j]
				}
				answer.Score = &score
			}
		}
		answers[q.ID] = answer
	}
	return PublicResponse{Model: request.Model, Answers: answers, Usage: PublicUsage{InputTokens: inputTokens, OutputTokens: 0}}, nil
}
