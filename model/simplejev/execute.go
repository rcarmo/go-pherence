package simplejev

import "fmt"

// LogitProvider is an injected, model-owned next-token label source. It must
// return exactly one distinct token ID and finite logit per ordered public
// label; the caller owns rendering/tokenisation before invoking this API.
type LogitProvider interface {
	LabelLogits(state string, question Question) ([]Label, error)
}

type Answer struct {
	QuestionID    string    `json:"question_id"`
	Kind          string    `json:"kind"`
	SelectedID    string    `json:"selected_id,omitempty"`
	ExpectedValue float64   `json:"expected_value,omitempty"`
	Probabilities []float32 `json:"probabilities"`
}

// Evaluate checks every provider result before returning any answers. A
// provider error or malformed row never yields a partially accepted response.
// Ordinal levels are the zero-based positions in Question.Labels.
func Evaluate(request Request, provider LogitProvider) ([]Answer, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	if provider == nil {
		return nil, fmt.Errorf("simplejev: nil logit provider")
	}
	answers := make([]Answer, 0, len(request.Questions))
	for _, question := range request.Questions {
		// Give the provider a separate label slice: a callback must not be able
		// to change the validated request or the authoritative label order.
		publicLabels := append([]string(nil), question.Labels...)
		providerQuestion := question
		providerQuestion.Labels = append([]string(nil), publicLabels...)
		labels, err := provider.LabelLogits(request.State, providerQuestion)
		if err != nil {
			return nil, fmt.Errorf("simplejev: question %q: %w", question.ID, err)
		}
		if len(labels) != len(publicLabels) {
			return nil, fmt.Errorf("simplejev: question %q: label count mismatch", question.ID)
		}
		// Copy the provider's slice before supplying ordinal values; providers
		// retain ownership of their buffers and may reuse them between calls.
		owned := make([]Label, len(labels))
		copy(owned, labels)
		for i, label := range owned {
			if label.ID != publicLabels[i] {
				return nil, fmt.Errorf("simplejev: question %q: label order mismatch", question.ID)
			}
			if question.Kind == "ordinal" {
				owned[i].Value = float64(i)
			}
		}
		answer := Answer{QuestionID: question.ID, Kind: question.Kind}
		if question.Kind == "choice" {
			answer.SelectedID, answer.Probabilities, err = Choice(owned)
		} else {
			answer.ExpectedValue, answer.Probabilities, err = Ordinal(owned)
		}
		if err != nil {
			return nil, fmt.Errorf("simplejev: question %q: %w", question.ID, err)
		}
		answers = append(answers, answer)
	}
	return answers, nil
}
