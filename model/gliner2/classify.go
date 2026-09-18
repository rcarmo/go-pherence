package gliner2

import "fmt"

// ClassificationScores preserves label order and independent probabilities.
// Selection (single-label versus multi-label) is a caller policy.
type ClassificationScores struct {
	Task          string    `json:"task"`
	Labels        []string  `json:"labels"`
	Logits        []float32 `json:"logits"`
	Probabilities []float64 `json:"probabilities"`
}

func (m *EntityModel) Classify(text, task string, labels []string, maxTokens int) (ClassificationScores, error) {
	if m == nil || m.Tokenizer == nil {
		return ClassificationScores{}, fmt.Errorf("uninitialised model")
	}
	input, err := m.Tokenizer.PrepareClassification(text, task, labels, maxTokens)
	if err != nil {
		return ClassificationScores{}, err
	}
	mask := make([]bool, len(input.IDs))
	for i := range mask {
		mask[i] = true
	}
	hidden, err := m.Encoder.Encode(input.IDs, mask)
	if err != nil {
		return ClassificationScores{}, err
	}
	choices := make([][]float32, len(labels))
	for i, p := range input.QueryPositions {
		choices[i] = hidden[p]
	}
	logits, err := m.Classifier.Forward(choices)
	if err != nil {
		return ClassificationScores{}, err
	}
	temperature := m.Config.BoundaryHead.ClassificationTemperature
	if temperature <= 0 {
		return ClassificationScores{}, fmt.Errorf("invalid classification temperature")
	}
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = sigmoid(float64(v) / temperature)
	}
	return ClassificationScores{task, append([]string(nil), labels...), logits, probs}, nil
}
