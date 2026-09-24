// Package simplejev holds independently authored, model-free label scoring
// primitives for a future Simple-JEV-style adapter. It does not implement the
// upstream prompt template or a model backend.
package simplejev

import (
	"fmt"
	"math"
)

// Label is one distinct, ordered next-token candidate. Value is the public
// value for an ordinal question; choice callers may leave it at zero.
type Label struct {
	TokenID int
	ID      string
	Logit   float32
	Value   float64
}

// Distribution returns probabilities in the input order. Only the admitted
// labels are normalised; logits for other vocabulary entries are not used.
// The caller must supply 2–50 distinct token IDs and nonempty public IDs.
func Distribution(labels []Label) ([]float32, error) {
	if len(labels) < 2 || len(labels) > 50 {
		return nil, fmt.Errorf("simplejev: labels count=%d outside [2,50]", len(labels))
	}
	seenIDs, seenTokens := make(map[string]struct{}, len(labels)), make(map[int]struct{}, len(labels))
	peak := float32(-math.MaxFloat32)
	for _, label := range labels {
		if label.TokenID < 0 || label.ID == "" || math.IsNaN(float64(label.Logit)) || math.IsInf(float64(label.Logit), 0) {
			return nil, fmt.Errorf("simplejev: invalid label or logit")
		}
		if _, ok := seenIDs[label.ID]; ok {
			return nil, fmt.Errorf("simplejev: duplicate label ID %q", label.ID)
		}
		if _, ok := seenTokens[label.TokenID]; ok {
			return nil, fmt.Errorf("simplejev: duplicate label token %d", label.TokenID)
		}
		seenIDs[label.ID], seenTokens[label.TokenID] = struct{}{}, struct{}{}
		if label.Logit > peak {
			peak = label.Logit
		}
	}
	weights := make([]float64, len(labels))
	var sum float64
	for i, label := range labels {
		// Subtract in float64 to avoid F32 overflow when finite logits span
		// the whole representable range; normalisation remains bounded.
		weights[i] = math.Exp(float64(label.Logit) - float64(peak))
		sum += weights[i]
	}
	probs := make([]float32, len(labels))
	for i, weight := range weights {
		probs[i] = float32(weight / sum)
	}
	return probs, nil
}

// Choice returns the highest-logit label and its ordered distribution.
// Input order breaks exact logit ties, even when probabilities round to zero.
func Choice(labels []Label) (string, []float32, error) {
	probs, err := Distribution(labels)
	if err != nil {
		return "", nil, err
	}
	winner := 0
	for i := 1; i < len(labels); i++ {
		if labels[i].Logit > labels[winner].Logit {
			winner = i
		}
	}
	return labels[winner].ID, probs, nil
}

// Ordinal returns the expected public value of the supplied ordered labels.
// Values are caller-specified rather than inferred from a prompt or token ID.
func Ordinal(labels []Label) (float64, []float32, error) {
	for _, label := range labels {
		if math.IsNaN(label.Value) || math.IsInf(label.Value, 0) {
			return 0, nil, fmt.Errorf("simplejev: non-finite ordinal value")
		}
	}
	probs, err := Distribution(labels)
	if err != nil {
		return 0, nil, err
	}
	var expected float64
	for i, label := range labels {
		expected += float64(probs[i]) * label.Value
	}
	if math.IsNaN(expected) || math.IsInf(expected, 0) {
		return 0, nil, fmt.Errorf("simplejev: ordinal expectation overflow")
	}
	return expected, probs, nil
}
