package decider

import (
	"encoding/json"
	"math"
)

// AnswerV15 is the opt-in Decider 1.5 answer shape. SystemOne continues to
// return its pinned, older Answer contract; this type changes no model path.
type AnswerV15 struct {
	Answer
	XPMax float32 `json:"x_p_max,omitempty"`
}

// MarshalJSON includes a zero Choice/Score confidence: the pinned Answer's
// omitempty tag would otherwise erase the v1.5 uniform-distribution value.
func (a AnswerV15) MarshalJSON() ([]byte, error) {
	type plain AnswerV15
	if a.Type != Choice && a.Type != Score {
		return json.Marshal(plain(a))
	}
	return json.Marshal(struct {
		plain
		Confidence float32 `json:"confidence"`
		XPMax      float32 `json:"x_p_max"`
	}{plain: plain(a), Confidence: a.Confidence, XPMax: a.XPMax})
}

// FormatAnswerV15 formats supplied option probabilities using the newer
// TypeSafe confidence semantics. The question must use this package's existing
// schema; no model or prompt is executed. Score probabilities must already
// represent the final distribution (including isolated-level combination).
func FormatAnswerV15(question Question, probabilities []float32) (AnswerV15, error) {
	q, err := renderQuestion(question)
	if err != nil {
		return AnswerV15{}, err
	}
	p := append([]float32(nil), probabilities...)
	legacy, err := formatAnswer(q, p) // validates and normalises owned p
	if err != nil {
		return AnswerV15{}, err
	}
	answer := AnswerV15{Answer: legacy}
	if q.typeName == Noul {
		return answer, nil
	}
	modal := 0
	for i := 1; i < len(p); i++ {
		if p[i] > p[modal] {
			modal = i
		}
	}
	answer.XPMax = round4(p[modal])
	var confidence float64
	switch q.typeName {
	case Choice:
		confidence = (float64(len(p))*float64(p[modal]) - 1) / float64(len(p)-1)
	case Score:
		var distance, uniformDistance float64
		middle := float64(len(p)-1) / 2
		for i, value := range p {
			distance += float64(value) * math.Abs(float64(i-modal))
			uniformDistance += math.Abs(float64(i) - middle)
		}
		confidence = 1 - distance/(uniformDistance/float64(len(p)))
	}
	answer.Confidence = round4(float32(max(0, min(1, confidence))))
	return answer, nil
}
