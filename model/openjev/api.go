package openjev

import (
	"fmt"
	"math"
)

type Prediction struct {
	Premise       string    `json:"premise"`
	Hypothesis    string    `json:"hypothesis"`
	Label         string    `json:"label"`
	Logits        []float32 `json:"logits"`
	Probabilities []float32 `json:"probabilities"`
	TokenIDs      []int     `json:"token_ids"`
}

type RankedOption struct {
	Option     string     `json:"option"`
	Entailment float32    `json:"entailment"`
	Prediction Prediction `json:"prediction"`
}
type RerankResult struct {
	Index   int            `json:"index"`
	Choice  string         `json:"choice"`
	Options []RankedOption `json:"options"`
}

func Summarize(premise, hypothesis string, ids []int, logits []float32) (Prediction, error) {
	if len(logits) != len(Labels) {
		return Prediction{}, fmt.Errorf("openjev: expected three logits")
	}
	peak := float32(-math.MaxFloat32)
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return Prediction{}, fmt.Errorf("openjev: non-finite logit")
		}
		if v > peak {
			peak = v
		}
	}
	p := make([]float32, len(logits))
	var sum float64
	best := 0
	for i, v := range logits {
		p[i] = float32(math.Exp(float64(v - peak)))
		sum += float64(p[i])
		if v > logits[best] {
			best = i
		}
	}
	for i := range p {
		p[i] = float32(float64(p[i]) / sum)
	}
	return Prediction{Premise: premise, Hypothesis: hypothesis, Label: Labels[best], Logits: append([]float32(nil), logits...), Probabilities: p, TokenIDs: append([]int(nil), ids...)}, nil
}

// Predict scores ordered premise/hypothesis pairs independently.
func (r *Runtime) Predict(pairs [][2]string) ([]Prediction, error) {
	if len(pairs) == 0 {
		return nil, fmt.Errorf("openjev: pairs must not be empty")
	}
	out := make([]Prediction, len(pairs))
	for i, p := range pairs {
		var e error
		out[i], e = r.Score(p[0], p[1])
		if e != nil {
			return nil, e
		}
	}
	return out, nil
}

// Rerank chooses the option with the greatest entailment probability for
// "The correct answer is: <option>". Option order is explicit and preserved.
func (r *Runtime) Rerank(question string, options []string) (RerankResult, error) {
	if len(options) < 2 {
		return RerankResult{}, fmt.Errorf("openjev: rerank needs at least two options")
	}
	pairs := make([][2]string, len(options))
	for i, o := range options {
		if o == "" {
			return RerankResult{}, fmt.Errorf("openjev: empty option")
		}
		pairs[i] = [2]string{question, "The correct answer is: " + o}
	}
	pred, e := r.Predict(pairs)
	if e != nil {
		return RerankResult{}, e
	}
	out := RerankResult{Options: make([]RankedOption, len(options))}
	for i, p := range pred {
		out.Options[i] = RankedOption{Option: options[i], Entailment: p.Probabilities[1], Prediction: p}
		if i == 0 || out.Options[i].Entailment > out.Options[out.Index].Entailment {
			out.Index = i
		}
	}
	out.Choice = options[out.Index]
	return out, nil
}

func (r *Runtime) Grade(question, reference, candidate string) (Prediction, error) {
	if question == "" || reference == "" || candidate == "" {
		return Prediction{}, fmt.Errorf("openjev: grade inputs must be nonempty")
	}
	return r.Score(question+"\nReference answer: "+reference, "Answer: "+candidate)
}
