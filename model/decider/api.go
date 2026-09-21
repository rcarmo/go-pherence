package decider

import (
	"fmt"
	"math"
)

type Answer struct {
	Type          QuestionType       `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Noul          float32            `json:"noul,omitempty"`
	Score         float32            `json:"score,omitempty"`
	Confidence    float32            `json:"confidence,omitempty"`
	Certainty     float32            `json:"certainty,omitempty"`
	Probabilities map[string]float32 `json:"probabilities,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	LevelFit      map[string]float32 `json:"level_fit,omitempty"`
	FitMass       float32            `json:"fit_mass,omitempty"`
}

type NamedAnswer struct {
	ID     string `json:"id"`
	Answer Answer `json:"answer"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type Response struct {
	Model    string        `json:"model"`
	Revision string        `json:"revision"`
	Answers  []NamedAnswer `json:"answers"`
	Usage    Usage         `json:"usage"`
}

// SystemOne scores every question independently, preserving question and
// criterion order in slices rather than relying on Go map iteration.
func (r *Runtime) SystemOne(state any, questions []NamedQuestion) (Response, error) {
	if r == nil || len(questions) == 0 {
		return Response{}, fmt.Errorf("decider: invalid request")
	}
	stateText, err := RenderState(state)
	if err != nil || stateText == "" {
		return Response{}, fmt.Errorf("decider: invalid state")
	}
	rendered := make([]renderedQuestion, len(questions))
	seen := map[string]bool{}
	for i, named := range questions {
		if named.ID == "" || seen[named.ID] {
			return Response{}, fmt.Errorf("decider: question IDs must be nonempty and unique")
		}
		seen[named.ID] = true
		rendered[i], err = renderQuestion(named.Question)
		if err != nil {
			return Response{}, fmt.Errorf("decider: question %q: %w", named.ID, err)
		}
	}
	rows := planRows(rendered)
	rowProb := make([][]float32, len(rows))
	usage := 0
	for i, row := range rows {
		logits, ids, err := r.scoreRow(stateText, row)
		if err != nil {
			return Response{}, err
		}
		usage += len(ids)
		rowProb[i], err = softmax(logits, r.Temperature)
		if err != nil {
			return Response{}, err
		}
	}
	answers, err := assemble(rendered, rows, rowProb)
	if err != nil {
		return Response{}, err
	}
	out := Response{Model: ModelID, Revision: ModelPin, Answers: make([]NamedAnswer, len(questions)), Usage: Usage{InputTokens: usage}}
	for i := range questions {
		out.Answers[i] = NamedAnswer{ID: questions[i].ID, Answer: answers[i]}
	}
	return out, nil
}

func assemble(questions []renderedQuestion, rows []scoringRow, probs [][]float32) ([]Answer, error) {
	if len(rows) != len(probs) {
		return nil, fmt.Errorf("decider: result row count mismatch")
	}
	out := make([]Answer, len(questions))
	isolated := make([][]float32, len(questions))
	for i, row := range rows {
		if row.owner < 0 || row.owner >= len(questions) || len(probs[i]) != len(row.options) {
			return nil, fmt.Errorf("decider: invalid result row")
		}
		q := questions[row.owner]
		if q.typeName == Score && !q.listwise {
			if row.level < 0 || row.level >= len(q.legend) || len(probs[i]) != 2 {
				return nil, fmt.Errorf("decider: invalid isolated score row")
			}
			if isolated[row.owner] == nil {
				isolated[row.owner] = make([]float32, len(q.legend))
			}
			isolated[row.owner][row.level] = probs[i][1]
			continue
		}
		answer, err := formatAnswer(q, probs[i])
		if err != nil {
			return nil, err
		}
		out[row.owner] = answer
	}
	for i, fits := range isolated {
		if fits == nil {
			continue
		}
		var mass float32
		for _, p := range fits {
			mass += p
		}
		denom := mass
		if denom == 0 {
			denom = 1e-9
		}
		p := make([]float32, len(fits))
		for j := range p {
			p[j] = fits[j] / denom
		}
		a, err := formatAnswer(questions[i], p)
		if err != nil {
			return nil, err
		}
		a.LevelFit = map[string]float32{}
		for j, fit := range fits {
			a.LevelFit[fmt.Sprint(j)] = round4(fit)
		}
		a.FitMass = round4(mass)
		out[i] = a
	}
	return out, nil
}

func formatAnswer(q renderedQuestion, p []float32) (Answer, error) {
	if len(p) != len(q.options) || len(p) < 2 {
		return Answer{}, fmt.Errorf("decider: probability count mismatch")
	}
	var sum float32
	best := 0
	for i, x := range p {
		if !isFinite(x) || x < 0 {
			return Answer{}, fmt.Errorf("decider: invalid probability")
		}
		sum += x
		if x > p[best] {
			best = i
		}
	}
	if sum <= 0 {
		return Answer{}, fmt.Errorf("decider: zero probability mass")
	}
	for i := range p {
		p[i] /= sum
	}
	switch q.typeName {
	case Noul:
		return Answer{Type: Noul, Noul: round4(p[1])}, nil
	case Choice:
		a := Answer{Type: Choice, Choice: q.names[best].(string), Confidence: round4(p[best]), Certainty: round4(certainty(p)), Probabilities: map[string]float32{}}
		for i, n := range q.names {
			a.Probabilities[n.(string)] = round4(p[i])
		}
		return a, nil
	case Score:
		a := Answer{Type: Score, Confidence: round4(p[best]), Certainty: round4(certainty(p)), Probabilities: map[string]float32{}, Legend: map[string]string{}}
		for i, x := range p {
			a.Score += float32(i) * x
			a.Probabilities[fmt.Sprint(i)] = round4(x)
			a.Legend[fmt.Sprint(i)] = q.legend[i]
		}
		a.Score = round2(a.Score)
		if math.IsNaN(float64(a.Score)) {
			return Answer{}, fmt.Errorf("decider: invalid score")
		}
		return a, nil
	default:
		return Answer{}, fmt.Errorf("decider: unsupported answer type")
	}
}
