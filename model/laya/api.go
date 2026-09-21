package laya

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/model/modernbert"
	"math"
	"strings"
)

type Criterion struct {
	ID          string `json:"id"`
	Description string `json:"description,omitempty"`
}
type Question struct {
	Type         QuestionType `json:"type"`
	Instructions string       `json:"instructions"`
	Criteria     []Criterion  `json:"criteria,omitempty"`
}

type NamedQuestion struct {
	ID       string
	Question Question
}

type Action struct {
	ActProbability float32 `json:"act_probability"`
}

type Answer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Score         float32            `json:"score,omitempty"`
	Noul          float32            `json:"noul,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float32 `json:"probabilities,omitempty"`
	Confidence    float32            `json:"confidence"`
	Action        Action             `json:"action"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type Response struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
}

func optionBucket(t QuestionType, options int) string {
	size := "11+"
	if options <= 2 {
		size = "2"
	} else if options <= 5 {
		size = "3-5"
	} else if options <= 10 {
		size = "6-10"
	}
	return t.String() + ":" + size
}
func validTemperatureBucket(bucket string) bool {
	for _, t := range []QuestionType{Choice, Score, Noul} {
		for _, n := range []int{2, 3, 6, 11} {
			if bucket == optionBucket(t, n) {
				return true
			}
		}
	}
	return false
}
func round4(v float32) float32 { return float32(math.Round(float64(v)*1e4) / 1e4) }

func (t QuestionType) String() string {
	switch t {
	case Choice:
		return "choice"
	case Score:
		return "score"
	case Noul:
		return "noul"
	}
	return ""
}
func renderOptions(q Question) ([]string, error) {
	switch q.Type {
	case Choice:
		if len(q.Criteria) < 2 {
			return nil, fmt.Errorf("laya: choice needs at least two criteria")
		}
		out := make([]string, len(q.Criteria))
		for i, c := range q.Criteria {
			if c.ID == "" {
				return nil, fmt.Errorf("laya: empty criterion ID")
			}
			out[i] = c.ID
			if c.Description != "" {
				out[i] += ": " + c.Description
			}
		}
		return out, nil
	case Score:
		if len(q.Criteria) < 2 {
			return nil, fmt.Errorf("laya: score needs at least two criteria")
		}
		out := make([]string, len(q.Criteria))
		for i, c := range q.Criteria {
			out[i] = fmt.Sprintf("level %d: %s", i, c.Description)
		}
		return out, nil
	case Noul:
		desc := map[string]string{}
		for _, c := range q.Criteria {
			if c.ID != "false" && c.ID != "true" {
				return nil, fmt.Errorf("laya: noul criterion must be false/true")
			}
			desc[c.ID] = c.Description
		}
		f, t := desc["false"], desc["true"]
		if f == "" {
			f = "no, the statement does not hold"
		}
		if t == "" {
			t = "yes, the statement holds"
		}
		return []string{"false: " + f, "true: " + t}, nil
	}
	return nil, fmt.Errorf("laya: invalid question type")
}

// BuildSequence returns the exact Laya input and option marker positions.
func BuildSequence(tok *modernbert.Tokenizer, state string, q Question, maxLen, headMaxLen int, truncateLeft bool) ([]int, []int, error) {
	if tok == nil || maxLen < 1 || headMaxLen < 16 || headMaxLen > maxLen {
		return nil, nil, fmt.Errorf("laya: invalid sequence limits")
	}
	opts, e := renderOptions(q)
	if e != nil {
		return nil, nil, e
	}
	if len(opts) > 255 {
		return nil, nil, fmt.Errorf("laya: too many options")
	}
	clean := func(s string) string { return strings.ReplaceAll(s, "[MASK]", " ") }
	head, _ := tok.Encode(q.Type.String()+" question: "+clean(q.Instructions), false)
	parts := make([][]int, len(opts))
	total := 0
	for i, o := range opts {
		ids, _ := tok.Encode(" "+clean(o), false)
		if len(ids) > 48 {
			ids = ids[:48]
		}
		parts[i] = append([]int{tok.Mask}, ids...)
		total += len(parts[i])
	}
	budget := headMaxLen - total
	if budget < 16 {
		per := max(4, (headMaxLen-16)/max(1, len(parts)))
		total = 0
		for i := range parts {
			if len(parts[i]) > per {
				parts[i] = parts[i][:per]
			}
			total += len(parts[i])
		}
		budget = headMaxLen - total
	}
	if len(head) > max(8, budget) {
		head = head[:max(8, budget)]
	}
	ids := make([]int, 0, maxLen)
	ids = append(ids, tok.CLS)
	ids = append(ids, head...)
	ids = append(ids, tok.SEP)
	markers := make([]int, 0, len(parts))
	for _, part := range parts {
		markers = append(markers, len(ids))
		ids = append(ids, part...)
	}
	ids = append(ids, tok.SEP)
	stateIDs, _ := tok.Encode(clean(state), false)
	room := max(0, maxLen-len(ids)-1)
	if len(stateIDs) > room {
		if truncateLeft {
			stateIDs = stateIDs[len(stateIDs)-room:]
		} else {
			stateIDs = stateIDs[:room]
		}
	}
	ids = append(ids, stateIDs...)
	ids = append(ids, tok.SEP)
	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}
	valid := markers[:0]
	for _, m := range markers {
		if m < len(ids) {
			valid = append(valid, m)
		}
	}
	return ids, valid, nil
}

// SystemOne evaluates ordered questions and assembles the public Laya response.
// Question order is explicit so choice and evaluation ordering never depends on a Go map.
func (m *Model) SystemOne(tok *modernbert.Tokenizer, state string, questions []NamedQuestion, cfg Config) (Response, error) {
	if m == nil || tok == nil || len(questions) == 0 {
		return Response{}, fmt.Errorf("laya: invalid request")
	}
	response := Response{Model: "laya-rl-agent", Answers: make(map[string]Answer, len(questions))}
	for _, named := range questions {
		if named.ID == "" {
			return Response{}, fmt.Errorf("laya: empty question ID")
		}
		if _, exists := response.Answers[named.ID]; exists {
			return Response{}, fmt.Errorf("laya: duplicate question ID %q", named.ID)
		}
		ids, markers, err := BuildSequence(tok, state, named.Question, cfg.MaxLen, cfg.HeadMaxLen, false)
		if err != nil {
			return Response{}, err
		}
		mask := make([]bool, len(ids))
		for i := range mask {
			mask[i] = true
		}
		answer, err := m.Answer(ids, mask, markers, named.Question)
		if err != nil {
			return Response{}, err
		}
		response.Answers[named.ID] = answer
		response.Usage.InputTokens += len(ids)
	}
	return response, nil
}

func (m *Model) Answer(ids []int, mask []bool, markers []int, q Question) (Answer, error) {
	opts, err := renderOptions(q)
	if err != nil {
		return Answer{}, err
	}
	if len(opts) != len(markers) {
		return Answer{}, fmt.Errorf("laya: option count mismatch")
	}
	out, err := m.Forward(ids, mask, markers, q.Type)
	if err != nil {
		return Answer{}, err
	}
	return m.answerFromOutput(out, q)
}

// answerFromOutput isolates the public response contract from model arithmetic.
func (m *Model) answerFromOutput(out Output, q Question) (Answer, error) {
	opts, err := renderOptions(q)
	if err != nil {
		return Answer{}, err
	}
	if len(opts) != len(out.Logits) {
		return Answer{}, fmt.Errorf("laya: option count mismatch")
	}
	for _, v := range append(out.Logits, out.ActionLogits...) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return Answer{}, fmt.Errorf("laya: nonfinite output")
		}
	}
	temp := m.temp[q.Type]
	if v, ok := m.tempByOptions[optionBucket(q.Type, len(out.Logits))]; ok {
		temp = v
	}
	if temp < 1e-3 {
		temp = 1e-3
	}
	probs := make([]float64, len(out.Logits))
	peak := float64(out.Logits[0] / temp)
	for _, v := range out.Logits[1:] {
		if z := float64(v / temp); z > peak {
			peak = z
		}
	}
	var sum float64
	for i, v := range out.Logits {
		probs[i] = math.Exp(float64(v/temp) - peak)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	var ent float64
	for _, p := range probs {
		if p > 0 {
			ent -= p * math.Log(p)
		}
	}
	confidence := float32(1 - ent/math.Log(float64(len(probs))))
	a := Answer{Type: q.Type.String(), Probabilities: map[string]float32{}, Confidence: round4(confidence)}
	actions := append([]float32(nil), out.ActionLogits...)
	simd.SoftmaxRowsInPlace(actions, 1, len(actions))
	if len(actions) > 0 {
		a.Action.ActProbability = round4(actions[0])
	}
	if q.Type != Noul {
		for i, p := range probs {
			key := fmt.Sprint(i)
			if q.Type == Choice {
				key = q.Criteria[i].ID
			}
			a.Probabilities[key] = round4(float32(p))
		}
	} else {
		a.Probabilities = nil
	}
	switch q.Type {
	case Choice:
		best := 0
		for i := 1; i < len(probs); i++ {
			if probs[i] > probs[best] {
				best = i
			}
		}
		a.Choice = q.Criteria[best].ID
	case Score:
		a.Legend = make(map[string]string, len(q.Criteria))
		for i, p := range probs {
			a.Score += float32(i) * float32(p)
			a.Legend[fmt.Sprint(i)] = q.Criteria[i].Description
		}
		a.Score = round4(a.Score)
	case Noul:
		a.Noul = round4(float32(probs[1]))
		a.Confidence = round4(float32(max(probs[1], 1-probs[1])))
	}
	return a, nil
}
