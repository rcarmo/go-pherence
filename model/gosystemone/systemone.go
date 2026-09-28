package gosystemone

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// SystemOneRequest implements TypeSafe's state/questions wire types. Questions
// remain raw until compilation so JSON insertion order is not lost in a Go map.
type SystemOneRequest struct {
	Model     string          `json:"model,omitempty"`
	State     json.RawMessage `json:"state"`
	Questions json.RawMessage `json:"questions"`
}

// UnmarshalJSON rejects duplicate and case-variant top-level members on this
// route without changing the established /v1/decision decoder.
func (r *SystemOneRequest) UnmarshalJSON(data []byte) error {
	members, err := orderedObject(data)
	if err != nil {
		return err
	}
	var decoded SystemOneRequest
	for _, member := range members {
		switch member.name {
		case "model":
			if err := json.Unmarshal(member.value, &decoded.Model); err != nil {
				return fmt.Errorf("model must be a string")
			}
		case "state":
			decoded.State = member.value
		case "questions":
			decoded.Questions = member.value
		default:
			return fmt.Errorf("unknown property %q", member.name)
		}
	}
	*r = decoded
	return nil
}

type SystemOneUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type SystemOneAnswer interface{ systemOneAnswer() }

type NoulAnswer struct {
	Type string  `json:"type"`
	Noul float64 `json:"noul"`
}

type ChoiceAnswer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}

type ScoreAnswer struct {
	Type          string                     `json:"type"`
	Score         float64                    `json:"score"`
	Confidence    float64                    `json:"confidence"`
	Probabilities map[string]float64         `json:"probabilities"`
	Legend        map[string]json.RawMessage `json:"legend"`
}

func (NoulAnswer) systemOneAnswer()   {}
func (ChoiceAnswer) systemOneAnswer() {}
func (ScoreAnswer) systemOneAnswer()  {}

type SystemOneResponse struct {
	Model   string                     `json:"model"`
	Answers map[string]SystemOneAnswer `json:"answers"`
	Usage   SystemOneUsage             `json:"usage"`
}

type namedJSON struct {
	name  string
	value json.RawMessage
}

// orderedObject rejects duplicate names instead of silently replacing an option.
func orderedObject(raw json.RawMessage) ([]namedJSON, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	token, err := dec.Token()
	if err != nil || token != json.Delim('{') {
		return nil, fmt.Errorf("expected an object")
	}
	var out []namedJSON
	seen := make(map[string]bool)
	for dec.More() {
		token, err := dec.Token()
		if err != nil {
			return nil, err
		}
		name, ok := token.(string)
		if !ok || seen[name] {
			return nil, fmt.Errorf("duplicate or invalid object key %q", name)
		}
		seen[name] = true
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil, err
		}
		out = append(out, namedJSON{name, value})
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("expected exactly one object")
	}
	return out, nil
}

type systemQuestion struct {
	name, kind  string
	keys        []string
	legend      map[string]json.RawMessage
	instruction json.RawMessage
	criteria    json.RawMessage
}

// entryText implements EntryType: string, object, array, or null. Numbers and
// booleans are allowed inside objects/arrays, but not as top-level entries.
func entryText(raw json.RawMessage, optional bool) (string, error) {
	if len(raw) == 0 {
		if optional {
			return "null", nil
		}
		return "", fmt.Errorf("entry is required (explicit null is allowed)")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var value any
	if err := dec.Decode(&value); err != nil {
		return "", err
	}
	var extra any
	if dec.Decode(&extra) != io.EOF {
		return "", fmt.Errorf("entry must contain one JSON value")
	}
	switch v := value.(type) {
	case string:
		return v, nil
	case nil, map[string]any, []any:
		var compact bytes.Buffer
		if err := json.Compact(&compact, raw); err != nil {
			return "", err
		}
		return compact.String(), nil
	default:
		return "", fmt.Errorf("entry must be text, object, array or null")
	}
}

// Validate decoded strings as well as JSON source: escaped control markers
// must not bypass the tokenizer's delimiter-forgery protection.
func validateSystemOneText(v UserTextValidator, raw json.RawMessage) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	for {
		token, err := dec.Token()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if text, ok := token.(string); ok {
			if err := v.ValidateUserText(text); err != nil {
				return err
			}
		}
	}
}

func compileSystemOne(request SystemOneRequest) (CompiledSchema, []systemQuestion, string, error) {
	state, err := entryText(request.State, false)
	if err != nil {
		return CompiledSchema{}, nil, "", fmt.Errorf("state: %w", err)
	}
	questions, err := orderedObject(request.Questions)
	if err != nil {
		return CompiledSchema{}, nil, "", fmt.Errorf("questions: %w", err)
	}
	if len(questions) < 1 || len(questions) > MaxFields {
		return CompiledSchema{}, nil, "", fmt.Errorf("questions must contain 1-%d questions", MaxFields)
	}
	var schema CompiledSchema
	var meta []systemQuestion
	var prompt strings.Builder
	prompt.WriteString("Answer each question independently using the supplied state and rubric. Select only an allowed JSON value for that question.\n\nQuestions:\n")
	for _, entry := range questions {
		q := systemQuestion{name: entry.name}
		if q.name == "" {
			return schema, nil, "", fmt.Errorf("question name must not be empty")
		}
		members, err := orderedObject(entry.value)
		if err != nil {
			return schema, nil, "", fmt.Errorf("question %q: %w", q.name, err)
		}
		for _, member := range members {
			switch member.name {
			case "type":
				if err := json.Unmarshal(member.value, &q.kind); err != nil {
					return schema, nil, "", fmt.Errorf("question %q: type must be a string", q.name)
				}
			case "instructions":
				q.instruction = member.value
			case "criteria":
				q.criteria = member.value
			default:
				return schema, nil, "", fmt.Errorf("question %q: unknown property %q", q.name, member.name)
			}
		}
		instruction, err := entryText(q.instruction, true)
		if err != nil {
			return schema, nil, "", fmt.Errorf("question %q instructions: %w", q.name, err)
		}
		fmt.Fprintf(&prompt, "%s (%s)\nInstructions: %s\n", strconv.Quote(q.name), q.kind, instruction)
		field := FieldSpec{Name: q.name, Type: q.kind}
		add := func(key, encoded, description string) {
			q.keys = append(q.keys, key)
			field.Candidates = append(field.Candidates, Candidate{ID: fmt.Sprintf("%s:%03d", q.name, len(field.Candidates)), Encoded: encoded, Value: json.RawMessage(encoded)})
			fmt.Fprintf(&prompt, "Value %s: %s\n", encoded, description)
		}
		switch q.kind {
		case "noul":
			descriptions := map[string]string{"false": "The answer is no.", "true": "The answer is yes."}
			if len(q.criteria) > 0 && string(bytes.TrimSpace(q.criteria)) != "null" {
				criteria, err := orderedObject(q.criteria)
				if err != nil {
					return schema, nil, "", fmt.Errorf("question %q: noul criteria must be an object or null", q.name)
				}
				for _, c := range criteria {
					if c.name != "false" && c.name != "true" {
						return schema, nil, "", fmt.Errorf("question %q: noul criteria keys must be true or false", q.name)
					}
					text, err := entryText(c.value, false)
					if err != nil {
						return schema, nil, "", fmt.Errorf("question %q criteria: %w", q.name, err)
					}
					if text != "null" {
						descriptions[c.name] = text
					}
				}
			}
			for _, key := range []string{"false", "true"} {
				add(key, key, descriptions[key])
			}
		case "choice":
			criteria, err := orderedObject(q.criteria)
			if err != nil || len(criteria) < 1 || len(criteria) > MaxCandidates {
				return schema, nil, "", fmt.Errorf("question %q: choice criteria need 1-%d named entries", q.name, MaxCandidates)
			}
			for _, c := range criteria {
				text, err := entryText(c.value, false)
				if err != nil {
					return schema, nil, "", fmt.Errorf("question %q criteria: %w", q.name, err)
				}
				encoded, _ := json.Marshal(c.name)
				add(c.name, string(encoded), text)
			}
		case "score":
			var levels []json.RawMessage
			if err := json.Unmarshal(q.criteria, &levels); err != nil || len(levels) < 2 || len(levels) > MaxCandidates {
				return schema, nil, "", fmt.Errorf("question %q: score criteria need 2-%d ordered entries", q.name, MaxCandidates)
			}
			prompt.WriteString("Rubric levels are ordered from lowest to highest. Select the best matching level.\n")
			q.legend = make(map[string]json.RawMessage, len(levels))
			for i, level := range levels {
				text, err := entryText(level, false)
				if err != nil {
					return schema, nil, "", fmt.Errorf("question %q criteria: %w", q.name, err)
				}
				key := strconv.Itoa(i)
				q.legend[key] = append(json.RawMessage(nil), level...)
				add(key, key, text)
			}
		default:
			return schema, nil, "", fmt.Errorf("question %q: supported types are noul, choice and score", q.name)
		}
		schema.Fields = append(schema.Fields, field)
		schema.Inputs = append(schema.Inputs, FieldInput{Suffix: "  " + strconv.Quote(q.name) + ": ", Candidates: field.Candidates})
		meta = append(meta, q)
		prompt.WriteByte('\n')
	}
	schema.SystemText = prompt.String()
	return schema, meta, "State: " + state, nil
}

// SystemOne uses the same isolated tree scorer as Decide. Full distributions
// are required for expected scores and noul, so greedy fallback is disabled.
func (e *Engine) SystemOne(ctx context.Context, request SystemOneRequest) (SystemOneResponse, error) {
	if e == nil || e.Tokenizer == nil || e.Scorer == nil {
		return SystemOneResponse{}, fmt.Errorf("go-system-one engine is not configured")
	}
	if err := ctx.Err(); err != nil {
		return SystemOneResponse{}, err
	}
	if validator, ok := e.Tokenizer.(UserTextValidator); ok {
		for _, raw := range []json.RawMessage{request.State, request.Questions} {
			if err := validateSystemOneText(validator, raw); err != nil {
				return SystemOneResponse{}, err
			}
		}
	}
	schema, questions, state, err := compileSystemOne(request)
	if err != nil {
		return SystemOneResponse{}, err
	}
	decision, err := e.decide(ctx, Request{Model: request.Model, Schema: request.Questions, Instructions: schema.SystemText, Contexts: []string{state}, Mode: ModeTree}, &schema)
	if err != nil {
		return SystemOneResponse{}, err
	}
	response := SystemOneResponse{Model: request.Model, Answers: make(map[string]SystemOneAnswer, len(questions)), Usage: SystemOneUsage{InputTokens: decision.Usage.PromptTokens}}
	for _, q := range questions {
		field := decision.Results[0].Fields[q.name]
		response.Answers[q.name], err = systemOneAnswer(q, field)
		if err != nil {
			return SystemOneResponse{}, err
		}
	}
	// Output token accounting measures the serialised answers; no free-form
	// continuation was generated by the model.
	encoded, err := json.Marshal(response.Answers)
	if err != nil {
		return SystemOneResponse{}, err
	}
	response.Usage.OutputTokens = len(e.Tokenizer.Encode(string(encoded)))
	return response, nil
}

func systemOneAnswer(q systemQuestion, field FieldResult) (SystemOneAnswer, error) {
	// A singleton tree has no branching nodes. The legacy decision scorer can
	// therefore return its certain value without a candidate distribution.
	if q.kind == "choice" && len(q.keys) == 1 && len(field.Candidates) == 0 {
		return ChoiceAnswer{Type: q.kind, Choice: q.keys[0], Confidence: 1, Probabilities: map[string]float64{q.keys[0]: 1}}, nil
	}
	if len(field.Candidates) != len(q.keys) || len(q.keys) == 0 {
		return nil, fmt.Errorf("question %q: full distribution required", q.name)
	}
	probabilities := make(map[string]float64, len(q.keys))
	sum, score, peak, winner := 0.0, 0.0, -1.0, 0
	for i, c := range field.Candidates {
		p := c.Probability
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return nil, fmt.Errorf("question %q: invalid probability", q.name)
		}
		sum += p
		score += float64(i) * p
		probabilities[q.keys[i]] = p
		if p > peak {
			peak, winner = p, i
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		return nil, fmt.Errorf("question %q: distribution is not normalised", q.name)
	}
	switch q.kind {
	case "noul":
		return NoulAnswer{Type: q.kind, Noul: probabilities["true"]}, nil
	case "choice":
		confidence := 1.0
		if len(q.keys) > 1 {
			uniform := 1 / float64(len(q.keys))
			confidence = (peak - uniform) / (1 - uniform)
		}
		return ChoiceAnswer{Type: q.kind, Choice: q.keys[winner], Confidence: min(1, max(0, confidence)), Probabilities: probabilities}, nil
	case "score":
		// Local approximation: TypeSafe's exact confidence formula is not
		// published. This is one minus normalised expected distance to mode.
		distance := 0.0
		for i, c := range field.Candidates {
			distance += c.Probability * math.Abs(float64(i-winner))
		}
		confidence := 1 - distance/float64(len(q.keys)-1)
		return ScoreAnswer{Type: q.kind, Score: score, Confidence: min(1, max(0, confidence)), Probabilities: probabilities, Legend: q.legend}, nil
	default:
		return nil, fmt.Errorf("unsupported question type %q", q.kind)
	}
}
