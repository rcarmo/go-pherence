// Package decider implements Mapika Decider's typed one-pass decision contract.
package decider

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	SourcePin = "c4daaac28af9fea95d627015cffa2dd5a5926ee6"
	ModelPin  = "1ea54127d3bd52f6d753d9257b32a6380b873907"
	BasePin   = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
	ModelID   = "Mapika/decider-0.8b"
	MaxChoice = 255
	MaxLevels = 10
)

type QuestionType string

const (
	Choice QuestionType = "choice"
	Score  QuestionType = "score"
	Noul   QuestionType = "noul"
)

type Criterion struct {
	Name        string `json:"name"`
	Description any    `json:"description,omitempty"`
}

// Property and Object preserve JSON object order explicitly. Go maps are
// rendered deterministically by encoding/json, but cannot retain caller order.
type Property struct {
	Name  string
	Value any
}
type Object []Property

func (o Object) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	seen := map[string]bool{}
	for i, p := range o {
		if p.Name == "" || seen[p.Name] {
			return nil, fmt.Errorf("decider: object keys must be nonempty and unique")
		}
		seen[p.Name] = true
		if i > 0 {
			b.WriteByte(',')
		}
		name, _ := json.Marshal(p.Name)
		value, err := json.Marshal(p.Value)
		if err != nil {
			return nil, err
		}
		b.Write(name)
		b.WriteByte(':')
		b.Write(value)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

type Question struct {
	Type         QuestionType `json:"type"`
	Instructions any          `json:"instructions"`
	Criteria     []Criterion  `json:"criteria,omitempty"`
	Levels       []any        `json:"levels,omitempty"`
	Listwise     bool         `json:"listwise,omitempty"`
}

type NamedQuestion struct {
	ID       string   `json:"id"`
	Question Question `json:"question"`
}

type renderedQuestion struct {
	typeName QuestionType
	text     string
	options  []string
	names    []any
	legend   []string
	listwise bool
}

type scoringRow struct {
	question string
	options  []string
	owner    int
	level    int
}

var levelPrefix = regexp.MustCompile(`^\s*-?\d+\s*:\s*`)

func textValue(v any) (string, error) {
	if v == nil {
		return "", nil
	}
	if s, ok := v.(string); ok {
		return s, nil
	}
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(v); err != nil {
		return "", err
	}
	return pythonJSONSpacing(strings.TrimSuffix(b.String(), "\n")), nil
}

func renderQuestion(q Question) (renderedQuestion, error) {
	text, err := textValue(q.Instructions)
	if err != nil || strings.TrimSpace(text) == "" {
		return renderedQuestion{}, fmt.Errorf("decider: question instructions must be nonempty")
	}
	r := renderedQuestion{typeName: q.Type, text: text, listwise: q.Listwise}
	switch q.Type {
	case Choice:
		if len(q.Criteria) < 2 || len(q.Criteria) > MaxChoice {
			return r, fmt.Errorf("decider: choice needs 2..%d criteria", MaxChoice)
		}
		seen := map[string]bool{}
		for _, c := range q.Criteria {
			if strings.TrimSpace(c.Name) == "" || seen[c.Name] {
				return r, fmt.Errorf("decider: choice names must be nonempty and unique")
			}
			seen[c.Name] = true
			o := c.Name
			if c.Description != nil && c.Description != "" {
				d, err := textValue(c.Description)
				if err != nil {
					return r, err
				}
				o += ": " + d
			}
			r.options = append(r.options, o)
			r.names = append(r.names, c.Name)
		}
	case Score:
		if len(q.Levels) < 2 || len(q.Levels) > MaxLevels {
			return r, fmt.Errorf("decider: score needs 2..%d levels", MaxLevels)
		}
		for i, level := range q.Levels {
			d, err := textValue(level)
			if err != nil || strings.TrimSpace(d) == "" {
				return r, fmt.Errorf("decider: score level %d is invalid", i)
			}
			r.options = append(r.options, strconv.Itoa(i)+": "+d)
			r.names = append(r.names, i)
			r.legend = append(r.legend, d)
		}
	case Noul:
		if len(q.Criteria) != 0 && len(q.Criteria) != 2 {
			return r, fmt.Errorf("decider: noul criteria must be absent or false/true")
		}
		desc := map[string]any{}
		for _, c := range q.Criteria {
			if c.Name != "false" && c.Name != "true" {
				return r, fmt.Errorf("decider: noul criterion must be false or true")
			}
			desc[c.Name] = c.Description
		}
		for _, p := range []struct {
			name string
			base string
			val  bool
		}{{"false", "no", false}, {"true", "yes", true}} {
			o := p.base
			if v := desc[p.name]; v != nil && v != "" {
				d, err := textValue(v)
				if err != nil {
					return r, err
				}
				o += ": " + d
			}
			r.options = append(r.options, o)
			r.names = append(r.names, p.val)
		}
	default:
		return r, fmt.Errorf("decider: unsupported question type %q", q.Type)
	}
	return r, nil
}

func planRows(questions []renderedQuestion) []scoringRow {
	var rows []scoringRow
	for i, q := range questions {
		if q.typeName == Score && !q.listwise {
			for j, level := range q.legend {
				level = levelPrefix.ReplaceAllString(level, "")
				rows = append(rows, scoringRow{question: q.text + "\nProposed answer: " + level + "\nDoes the proposed answer fit?", options: []string{"no", "yes"}, owner: i, level: j})
			}
		} else {
			rows = append(rows, scoringRow{question: q.text, options: append([]string(nil), q.options...), owner: i, level: -1})
		}
	}
	return rows
}

// RenderState reproduces upstream's ensure_ascii=False JSON rendering. String
// states pass through unchanged; long arrays are annotated with their indices.
func RenderState(state any) (string, error) {
	if s, ok := state.(string); ok {
		return s, nil
	}
	v := annotateIndices(state)
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(v); err != nil {
		return "", fmt.Errorf("decider: render state: %w", err)
	}
	return pythonJSONSpacing(strings.TrimSuffix(b.String(), "\n")), nil
}

func pythonJSONSpacing(compact string) string {
	var out strings.Builder
	out.Grow(len(compact) + 16)
	quoted, escaped := false, false
	for _, r := range compact {
		if quoted {
			out.WriteRune(r)
			if escaped {
				escaped = false
			} else if r == '\\' {
				escaped = true
			} else if r == '"' {
				quoted = false
			}
			continue
		}
		switch r {
		case '"':
			quoted = true
			out.WriteRune(r)
		case ',':
			out.WriteString(", ")
		case ':':
			out.WriteString(": ")
		default:
			out.WriteRune(r)
		}
	}
	return out.String()
}

func annotateIndices(v any) any {
	switch x := v.(type) {
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			item = annotateIndices(item)
			if len(x) >= 8 {
				if m, ok := item.(map[string]any); ok {
					z := map[string]any{"_index": i}
					for k, value := range m {
						z[k] = value
					}
					item = z
				} else {
					item = map[string]any{"_index": i, "value": item}
				}
			}
			out[i] = item
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, item := range x {
			out[k] = annotateIndices(item)
		}
		return out
	case Object:
		out := make(Object, len(x))
		for i, p := range x {
			out[i] = Property{Name: p.Name, Value: annotateIndices(p.Value)}
		}
		return out
	default:
		return v
	}
}

func softmax(logits []float32, temperature float32) ([]float32, error) {
	if temperature <= 0 || math.IsNaN(float64(temperature)) || math.IsInf(float64(temperature), 0) || len(logits) == 0 {
		return nil, fmt.Errorf("decider: invalid logits or temperature")
	}
	peak := float32(-math.MaxFloat32)
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("decider: non-finite logit")
		}
		if v > peak {
			peak = v
		}
	}
	out := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		out[i] = float32(math.Exp(float64((v - peak) / temperature)))
		sum += float64(out[i])
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
	return out, nil
}

func certainty(p []float32) float32 {
	if len(p) < 2 {
		return 1
	}
	var h float64
	for _, x := range p {
		if x > 0 {
			h -= float64(x) * math.Log(float64(x))
		}
	}
	return max(0, float32(1-h/math.Log(float64(len(p)))))
}

func round4(x float32) float32 { return float32(math.Round(float64(x)*1e4) / 1e4) }
func round2(x float32) float32 { return float32(math.Round(float64(x)*1e2) / 1e2) }

// SortedLegend is a convenience for callers translating numeric JSON legends.
func SortedLegend(legend map[string]any) ([]any, error) {
	keys := make([]string, 0, len(legend))
	for k := range legend {
		if _, err := strconv.ParseFloat(k, 64); err != nil {
			return nil, fmt.Errorf("decider: nonnumeric legend key %q", k)
		}
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, _ := strconv.ParseFloat(keys[i], 64)
		b, _ := strconv.ParseFloat(keys[j], 64)
		return a < b
	})
	out := make([]any, len(keys))
	for i, k := range keys {
		out[i] = legend[k]
	}
	return out, nil
}
