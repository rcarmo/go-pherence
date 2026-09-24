package simplejev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"unicode/utf8"
)

// TextCriterion preserves the public candidate ID and optional text description.
// This state-only boundary does not render upstream prompts or run a model.
type TextCriterion struct {
	ID          string
	Description *string
}

type TextStateQuestion struct {
	ID           string
	Type         string
	Instructions *string
	Choices      []TextCriterion // insertion order matters for model labels and ties
	Levels       []*string       // lowest to highest
	NoulCriteria []TextCriterion // optional, keys must be true/false
}

type TextStateRequest struct {
	Model     string
	State     string
	Questions []TextStateQuestion
	RawLogits bool
}

// DecodeTextStateRequest admits only a bounded text-state subset of the pinned
// upstream wire format. Existing DecodeRequest retains its separate contract.
// Unsupported messages, media and structured content fail before inference.
func DecodeTextStateRequest(reader io.Reader) (TextStateRequest, error) {
	var empty TextStateRequest
	if reader == nil {
		return empty, fmt.Errorf("simplejev: nil reader")
	}
	data, err := io.ReadAll(&io.LimitedReader{R: reader, N: MaxRequestBytes + 1})
	if err != nil {
		return empty, err
	}
	if len(data) > MaxRequestBytes || !utf8.Valid(data) {
		return empty, fmt.Errorf("simplejev: invalid request bytes")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	if err := expectDelimiter(d, '{'); err != nil {
		return empty, err
	}
	var request TextStateRequest
	seen := map[string]bool{}
	for d.More() {
		key, err := readKey(d, seen)
		if err != nil {
			return empty, err
		}
		switch key {
		case "model":
			request.Model, err = requiredText(d)
		case "state":
			request.State, err = requiredText(d)
		case "questions":
			request.Questions, err = parseTextQuestions(d)
		case "options":
			request.RawLogits, err = parseTextOptions(d)
		case "messages", "tools", "mm_processor_kwargs", "media_io_kwargs":
			return empty, fmt.Errorf("simplejev: unsupported field %q", key)
		default:
			var ignored json.RawMessage
			err = d.Decode(&ignored) // upstream ignores unrelated top-level fields
		}
		if err != nil {
			return empty, fmt.Errorf("simplejev: %s: %w", key, err)
		}
	}
	if err := expectDelimiter(d, '}'); err != nil {
		return empty, err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return empty, fmt.Errorf("simplejev: trailing data")
	}
	if !seen["model"] || request.Model == "" || !seen["state"] || len(request.State) > MaxStateBytes || !seen["questions"] || len(request.Questions) == 0 {
		return empty, fmt.Errorf("simplejev: missing or invalid model, state or questions")
	}
	return request, nil
}

// Decode into a nullable string rather than a Go string: encoding/json accepts
// JSON null into a string destination without error.
func nullableText(d *json.Decoder) (*string, error) {
	var raw json.RawMessage
	if err := d.Decode(&raw); err != nil {
		return nil, err
	}
	if bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, err
	}
	return &value, nil
}

func requiredText(d *json.Decoder) (string, error) {
	value, err := nullableText(d)
	if err != nil {
		return "", err
	}
	if value == nil {
		return "", fmt.Errorf("expected text, got null")
	}
	return *value, nil
}

func parseTextOptions(d *json.Decoder) (bool, error) {
	if err := expectDelimiter(d, '{'); err != nil {
		return false, err
	}
	seen := map[string]bool{}
	var raw bool
	for d.More() {
		key, err := readKey(d, seen)
		if err != nil {
			return false, err
		}
		if key != "raw_logits" {
			return false, fmt.Errorf("unknown option %q", key)
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return false, err
		}
		if !bytes.Equal(value, []byte("true")) && !bytes.Equal(value, []byte("false")) {
			return false, fmt.Errorf("raw_logits must be a boolean")
		}
		raw = bytes.Equal(value, []byte("true"))
	}
	return raw, expectDelimiter(d, '}')
}

func parseTextQuestions(d *json.Decoder) ([]TextStateQuestion, error) {
	if err := expectDelimiter(d, '{'); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	questions := make([]TextStateQuestion, 0)
	for d.More() {
		if len(questions) >= MaxQuestions {
			return nil, fmt.Errorf("too many questions")
		}
		id, err := readKey(d, seen)
		if err != nil {
			return nil, err
		}
		if id == "" || len(id) > 128 {
			return nil, fmt.Errorf("invalid question ID")
		}
		q, err := parseTextQuestion(d, id)
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", id, err)
		}
		questions = append(questions, q)
	}
	if err := expectDelimiter(d, '}'); err != nil {
		return nil, err
	}
	return questions, nil
}

func parseTextQuestion(d *json.Decoder, id string) (TextStateQuestion, error) {
	var empty TextStateQuestion
	if err := expectDelimiter(d, '{'); err != nil {
		return empty, err
	}
	q := TextStateQuestion{ID: id}
	seen := map[string]bool{}
	var criteria json.RawMessage
	for d.More() {
		key, err := readKey(d, seen)
		if err != nil {
			return empty, err
		}
		switch key {
		case "type":
			q.Type, err = requiredText(d)
		case "instructions":
			q.Instructions, err = nullableText(d)
		case "criteria":
			err = d.Decode(&criteria)
		default:
			return empty, fmt.Errorf("unknown question field %q", key)
		}
		if err != nil {
			return empty, err
		}
	}
	if err := expectDelimiter(d, '}'); err != nil {
		return empty, err
	}
	if !seen["type"] || !seen["instructions"] {
		return empty, fmt.Errorf("missing type or instructions")
	}
	if q.Instructions != nil && len(*q.Instructions) > MaxInstructionBytes {
		return empty, fmt.Errorf("instructions too long")
	}
	if q.Type != "noul" && !seen["criteria"] {
		return empty, fmt.Errorf("missing criteria")
	}
	if bytes.Equal(criteria, []byte("null")) && seen["criteria"] && q.Type != "noul" {
		return empty, fmt.Errorf("null criteria")
	}
	if len(criteria) != 0 && !bytes.Equal(criteria, []byte("null")) {
		branch := json.NewDecoder(bytes.NewReader(criteria))
		var err error
		switch q.Type {
		case "choice", "noul":
			criteria, parseErr := parseTextCriteria(branch, q.Type == "noul")
			err = parseErr
			if q.Type == "noul" {
				q.NoulCriteria = criteria
			} else {
				q.Choices = criteria
			}
		case "score":
			q.Levels, err = parseTextLevels(branch)
		default:
			return empty, fmt.Errorf("unsupported question type %q", q.Type)
		}
		if err != nil {
			return empty, err
		}
	} else if q.Type != "noul" {
		return empty, fmt.Errorf("missing criteria")
	}
	if q.Type != "choice" && q.Type != "score" && q.Type != "noul" {
		return empty, fmt.Errorf("unsupported question type %q", q.Type)
	}
	if q.Type == "choice" && len(q.Choices) < 2 || q.Type == "score" && len(q.Levels) < 2 {
		return empty, fmt.Errorf("too few criteria")
	}
	return q, nil
}

func parseTextCriteria(d *json.Decoder, noul bool) ([]TextCriterion, error) {
	if err := expectDelimiter(d, '{'); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	criteria := make([]TextCriterion, 0)
	for d.More() {
		if len(criteria) >= 50 {
			return nil, fmt.Errorf("too many criteria")
		}
		key, err := readKey(d, seen)
		if err != nil {
			return nil, err
		}
		if noul && key != "true" && key != "false" {
			return nil, fmt.Errorf("invalid Noul criterion %q", key)
		}
		text, err := nullableText(d)
		if err != nil {
			return nil, err
		}
		if text != nil && len(*text) > 256 {
			return nil, fmt.Errorf("criterion text too long")
		}
		criteria = append(criteria, TextCriterion{ID: key, Description: text})
	}
	if err := expectDelimiter(d, '}'); err != nil {
		return nil, err
	}
	return criteria, nil
}

func parseTextLevels(d *json.Decoder) ([]*string, error) {
	if err := expectDelimiter(d, '['); err != nil {
		return nil, err
	}
	levels := make([]*string, 0)
	for d.More() {
		if len(levels) >= 50 {
			return nil, fmt.Errorf("too many levels")
		}
		text, err := nullableText(d)
		if err != nil {
			return nil, err
		}
		if text != nil && len(*text) > 256 {
			return nil, fmt.Errorf("level text too long")
		}
		levels = append(levels, text)
	}
	if err := expectDelimiter(d, ']'); err != nil {
		return nil, err
	}
	return levels, nil
}
