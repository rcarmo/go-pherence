package mojev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// TextRequest is a bounded, text-state subset of MoJev's public request.
// Field order and caller-order keys are retained for prompt and answer assembly.
type TextRequest struct {
	Model, State string
	Fields       []TextRequestField
}
type TextRequestField struct {
	ID, Kind, Instructions       string
	Keys, Options, SortedOptions []string
	SortedIndices                []int
}

const maxTextRequestBytes = 1 << 20

// DecodeTextRequest validates a text-only subset. It does not invoke a model,
// accept image/structured state, or implement HTTP/SDK request validation.
func DecodeTextRequest(reader io.Reader) (TextRequest, error) {
	var zero TextRequest
	if reader == nil {
		return zero, fmt.Errorf("mojev: nil reader")
	}
	data, err := io.ReadAll(&io.LimitedReader{R: reader, N: maxTextRequestBytes + 1})
	if err != nil {
		return zero, err
	}
	if len(data) > maxTextRequestBytes || !utf8.Valid(data) {
		return zero, fmt.Errorf("mojev: invalid request bytes")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	if err = objectStart(d); err != nil {
		return zero, err
	}
	var out TextRequest
	seen := make(map[string]bool)
	for d.More() {
		key, err := uniqueKey(d, seen)
		if err != nil {
			return zero, err
		}
		switch key {
		case "model":
			out.Model, err = textValue(d)
		case "state":
			out.State, err = textValue(d)
		case "questions":
			out.Fields, err = readTextQuestions(d)
		default:
			return zero, fmt.Errorf("mojev: unsupported request field %q", key)
		}
		if err != nil {
			return zero, fmt.Errorf("mojev: %s: %w", key, err)
		}
	}
	if err = objectEnd(d); err != nil {
		return zero, err
	}
	var extra any
	if err = d.Decode(&extra); err != io.EOF {
		return zero, fmt.Errorf("mojev: trailing data")
	}
	if !seen["model"] || out.Model == "" || !seen["state"] || out.State == "" || len(out.State) > 64<<10 || strings.Contains(out.State, "<|image_pad|>") || !seen["questions"] || len(out.Fields) == 0 {
		return zero, fmt.Errorf("mojev: missing or invalid text model, state or questions")
	}
	return out, nil
}

func objectStart(d *json.Decoder) error {
	tok, err := d.Token()
	if err != nil {
		return err
	}
	if tok != json.Delim('{') {
		return fmt.Errorf("mojev: expected object")
	}
	return nil
}
func objectEnd(d *json.Decoder) error {
	tok, err := d.Token()
	if err != nil {
		return err
	}
	if tok != json.Delim('}') {
		return fmt.Errorf("mojev: expected object end")
	}
	return nil
}
func uniqueKey(d *json.Decoder, seen map[string]bool) (string, error) {
	tok, err := d.Token()
	if err != nil {
		return "", err
	}
	key, ok := tok.(string)
	if !ok || key == "" || seen[key] {
		return "", fmt.Errorf("mojev: empty or duplicate key")
	}
	seen[key] = true
	return key, nil
}
func textValue(d *json.Decoder) (string, error) {
	var raw json.RawMessage
	if err := d.Decode(&raw); err != nil {
		return "", err
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil || bytes.Equal(raw, []byte("null")) {
		return "", fmt.Errorf("mojev: expected text")
	}
	return value, nil
}
func displayValue(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || raw[0] != '"' {
		return "", fmt.Errorf("mojev: text-only value required")
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", err
	}
	return s, nil
}

func readTextQuestions(d *json.Decoder) ([]TextRequestField, error) {
	if err := objectStart(d); err != nil {
		return nil, err
	}
	seen := make(map[string]bool)
	out := make([]TextRequestField, 0)
	for d.More() {
		if len(out) >= 256 {
			return nil, fmt.Errorf("mojev: too many questions")
		}
		id, err := uniqueKey(d, seen)
		if err != nil {
			return nil, err
		}
		if len(id) > 128 {
			return nil, fmt.Errorf("mojev: question ID too long")
		}
		f, err := readTextQuestion(d, id)
		if err != nil {
			return nil, fmt.Errorf("mojev: question %q: %w", id, err)
		}
		out = append(out, f)
	}
	if err := objectEnd(d); err != nil {
		return nil, err
	}
	return out, nil
}
func readTextQuestion(d *json.Decoder, id string) (TextRequestField, error) {
	var zero TextRequestField
	if err := objectStart(d); err != nil {
		return zero, err
	}
	seen := make(map[string]bool)
	out := TextRequestField{ID: id}
	var criteria json.RawMessage
	for d.More() {
		key, err := uniqueKey(d, seen)
		if err != nil {
			return zero, err
		}
		switch key {
		case "type":
			out.Kind, err = textValue(d)
		case "instructions":
			var raw json.RawMessage
			err = d.Decode(&raw)
			if err == nil && !bytes.Equal(raw, []byte("null")) {
				out.Instructions, err = displayValue(raw)
			}
		case "criteria":
			err = d.Decode(&criteria)
		default:
			return zero, fmt.Errorf("mojev: unsupported question field %q", key)
		}
		if err != nil {
			return zero, err
		}
	}
	if err := objectEnd(d); err != nil {
		return zero, err
	}
	if !seen["type"] || len(out.Instructions) > 8<<10 || strings.Contains(out.Instructions, "<|image_pad|>") {
		return zero, fmt.Errorf("mojev: invalid type or text instructions")
	}
	var err error
	switch out.Kind {
	case "choice":
		out.Keys, out.Options, err = choiceCriteria(criteria)
	case "noul":
		out.Keys, out.Options, err = noulCriteria(criteria)
	case "score":
		out.Keys, out.Options, err = scoreCriteria(criteria)
	default:
		return zero, fmt.Errorf("mojev: unsupported question type")
	}
	if err != nil {
		return zero, err
	}
	if len(out.Options) < 2 || len(out.Options) > 64 {
		return zero, fmt.Errorf("mojev: invalid option count")
	}
	seenOptions := make(map[string]bool, len(out.Options))
	for _, option := range out.Options {
		if option == "" || len(option) > 8<<10 || strings.Contains(option, "<|image_pad|>") || seenOptions[option] {
			return zero, fmt.Errorf("mojev: invalid or duplicate text candidate")
		}
		seenOptions[option] = true
	}
	order := make([]int, len(out.Options))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return out.Options[order[i]] < out.Options[order[j]] })
	out.SortedIndices = order
	out.SortedOptions = make([]string, len(order))
	for i, n := range order {
		out.SortedOptions[i] = out.Options[n]
	}
	return out, nil
}
func choiceCriteria(raw json.RawMessage) ([]string, []string, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	if err := objectStart(d); err != nil {
		return nil, nil, err
	}
	seen := make(map[string]bool)
	keys, options := []string{}, []string{}
	for d.More() {
		if len(keys) >= 64 {
			return nil, nil, fmt.Errorf("mojev: too many choices")
		}
		key, err := uniqueKey(d, seen)
		if err != nil {
			return nil, nil, err
		}
		var value json.RawMessage
		if err = d.Decode(&value); err != nil {
			return nil, nil, err
		}
		text := key
		if !bytes.Equal(value, []byte("null")) {
			v, err := displayValue(value)
			if err != nil {
				return nil, nil, err
			}
			if v != "" {
				text = key + ": " + v
			}
		}
		keys = append(keys, key)
		options = append(options, text)
	}
	if err := objectEnd(d); err != nil {
		return nil, nil, err
	}
	return keys, options, nil
}
func noulCriteria(raw json.RawMessage) ([]string, []string, error) {
	keys := []string{"false", "true"}
	options := []string{"no", "yes"}
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return keys, options, nil
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	if err := objectStart(d); err != nil {
		return nil, nil, err
	}
	seen := make(map[string]bool)
	for d.More() {
		key, err := uniqueKey(d, seen)
		if err != nil {
			return nil, nil, err
		}
		var value json.RawMessage
		if err = d.Decode(&value); err != nil {
			return nil, nil, err
		}
		if key != "false" && key != "true" {
			continue
		}
		if bytes.Equal(value, []byte("null")) {
			continue
		}
		text, err := displayValue(value)
		if err != nil {
			return nil, nil, err
		}
		if text != "" {
			if key == "false" {
				options[0] = text
			} else {
				options[1] = text
			}
		}
	}
	if err := objectEnd(d); err != nil {
		return nil, nil, err
	}
	return keys, options, nil
}
func scoreCriteria(raw json.RawMessage) ([]string, []string, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	tok, err := d.Token()
	if err != nil {
		return nil, nil, err
	}
	if tok != json.Delim('[') {
		return nil, nil, fmt.Errorf("mojev: score needs list")
	}
	keys, options := []string{}, []string{}
	for d.More() {
		if len(keys) >= 64 {
			return nil, nil, fmt.Errorf("mojev: too many levels")
		}
		var value json.RawMessage
		if err = d.Decode(&value); err != nil {
			return nil, nil, err
		}
		text, err := displayValue(value)
		if err != nil {
			return nil, nil, err
		}
		keys = append(keys, strconv.Itoa(len(keys)))
		options = append(options, text)
	}
	tok, err = d.Token()
	if err != nil || tok != json.Delim(']') {
		return nil, nil, fmt.Errorf("mojev: invalid score list")
	}
	return keys, options, nil
}
