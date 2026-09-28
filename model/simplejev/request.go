package simplejev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"unicode"
)

// Request is a model-free, independently authored envelope. It is not the
// upstream Simple-JEV wire format and contains no prompt template.
type Request struct {
	State     string     `json:"state"`
	Questions []Question `json:"questions"`
}

type Question struct {
	ID          string   `json:"id"`
	Instruction string   `json:"instruction"`
	Kind        string   `json:"kind"`
	Labels      []string `json:"labels"`
}

const MaxRequestBytes = 1 << 20
const MaxQuestions = 256
const MaxStateBytes = 64 << 10
const MaxInstructionBytes = 8 << 10

// DecodeRequest refuses trailing data, unknown fields and inputs over the
// caller-independent byte limit. It validates the complete envelope before
// returning it; no model or network operation occurs here.
func DecodeRequest(reader io.Reader) (Request, error) {
	var request Request
	if reader == nil {
		return request, fmt.Errorf("simplejev: nil request reader")
	}
	limited := &io.LimitedReader{R: reader, N: MaxRequestBytes + 1}
	data, err := io.ReadAll(limited)
	if err != nil {
		return request, err
	}
	if len(data) > MaxRequestBytes {
		return request, fmt.Errorf("simplejev: request exceeds %d bytes", MaxRequestBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err := parseRequest(decoder, &request); err != nil {
		return Request{}, fmt.Errorf("simplejev: decode request: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return Request{}, fmt.Errorf("simplejev: request contains trailing data")
	}
	if err := request.Validate(); err != nil {
		return Request{}, err
	}
	return request, nil
}

// Parse object tokens explicitly: encoding/json's struct decoder accepts
// duplicate and case-variant members, which this bounded schema forbids.
func parseRequest(d *json.Decoder, request *Request) error {
	if err := expectDelimiter(d, '{'); err != nil {
		return err
	}
	seen := map[string]bool{}
	for d.More() {
		key, err := readKey(d, seen)
		if err != nil {
			return err
		}
		switch key {
		case "state":
			err = d.Decode(&request.State)
		case "questions":
			err = parseQuestions(d, &request.Questions)
		default:
			return fmt.Errorf("unknown field %q", key)
		}
		if err != nil {
			return err
		}
	}
	return expectDelimiter(d, '}')
}

func parseQuestions(d *json.Decoder, questions *[]Question) error {
	if err := expectDelimiter(d, '['); err != nil {
		return err
	}
	for d.More() {
		if len(*questions) >= MaxQuestions {
			return fmt.Errorf("too many questions")
		}
		var q Question
		if err := expectDelimiter(d, '{'); err != nil {
			return err
		}
		seen := map[string]bool{}
		for d.More() {
			key, err := readKey(d, seen)
			if err != nil {
				return err
			}
			switch key {
			case "id":
				err = d.Decode(&q.ID)
			case "instruction":
				err = d.Decode(&q.Instruction)
			case "kind":
				err = d.Decode(&q.Kind)
			case "labels":
				err = d.Decode(&q.Labels)
			default:
				return fmt.Errorf("unknown question field %q", key)
			}
			if err != nil {
				return err
			}
		}
		if err := expectDelimiter(d, '}'); err != nil {
			return err
		}
		*questions = append(*questions, q)
	}
	return expectDelimiter(d, ']')
}

func readKey(d *json.Decoder, seen map[string]bool) (string, error) {
	token, err := d.Token()
	if err != nil {
		return "", err
	}
	key, ok := token.(string)
	if !ok || seen[key] {
		return "", fmt.Errorf("duplicate or invalid JSON member %q", token)
	}
	seen[key] = true
	return key, nil
}

func expectDelimiter(d *json.Decoder, want json.Delim) error {
	token, err := d.Token()
	if err != nil {
		return err
	}
	if token != want {
		return fmt.Errorf("expected JSON delimiter %q", want)
	}
	return nil
}

// All string ceilings are decoded UTF-8 byte limits. Format/control-only text
// is blank even if unicode.TrimSpace would leave zero-width code points.
func visibleText(text string) bool {
	for _, r := range text {
		if !unicode.IsSpace(r) && !unicode.IsControl(r) && !unicode.Is(unicode.Cf, r) {
			return true
		}
	}
	return false
}

func (r Request) Validate() error {
	if !visibleText(r.State) || len(r.State) > MaxStateBytes || len(r.Questions) == 0 || len(r.Questions) > MaxQuestions {
		return fmt.Errorf("simplejev: invalid state or question count")
	}
	ids := make(map[string]struct{}, len(r.Questions))
	for _, q := range r.Questions {
		if !visibleText(q.ID) || len(q.ID) > 128 || !visibleText(q.Instruction) || len(q.Instruction) > MaxInstructionBytes || (q.Kind != "choice" && q.Kind != "ordinal") || len(q.Labels) < 2 || len(q.Labels) > 50 {
			return fmt.Errorf("simplejev: invalid question %q", q.ID)
		}
		for _, char := range q.ID {
			if unicode.IsSpace(char) || unicode.IsControl(char) || unicode.Is(unicode.Cf, char) {
				return fmt.Errorf("simplejev: invalid question ID %q", q.ID)
			}
		}
		if _, exists := ids[q.ID]; exists {
			return fmt.Errorf("simplejev: duplicate question ID %q", q.ID)
		}
		ids[q.ID] = struct{}{}
		labels := make(map[string]struct{}, len(q.Labels))
		for _, label := range q.Labels {
			if !visibleText(label) || len(label) > 256 {
				return fmt.Errorf("simplejev: invalid public label in %q", q.ID)
			}
			if _, exists := labels[label]; exists {
				return fmt.Errorf("simplejev: duplicate public label in %q", q.ID)
			}
			labels[label] = struct{}{}
		}
	}
	return nil
}
