package mojev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// DecodeSystemOneTextRequest accepts the same text questions as DecodeTextRequest,
// and additionally accepts object, array and null states. Structured states use
// GSO's compact-JSON rendering: member order, number spelling and string escapes
// are preserved. Text states are unchanged. It is not full GSO/SDK compatibility:
// criteria/instructions remain text-only, and existing request bounds still apply.
func DecodeSystemOneTextRequest(r io.Reader) (TextRequest, error) {
	return decodeTextRequest(r, true)
}

func readSystemOneState(d *json.Decoder) (text string, structured bool, err error) {
	var raw json.RawMessage
	if err = d.Decode(&raw); err != nil {
		return
	}
	raw = bytes.TrimSpace(raw)
	if len(raw) > 0 && raw[0] == '"' {
		err = json.Unmarshal(raw, &text)
		return
	}
	if err = validateStructuredState(raw, nil); err != nil {
		return "", false, err
	}
	var compact bytes.Buffer
	if err = json.Compact(&compact, raw); err != nil {
		return "", false, err
	}
	return compact.String(), true, nil
}

// validateStructuredState bounds recursion and rejects duplicate members before
// rendering. visit sees decoded keys/strings, so escaped reserved tokens cannot
// bypass the tokenizer's normal user-text checks.
func validateStructuredState(raw []byte, visit func(string) error) error {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || len(raw) > 64<<10 || (raw[0] != '{' && raw[0] != '[' && !bytes.Equal(raw, []byte("null"))) {
		return fmt.Errorf("state must be text, object, array or null")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := walkStateValue(d, 0, visit); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("state must contain one JSON value")
	}
	return nil
}

func walkStateValue(d *json.Decoder, depth int, visit func(string) error) error {
	tok, err := d.Token()
	if err != nil {
		return err
	}
	switch v := tok.(type) {
	case string:
		if visit != nil {
			return visit(v)
		}
	case json.Delim:
		if depth >= 32 {
			return fmt.Errorf("structured state exceeds depth32")
		}
		var close json.Delim
		switch v {
		case '{':
			close = '}'
			seen := make(map[string]bool)
			for d.More() {
				key, err := d.Token()
				if err != nil {
					return err
				}
				s, ok := key.(string)
				if !ok || seen[s] {
					return fmt.Errorf("duplicate or invalid structured state key")
				}
				seen[s] = true
				if visit != nil {
					if err := visit(s); err != nil {
						return err
					}
				}
				if err := walkStateValue(d, depth+1, visit); err != nil {
					return err
				}
			}
		case '[':
			close = ']'
			for d.More() {
				if err := walkStateValue(d, depth+1, visit); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("invalid structured state delimiter")
		}
		if end, err := d.Token(); err != nil || end != close {
			return fmt.Errorf("invalid structured state end")
		}
	case nil, bool, json.Number:
		// Nested values retain the original JSON lexical form in rendering.
	default:
		return fmt.Errorf("invalid structured state value")
	}
	return nil
}
