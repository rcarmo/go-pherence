package simplejev

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

const MaxStructuredStateBytes = 64 << 10
const MaxStructuredStateDepth = 32

// CanonicalStructuredState renders a bounded JSON state using sorted object
// keys and readable UTF-8. It accepts strings, objects and arrays at the root,
// with signed 64-bit integer numbers only. Decimal/exponent numbers, duplicate
// keys, U+FFFD, U+2028, U+2029, invalid UTF-8 and excessive nesting fail before
// output. These restrictions avoid Python/Go JSON representation differences.
// This does not render an upstream prompt or broaden DecodeTextStateRequest.
func CanonicalStructuredState(reader io.Reader) (string, error) {
	if reader == nil {
		return "", fmt.Errorf("simplejev: nil structured state")
	}
	data, err := io.ReadAll(&io.LimitedReader{R: reader, N: MaxStructuredStateBytes + 1})
	if err != nil {
		return "", err
	}
	if len(data) > MaxStructuredStateBytes || !utf8.Valid(data) {
		return "", fmt.Errorf("simplejev: invalid structured-state bytes")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	value, err := parseStructuredValue(d, 0)
	if err != nil {
		return "", err
	}
	if _, err := d.Token(); err != io.EOF {
		return "", fmt.Errorf("simplejev: trailing structured-state data")
	}
	switch value.(type) {
	case string, map[string]any, []any:
	default:
		return "", fmt.Errorf("simplejev: structured state must be text, object or array")
	}
	var out bytes.Buffer
	if err := writeStructuredValue(&out, value); err != nil {
		return "", err
	}
	return out.String(), nil
}

func parseStructuredValue(d *json.Decoder, depth int) (any, error) {
	if depth > MaxStructuredStateDepth {
		return nil, fmt.Errorf("simplejev: structured state too deep")
	}
	tok, err := d.Token()
	if err != nil {
		return nil, err
	}
	switch tok {
	case json.Delim('{'):
		object := make(map[string]any)
		for d.More() {
			keyToken, err := d.Token()
			if err != nil {
				return nil, err
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, fmt.Errorf("simplejev: invalid structured-state key")
			}
			if err := validateStructuredString(key); err != nil {
				return nil, err
			}
			if _, exists := object[key]; exists {
				return nil, fmt.Errorf("simplejev: duplicate structured-state key")
			}
			value, err := parseStructuredValue(d, depth+1)
			if err != nil {
				return nil, err
			}
			object[key] = value
		}
		if end, err := d.Token(); err != nil || end != json.Delim('}') {
			return nil, fmt.Errorf("simplejev: unterminated structured-state object")
		}
		return object, nil
	case json.Delim('['):
		array := make([]any, 0)
		for d.More() {
			value, err := parseStructuredValue(d, depth+1)
			if err != nil {
				return nil, err
			}
			array = append(array, value)
		}
		if end, err := d.Token(); err != nil || end != json.Delim(']') {
			return nil, fmt.Errorf("simplejev: unterminated structured-state array")
		}
		return array, nil
	}
	switch value := tok.(type) {
	case string:
		if err := validateStructuredString(value); err != nil {
			return nil, err
		}
		return value, nil
	case bool, nil:
		return value, nil
	case json.Number:
		if strings.ContainsAny(string(value), ".eE") {
			return nil, fmt.Errorf("simplejev: non-integer structured-state number")
		}
		integer, err := strconv.ParseInt(string(value), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("simplejev: structured-state integer out of range")
		}
		return integer, nil
	default:
		return nil, fmt.Errorf("simplejev: invalid structured-state value")
	}
}

func validateStructuredString(value string) error {
	if strings.ContainsAny(value, "\ufffd\u2028\u2029") {
		return fmt.Errorf("simplejev: unsupported structured-state character")
	}
	return nil
}

func writeStructuredValue(out *bytes.Buffer, value any) error {
	switch v := value.(type) {
	case map[string]any:
		out.WriteByte('{')
		keys := make([]string, 0, len(v))
		for key := range v {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for i, key := range keys {
			if i > 0 {
				out.WriteByte(',')
			}
			if err := writeStructuredString(out, key); err != nil {
				return err
			}
			out.WriteByte(':')
			if err := writeStructuredValue(out, v[key]); err != nil {
				return err
			}
		}
		out.WriteByte('}')
	case []any:
		out.WriteByte('[')
		for i, item := range v {
			if i > 0 {
				out.WriteByte(',')
			}
			if err := writeStructuredValue(out, item); err != nil {
				return err
			}
		}
		out.WriteByte(']')
	case string:
		return writeStructuredString(out, v)
	case bool:
		if v {
			out.WriteString("true")
		} else {
			out.WriteString("false")
		}
	case nil:
		out.WriteString("null")
	case int64:
		out.WriteString(strconv.FormatInt(v, 10))
	default:
		return fmt.Errorf("simplejev: invalid structured-state output")
	}
	return nil
}

func writeStructuredString(out *bytes.Buffer, value string) error {
	// Go's default encoder escapes HTML characters; the independent oracle
	// keeps them literal. The input validator already checked this string.
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(value); err != nil {
		return err
	}
	_, err := out.Write(bytes.TrimSuffix(b.Bytes(), []byte{'\n'}))
	return err
}
