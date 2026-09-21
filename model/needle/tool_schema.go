package needle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"regexp/syntax"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	toolGrammarMaxCalls      = 4
	toolGrammarMaxTools      = 16
	toolSchemaMaxBytes       = 64 << 10
	toolSchemaTotalMaxBytes  = 128 << 10
	toolGrammarRegexMaxBytes = 256 << 10
	toolGrammarProgMaxInst   = 50000
	toolSchemaMaxDepth       = 8
	toolSchemaMaxProps       = 32
	toolSchemaMaxArrayItems  = 8
)

const (
	toolSchemaWSPattern     = `[\x20\x09\x0A\x0D]*`
	toolSchemaIntPattern    = `-?(?:\x30|[\x31-\x39][\x30-\x39]*)`
	toolSchemaNumPattern    = `-?(?:\x30|[\x31-\x39][\x30-\x39]*)(?:\.[\x30-\x39]+)?(?:[\x45\x65][\x2B\x2D]?[\x30-\x39]+)?`
	toolSchemaStringPattern = `\x22(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|\x5C(?:[\x22\x5C\x2F\x62\x66\x6E\x72\x74]|\x75(?:[0-9A-Ca-cE-Fe-f][0-9A-Fa-f]{3}|[Dd][0-7][0-9A-Fa-f]{2}|[Dd][89AaBb][0-9A-Fa-f]{2}\x5C\x75[Dd][C-Fc-f][0-9A-Fa-f]{2}))|(?:[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|[\xEE-\xEF][\x80-\xBF]{2}|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}))*\x22`
)

// ToolSchema declares one admissible tool call shape.
type ToolSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolGrammar is an immutable regular-language recognizer for constrained tool calls.
type ToolGrammar struct {
	prog  *syntax.Prog
	start []uint32
}

// ToolState is one immutable prefix-recognition state.
type ToolState struct {
	g   *ToolGrammar
	pcs []uint32
}

// CompileToolGrammar compiles tool schemas into a bounded JSON grammar.
func CompileToolGrammar(tools []ToolSchema, maxCalls int) (*ToolGrammar, error) {
	if maxCalls < 1 || maxCalls > toolGrammarMaxCalls {
		return nil, fmt.Errorf("needle: maxCalls must be 1..%d", toolGrammarMaxCalls)
	}
	if len(tools) > toolGrammarMaxTools {
		return nil, fmt.Errorf("needle: at most %d tools are supported", toolGrammarMaxTools)
	}
	var total int
	seenNames := make(map[string]struct{}, len(tools))
	toolPatterns := make([]string, 0, len(tools))
	for _, tool := range tools {
		if len(tool.Description) > 4096 {
			return nil, fmt.Errorf("needle: tool description exceeds 4096 bytes")
		}
		if !utf8.ValidString(tool.Description) || !utf8.Valid(tool.Parameters) {
			return nil, fmt.Errorf("needle: schema is not valid UTF-8")
		}
		if !validToolName(tool.Name) {
			return nil, fmt.Errorf("needle: tool name %q must match [A-Za-z_][A-Za-z0-9_.-]*", tool.Name)
		}
		if _, ok := seenNames[tool.Name]; ok {
			return nil, fmt.Errorf("needle: duplicate tool name %q", tool.Name)
		}
		seenNames[tool.Name] = struct{}{}
		total += len(tool.Description) + len(tool.Name)
		if len(tool.Parameters) > toolSchemaMaxBytes {
			return nil, fmt.Errorf("needle: tool %q parameters exceed %d bytes", tool.Name, toolSchemaMaxBytes)
		}
		total += len(tool.Parameters)
		if total > toolSchemaTotalMaxBytes {
			return nil, fmt.Errorf("needle: tool schemas exceed total %d bytes", toolSchemaTotalMaxBytes)
		}
		params, err := decodeJSONValue(tool.Parameters)
		if err != nil {
			return nil, fmt.Errorf("needle: tool %q parameters: %w", tool.Name, err)
		}
		path := fmt.Sprintf("tool %q parameters", tool.Name)
		pat, kind, err := compileSchemaPattern(params, path, 1)
		if err != nil {
			return nil, err
		}
		if kind != "object" {
			return nil, fmt.Errorf("needle: %s must be an object schema", path)
		}
		callPat, err := buildToolCallPattern(tool.Name, pat)
		if err != nil {
			return nil, err
		}
		toolPatterns = append(toolPatterns, callPat)
	}
	pattern, err := buildTopLevelPattern(toolPatterns, maxCalls)
	if err != nil {
		return nil, err
	}
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil, fmt.Errorf("needle: internal regex parse failed: %w", err)
	}
	if regexExpandedCost(re, toolGrammarProgMaxInst) > toolGrammarProgMaxInst {
		return nil, fmt.Errorf("needle: grammar program exceeds %d instructions", toolGrammarProgMaxInst)
	}
	prog, err := syntax.Compile(re.Simplify())
	if err != nil {
		return nil, fmt.Errorf("needle: internal regex compile failed: %w", err)
	}
	if len(prog.Inst) > toolGrammarProgMaxInst {
		return nil, fmt.Errorf("needle: compiled grammar exceeds %d instructions", toolGrammarProgMaxInst)
	}
	for _, inst := range prog.Inst {
		if inst.Op == syntax.InstEmptyWidth {
			return nil, fmt.Errorf("needle: grammar assertions unsupported")
		}
	}
	g := &ToolGrammar{prog: prog}
	w := g.newMatcher()
	g.start = w.closure([]uint32{uint32(prog.Start)}, nil)
	return g, nil
}

// Start returns a fresh prefix-recognition state.
func (g *ToolGrammar) Start() *ToolState {
	if g == nil {
		return &ToolState{}
	}
	return &ToolState{g: g, pcs: append([]uint32(nil), g.start...)}
}

// Advance returns the next owned state if fragment keeps at least one valid continuation.
func (s *ToolState) Advance(fragment []byte) (*ToolState, bool) {
	if s == nil || s.g == nil || s.g.prog == nil {
		return nil, false
	}
	return s.g.newMatcher().advance(s, fragment)
}

// Complete reports whether the current state is a complete valid tool-call JSON value.
func (s *ToolState) Complete() bool {
	if s == nil || s.g == nil || s.g.prog == nil {
		return false
	}
	for _, pc := range s.pcs {
		if s.g.prog.Inst[pc].Op == syntax.InstMatch {
			return true
		}
	}
	return false
}

func buildTopLevelPattern(toolPatterns []string, maxCalls int) (string, error) {
	emptyArray := `\[` + toolSchemaWSPattern + `\]`
	body := emptyArray
	if len(toolPatterns) > 0 {
		callAlt := alternate(toolPatterns)
		nonEmpty := `\[` + toolSchemaWSPattern + callAlt
		if maxCalls > 1 {
			nonEmpty += `(?:` + toolSchemaWSPattern + `,` + toolSchemaWSPattern + callAlt + `){0,` + strconv.Itoa(maxCalls-1) + `}`
		}
		nonEmpty += toolSchemaWSPattern + `\]`
		body = alternate([]string{emptyArray, nonEmpty})
	}
	pattern := toolSchemaWSPattern + body + toolSchemaWSPattern
	if len(pattern) > toolGrammarRegexMaxBytes {
		return "", fmt.Errorf("needle: grammar regex exceeds %d bytes", toolGrammarRegexMaxBytes)
	}
	return pattern, nil
}

func buildToolCallPattern(name, params string) (string, error) {
	nameKey, _ := json.Marshal("name")
	argsKey, _ := json.Marshal("arguments")
	nameValue, _ := json.Marshal(name)
	parts := []string{
		`\{`, toolSchemaWSPattern,
		regexQuoteBytes(nameKey), toolSchemaWSPattern, `:`, toolSchemaWSPattern, regexQuoteBytes(nameValue),
		toolSchemaWSPattern, `,`, toolSchemaWSPattern,
		regexQuoteBytes(argsKey), toolSchemaWSPattern, `:`, toolSchemaWSPattern, group(params),
		toolSchemaWSPattern, `\}`,
	}
	pat, err := concatPattern(parts...)
	if err != nil {
		return "", err
	}
	return pat, nil
}

func compileSchemaPattern(node any, path string, depth int) (string, string, error) {
	if depth > toolSchemaMaxDepth {
		return "", "", fmt.Errorf("needle: %s exceeds maximum schema depth %d", path, toolSchemaMaxDepth)
	}
	m, ok := node.(map[string]any)
	if !ok {
		return "", "", fmt.Errorf("needle: %s must be a schema object", path)
	}
	if err := rejectUnsupportedSchemaKeys(m, path); err != nil {
		return "", "", err
	}
	kind, err := schemaKind(m, path)
	if err != nil {
		return "", "", err
	}
	switch kind {
	case "object":
		pat, err := compileObjectSchema(m, path, depth)
		return pat, kind, err
	case "array":
		pat, err := compileArraySchema(m, path, depth)
		return pat, kind, err
	case "string", "integer", "number", "boolean", "null":
		pat, err := compileScalarSchema(m, path, kind)
		return pat, kind, err
	default:
		return "", "", fmt.Errorf("needle: %s has unsupported type %q", path, kind)
	}
}

func rejectUnsupportedSchemaKeys(m map[string]any, path string) error {
	for _, key := range []string{"title", "description"} {
		if raw, exists := m[key]; exists {
			if _, ok := raw.(string); !ok {
				return fmt.Errorf("needle: %s.%s must be a string", path, key)
			}
		}
	}
	for key := range m {
		switch key {
		case "type", "properties", "required", "additionalProperties", "items", "maxItems", "minItems", "enum", "const", "description", "title":
			continue
		default:
			return fmt.Errorf("needle: %s has unsupported keyword %q", path, key)
		}
	}
	return nil
}

func schemaKind(m map[string]any, path string) (string, error) {
	var kind string
	if raw, ok := m["type"]; ok {
		s, ok := raw.(string)
		if !ok {
			return "", fmt.Errorf("needle: %s.type must be a string; union types are not supported", path)
		}
		kind = canonicalSchemaType(s)
		if kind == "" {
			return "", fmt.Errorf("needle: %s.type %q is unsupported", path, s)
		}
	}
	kwKind := ""
	if _, ok := m["properties"]; ok || hasKey(m, "required") || hasKey(m, "additionalProperties") {
		kwKind = "object"
	}
	if _, ok := m["items"]; ok || hasKey(m, "maxItems") || hasKey(m, "minItems") {
		if kwKind != "" && kwKind != "array" {
			return "", fmt.Errorf("needle: %s mixes object and array schema keywords", path)
		}
		kwKind = "array"
	}
	if kind == "" && kwKind != "" {
		kind = kwKind
	}
	if kind != "" && kwKind != "" && kind != kwKind {
		return "", fmt.Errorf("needle: %s.type %q conflicts with its schema keywords", path, kind)
	}
	if kind == "" {
		if v, ok := m["const"]; ok {
			k, err := inferScalarKind(v)
			if err != nil {
				return "", fmt.Errorf("needle: %s.const: %w", path, err)
			}
			kind = k
		} else if v, ok := m["enum"]; ok {
			vals, ok := v.([]any)
			if !ok {
				return "", fmt.Errorf("needle: %s.enum must be an array", path)
			}
			k, err := inferEnumKind(vals)
			if err != nil {
				return "", fmt.Errorf("needle: %s.enum: %w", path, err)
			}
			kind = k
		}
	}
	if kind == "" {
		return "", fmt.Errorf("needle: %s must declare a supported type", path)
	}
	return kind, nil
}

func compileObjectSchema(m map[string]any, path string, depth int) (string, error) {
	for _, key := range []string{"items", "maxItems", "minItems", "enum", "const"} {
		if hasKey(m, key) {
			return "", fmt.Errorf("needle: %s.%s is unsupported for object schemas", path, key)
		}
	}
	if v, ok := m["additionalProperties"]; ok {
		allow, ok := v.(bool)
		if !ok || allow {
			return "", fmt.Errorf("needle: %s.additionalProperties must be false or omitted", path)
		}
	}
	propsAny, ok := m["properties"]
	if !ok {
		propsAny = map[string]any{}
	}
	props, ok := propsAny.(map[string]any)
	if !ok {
		return "", fmt.Errorf("needle: %s.properties must be an object", path)
	}
	if len(props) > toolSchemaMaxProps {
		return "", fmt.Errorf("needle: %s has %d properties; maximum is %d", path, len(props), toolSchemaMaxProps)
	}
	if raw, exists := m["required"]; exists && raw == nil {
		return "", fmt.Errorf("needle: %s.required must be an array of property names", path)
	}
	_, err := parseRequired(m["required"], path, props)
	if err != nil {
		return "", err
	}
	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	sort.Strings(names)
	if len(names) == 0 {
		return `\{` + toolSchemaWSPattern + `\}`, nil
	}
	parts := []string{`\{`, toolSchemaWSPattern}
	for i, name := range names {
		if i > 0 {
			parts = append(parts, toolSchemaWSPattern, `,`, toolSchemaWSPattern)
		}
		quotedName, _ := json.Marshal(name)
		childPath := path + `.properties.` + name
		childPat, _, err := compileSchemaPattern(props[name], childPath, depth+1)
		if err != nil {
			return "", err
		}
		parts = append(parts, regexQuoteBytes(quotedName), toolSchemaWSPattern, `:`, toolSchemaWSPattern, group(childPat))
	}
	parts = append(parts, toolSchemaWSPattern, `\}`)
	return concatPattern(parts...)
}

func parseRequired(raw any, path string, props map[string]any) (map[string]struct{}, error) {
	if len(props) == 0 {
		if raw == nil {
			return map[string]struct{}{}, nil
		}
	}
	if raw == nil {
		if len(props) == 0 {
			return map[string]struct{}{}, nil
		}
		return nil, fmt.Errorf("needle: %s requires all properties to be listed in required; optional properties are not supported", path)
	}
	items, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("needle: %s.required must be an array of property names", path)
	}
	required := make(map[string]struct{}, len(items))
	for _, item := range items {
		name, ok := item.(string)
		if !ok {
			return nil, fmt.Errorf("needle: %s.required must contain only strings", path)
		}
		if _, ok := required[name]; ok {
			return nil, fmt.Errorf("needle: %s.required contains duplicate property %q", path, name)
		}
		required[name] = struct{}{}
		if _, ok := props[name]; !ok {
			return nil, fmt.Errorf("needle: %s.required includes unknown property %q", path, name)
		}
	}
	if len(required) != len(props) {
		missing := make([]string, 0, len(props)-len(required))
		for name := range props {
			if _, ok := required[name]; !ok {
				missing = append(missing, name)
			}
		}
		sort.Strings(missing)
		return nil, fmt.Errorf("needle: %s requires all properties to be listed in required; optional properties are not supported (missing %q)", path, missing[0])
	}
	return required, nil
}

func compileArraySchema(m map[string]any, path string, depth int) (string, error) {
	for _, key := range []string{"properties", "required", "additionalProperties", "enum", "const"} {
		if hasKey(m, key) {
			return "", fmt.Errorf("needle: %s.%s is unsupported for array schemas", path, key)
		}
	}
	items, ok := m["items"]
	if !ok {
		return "", fmt.Errorf("needle: %s.items is required for array schemas", path)
	}
	maxItems, err := parseBoundedInt(m["maxItems"], path+`.maxItems`, 1, toolSchemaMaxArrayItems, true)
	if err != nil {
		return "", err
	}
	if raw, exists := m["minItems"]; exists && raw == nil {
		return "", fmt.Errorf("needle: %s.minItems must be an integer", path)
	}
	minItems, err := parseBoundedInt(m["minItems"], path+`.minItems`, 0, maxItems, false)
	if err != nil {
		return "", err
	}
	itemPat, _, err := compileSchemaPattern(items, path+`.items`, depth+1)
	if err != nil {
		return "", err
	}
	elem := group(itemPat)
	tail := `(?:` + toolSchemaWSPattern + `,` + toolSchemaWSPattern + elem + `)`
	parts := []string{`\[`, toolSchemaWSPattern}
	switch {
	case minItems == 0 && maxItems == 1:
		parts = append(parts, `(?:`+elem+`)?`)
	case minItems == 0:
		parts = append(parts, `(?:`+elem)
		if maxItems > 1 {
			parts = append(parts, tail+`{0,`+strconv.Itoa(maxItems-1)+`}`)
		}
		parts = append(parts, `)?`)
	default:
		parts = append(parts, elem)
		if minItems > 1 {
			parts = append(parts, tail+`{`+strconv.Itoa(minItems-1)+`}`)
		}
		if maxItems > minItems {
			parts = append(parts, tail+`{0,`+strconv.Itoa(maxItems-minItems)+`}`)
		}
	}
	parts = append(parts, toolSchemaWSPattern, `\]`)
	return concatPattern(parts...)
}

func compileScalarSchema(m map[string]any, path, kind string) (string, error) {
	for _, key := range []string{"properties", "required", "additionalProperties", "items", "maxItems", "minItems"} {
		if hasKey(m, key) {
			return "", fmt.Errorf("needle: %s.%s is unsupported for %s schemas", path, key, kind)
		}
	}
	if v, ok := m["const"]; ok {
		if enum, ok := m["enum"]; ok {
			vals, ok := enum.([]any)
			if !ok {
				return "", fmt.Errorf("needle: %s.enum must be an array", path)
			}
			lit, err := canonicalScalarLiteral(v, kind)
			if err != nil {
				return "", fmt.Errorf("needle: %s.const: %w", path, err)
			}
			found := false
			for _, item := range vals {
				cand, err := canonicalScalarLiteral(item, kind)
				if err != nil {
					return "", fmt.Errorf("needle: %s.enum: %w", path, err)
				}
				if cand == lit {
					found = true
				}
			}
			if !found {
				return "", fmt.Errorf("needle: %s.const is not present in enum", path)
			}
			return regexQuoteBytes([]byte(lit)), nil
		}
		lit, err := canonicalScalarLiteral(v, kind)
		if err != nil {
			return "", fmt.Errorf("needle: %s.const: %w", path, err)
		}
		return regexQuoteBytes([]byte(lit)), nil
	}
	if v, ok := m["enum"]; ok {
		vals, ok := v.([]any)
		if !ok {
			return "", fmt.Errorf("needle: %s.enum must be an array", path)
		}
		if len(vals) == 0 {
			return "", fmt.Errorf("needle: %s.enum must not be empty", path)
		}
		alts := make([]string, 0, len(vals))
		seen := make(map[string]struct{}, len(vals))
		for _, item := range vals {
			lit, err := canonicalScalarLiteral(item, kind)
			if err != nil {
				return "", fmt.Errorf("needle: %s.enum: %w", path, err)
			}
			if _, ok := seen[lit]; ok {
				continue
			}
			seen[lit] = struct{}{}
			alts = append(alts, regexQuoteBytes([]byte(lit)))
		}
		return alternate(alts), nil
	}
	switch kind {
	case "string":
		return toolSchemaStringPattern, nil
	case "integer":
		return toolSchemaIntPattern, nil
	case "number":
		return toolSchemaNumPattern, nil
	case "boolean":
		return `(?:true|false)`, nil
	case "null":
		return `null`, nil
	default:
		return "", fmt.Errorf("needle: %s has unsupported scalar type %q", path, kind)
	}
}

func parseBoundedInt(raw any, path string, minValue, maxValue int, required bool) (int, error) {
	if raw == nil {
		if required {
			return 0, fmt.Errorf("needle: %s is required", path)
		}
		return minValue, nil
	}
	n, ok := raw.(json.Number)
	if !ok || !isJSONIntegerLiteral(n.String()) {
		return 0, fmt.Errorf("needle: %s must be an integer", path)
	}
	v, err := strconv.Atoi(n.String())
	if err != nil {
		return 0, fmt.Errorf("needle: %s must be an integer", path)
	}
	if v < minValue || v > maxValue {
		return 0, fmt.Errorf("needle: %s must be %d..%d", path, minValue, maxValue)
	}
	return v, nil
}

func inferEnumKind(vals []any) (string, error) {
	if len(vals) == 0 {
		return "", fmt.Errorf("enum must not be empty")
	}
	kind := ""
	for _, v := range vals {
		k, err := inferScalarKind(v)
		if err != nil {
			return "", err
		}
		if kind == "" {
			kind = k
			continue
		}
		if kind == k {
			continue
		}
		if (kind == "integer" && k == "number") || (kind == "number" && k == "integer") {
			kind = "number"
			continue
		}
		return "", fmt.Errorf("mixed scalar enum types are unsupported")
	}
	return kind, nil
}

func inferScalarKind(v any) (string, error) {
	switch x := v.(type) {
	case string:
		return "string", nil
	case bool:
		return "boolean", nil
	case nil:
		return "null", nil
	case json.Number:
		if isJSONIntegerLiteral(x.String()) {
			return "integer", nil
		}
		return "number", nil
	default:
		return "", fmt.Errorf("only scalar const/enum values are supported")
	}
}

func canonicalScalarLiteral(v any, kind string) (string, error) {
	if !scalarMatchesKind(v, kind) {
		return "", fmt.Errorf("value does not match type %q", kind)
	}
	switch x := v.(type) {
	case string:
		b, _ := json.Marshal(x)
		return string(b), nil
	case bool:
		if x {
			return "true", nil
		}
		return "false", nil
	case nil:
		return "null", nil
	case json.Number:
		return x.String(), nil
	default:
		return "", fmt.Errorf("only scalar const/enum values are supported")
	}
}

func scalarMatchesKind(v any, kind string) bool {
	switch kind {
	case "string":
		_, ok := v.(string)
		return ok
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "null":
		return v == nil
	case "integer":
		n, ok := v.(json.Number)
		return ok && isJSONIntegerLiteral(n.String())
	case "number":
		_, ok := v.(json.Number)
		return ok
	default:
		return false
	}
}

func decodeJSONValue(raw []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := decodeSchemaValue(dec, 0)
	if err != nil {
		return nil, err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("multiple JSON values")
		}
		return nil, err
	}
	return v, nil
}

// Preserve every schema field during admission: encoding/json's usual
// last-key-wins map decoding could hide a marker in an earlier duplicate field
// while ToolPrompt still embeds the original RawMessage.
func decodeSchemaValue(dec *json.Decoder, depth int) (any, error) {
	if depth > 32 {
		return nil, fmt.Errorf("needle: schema JSON nesting exceeds 32")
	}
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, composite := tok.(json.Delim)
	if !composite {
		return tok, nil
	}
	switch delim {
	case '{':
		obj := make(map[string]any)
		for dec.More() {
			keyToken, err := dec.Token()
			if err != nil {
				return nil, err
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, fmt.Errorf("needle: invalid schema object key")
			}
			if _, exists := obj[key]; exists {
				return nil, fmt.Errorf("needle: duplicate schema key %q", key)
			}
			value, err := decodeSchemaValue(dec, depth+1)
			if err != nil {
				return nil, err
			}
			obj[key] = value
		}
		if _, err := dec.Token(); err != nil {
			return nil, err
		}
		return obj, nil
	case '[':
		values := make([]any, 0)
		for dec.More() {
			value, err := decodeSchemaValue(dec, depth+1)
			if err != nil {
				return nil, err
			}
			values = append(values, value)
		}
		if _, err := dec.Token(); err != nil {
			return nil, err
		}
		return values, nil
	default:
		return nil, fmt.Errorf("needle: unexpected schema delimiter")
	}
}

func alternate(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	if len(parts) == 1 {
		return parts[0]
	}
	return `(?:` + strings.Join(parts, `|`) + `)`
}

func concatPattern(parts ...string) (string, error) {
	total := 0
	for _, part := range parts {
		total += len(part)
		if total > toolGrammarRegexMaxBytes {
			return "", fmt.Errorf("needle: grammar regex exceeds %d bytes", toolGrammarRegexMaxBytes)
		}
	}
	return strings.Join(parts, ""), nil
}

func group(s string) string { return `(?:` + s + `)` }

func regexQuoteBytes(b []byte) string {
	var out strings.Builder
	out.Grow(len(b) * 2)
	for _, c := range b {
		switch c {
		case '\\', '.', '+', '*', '?', '(', ')', '|', '[', ']', '{', '}', '^', '$':
			out.WriteByte('\\')
			out.WriteByte(c)
		default:
			if c >= 0x20 && c <= 0x7e {
				out.WriteByte(c)
			} else {
				out.WriteString(`\x`)
				out.WriteByte(hexUpper[c>>4])
				out.WriteByte(hexUpper[c&0x0f])
			}
		}
	}
	return out.String()
}

func validToolName(name string) bool {
	if name == "" {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if c > 0x7f {
			return false
		}
		if i == 0 {
			if !(c == '_' || 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z') {
				return false
			}
			continue
		}
		if c == '_' || c == '.' || c == '-' || 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' {
			continue
		}
		return false
	}
	return true
}

func canonicalSchemaType(s string) string {
	switch s {
	case "object", "array", "string", "integer", "number", "null":
		return s
	case "boolean", "bool":
		return "boolean"
	default:
		return ""
	}
}

func hasKey(m map[string]any, key string) bool {
	_, ok := m[key]
	return ok
}

func isJSONIntegerLiteral(s string) bool {
	if s == "" {
		return false
	}
	if s[0] == '-' {
		s = s[1:]
		if s == "" {
			return false
		}
	}
	if s == "0" {
		return true
	}
	if len(s) > 1 && s[0] == '0' {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

var hexUpper = [16]byte{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'}

// Bound repeat expansion before Simplify/Compile allocates the expanded NFA.
func regexExpandedCost(r *syntax.Regexp, limit int) int {
	cost := 2 + len(r.Rune)
	for _, sub := range r.Sub {
		c := regexExpandedCost(sub, limit)
		if c > limit-cost {
			return limit + 1
		}
		cost += c
	}
	if r.Op == syntax.OpRepeat {
		n := max(r.Min, r.Max)
		if n < 0 {
			n = r.Min + 1
		}
		n = max(n, 1)
		if cost > limit/n {
			return limit + 1
		}
		cost *= n
	}
	return cost
}

// A matcher is private to one generation. Immutable ToolState values can share
// grammar programs; mutable traversal scratch is never stored on the grammar.
type toolMatcher struct {
	g                    *ToolGrammar
	marks                []uint32
	epoch                uint32
	stack, current, next []uint32
}

func (g *ToolGrammar) newMatcher() *toolMatcher {
	return &toolMatcher{g: g, marks: make([]uint32, len(g.prog.Inst)), stack: make([]uint32, 0, len(g.prog.Inst)), current: make([]uint32, 0, len(g.prog.Inst)), next: make([]uint32, 0, len(g.prog.Inst))}
}
func (w *toolMatcher) closure(seeds, dst []uint32) []uint32 {
	w.epoch++
	if w.epoch == 0 {
		clear(w.marks)
		w.epoch = 1
	}
	w.stack = append(w.stack[:0], seeds...)
	dst = dst[:0]
	for len(w.stack) > 0 {
		pc := w.stack[len(w.stack)-1]
		w.stack = w.stack[:len(w.stack)-1]
		if w.marks[pc] == w.epoch {
			continue
		}
		w.marks[pc] = w.epoch
		i := w.g.prog.Inst[pc]
		switch i.Op {
		case syntax.InstAlt, syntax.InstAltMatch:
			w.stack = append(w.stack, i.Out, i.Arg)
		case syntax.InstNop, syntax.InstCapture:
			w.stack = append(w.stack, i.Out)
		default:
			dst = append(dst, pc)
		}
	}
	return dst
}
func (w *toolMatcher) advance(s *ToolState, fragment []byte) (*ToolState, bool) {
	if s == nil || s.g != w.g {
		return nil, false
	}
	w.current = append(w.current[:0], s.pcs...)
	for _, b := range fragment {
		w.next = w.next[:0]
		for _, pc := range w.current {
			i := &w.g.prog.Inst[pc]
			if (i.Op == syntax.InstRune || i.Op == syntax.InstRune1) && i.MatchRune(rune(b)) {
				w.next = append(w.next, i.Out)
			}
		}
		if len(w.next) == 0 {
			return nil, false
		}
		w.current = w.closure(w.next, w.current)
	}
	return &ToolState{g: w.g, pcs: append([]uint32(nil), w.current...)}, true
}
