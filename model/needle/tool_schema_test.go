package needle

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestToolGrammarPrefixPartial(t *testing.T) {
	g := mustCompileToolGrammar(t, []ToolSchema{{
		Name:       "sum",
		Parameters: rawSchema(t, `{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"],"additionalProperties":false}`),
	}}, 2)
	s := g.Start()
	fragments := []string{
		`[{"name":"s`,
		`um","arguments":`,
		`{"count":`,
		`1`,
		`}}]`,
	}
	for i, frag := range fragments {
		var ok bool
		s, ok = s.Advance([]byte(frag))
		if !ok {
			t.Fatalf("fragment %d rejected: %q", i, frag)
		}
		if i != len(fragments)-1 && s.Complete() {
			t.Fatalf("fragment %d completed too early", i)
		}
	}
	if !s.Complete() {
		t.Fatal("final state not complete")
	}
}

func TestToolGrammarRejectsImpossibleNameWrongKeyAndStringNumber(t *testing.T) {
	g := mustCompileToolGrammar(t, []ToolSchema{{
		Name:       "sum",
		Parameters: rawSchema(t, `{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"],"additionalProperties":false}`),
	}}, 1)
	cases := []string{
		`[{"name":"sub","arguments":{"count":1}}]`,
		`[{"name":"sum","argz":{"count":1}}]`,
		`[{"name":"sum","arguments":{"value":1}}]`,
		`[{"name":"sum","arguments":{"count":"1"}}]`,
	}
	for _, input := range cases {
		if _, ok := g.Start().Advance([]byte(input)); ok {
			t.Fatalf("unexpectedly accepted %q", input)
		}
	}
}

func TestToolGrammarIncompleteNotComplete(t *testing.T) {
	g := mustCompileToolGrammar(t, []ToolSchema{{
		Name:       "sum",
		Parameters: rawSchema(t, `{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"],"additionalProperties":false}`),
	}}, 1)
	s, ok := g.Start().Advance([]byte(`[{"name":"sum","arguments":{"count":1}`))
	if !ok {
		t.Fatal("valid prefix rejected")
	}
	if s.Complete() {
		t.Fatal("incomplete prefix reported complete")
	}
}

func TestToolGrammarArrayDepthAndLimits(t *testing.T) {
	accept := ToolSchema{Name: "nest", Parameters: rawSchema(t, nestedArrayObjectSchema(6))}
	g := mustCompileToolGrammar(t, []ToolSchema{accept}, 1)
	input := `[{"name":"nest","arguments":{"v":[[[[[[1]]]]]]}}]`
	s, ok := g.Start().Advance([]byte(input))
	if !ok || !s.Complete() {
		t.Fatalf("nested array input rejected: ok=%v complete=%v", ok, stateComplete(s))
	}
	reject := ToolSchema{Name: "deep", Parameters: rawSchema(t, nestedArrayObjectSchema(7))}
	if _, err := CompileToolGrammar([]ToolSchema{reject}, 1); err == nil || !strings.Contains(err.Error(), "maximum schema depth") {
		t.Fatalf("expected depth error, got %v", err)
	}
}

func TestToolGrammarObjectOrderAndRequired(t *testing.T) {
	g := mustCompileToolGrammar(t, []ToolSchema{{
		Name:       "ord",
		Parameters: rawSchema(t, `{"type":"object","properties":{"b":{"type":"integer"},"a":{"type":"integer"}},"required":["b","a"],"additionalProperties":false}`),
	}}, 1)
	good := `[{"name":"ord","arguments":{"a":1,"b":2}}]`
	s, ok := g.Start().Advance([]byte(good))
	if !ok || !s.Complete() {
		t.Fatalf("ordered object rejected: ok=%v complete=%v", ok, stateComplete(s))
	}
	bad := `[{"name":"ord","arguments":{"b":2,"a":1}}]`
	if _, ok := g.Start().Advance([]byte(bad)); ok {
		t.Fatal("out-of-order object accepted")
	}
}

func TestToolGrammarRejectsUnsupportedKeywordsRefsAndOptionalProps(t *testing.T) {
	cases := []struct {
		name   string
		schema string
		want   string
	}{
		{
			name:   "pattern",
			schema: `{"type":"object","properties":{"x":{"type":"string","pattern":"[a-z]+"}},"required":["x"],"additionalProperties":false}`,
			want:   `unsupported keyword "pattern"`,
		},
		{
			name:   "ref",
			schema: `{"type":"object","properties":{"x":{"$ref":"#/defs/x"}},"required":["x"],"additionalProperties":false}`,
			want:   `unsupported keyword "$ref"`,
		},
		{
			name:   "optional",
			schema: `{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a"],"additionalProperties":false}`,
			want:   `optional properties are not supported`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := CompileToolGrammar([]ToolSchema{{Name: "tool", Parameters: rawSchema(t, tc.schema)}}, 1)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected error containing %q, got %v", tc.want, err)
			}
		})
	}
}

func TestToolGrammarStateDoesNotMutate(t *testing.T) {
	g := mustCompileToolGrammar(t, []ToolSchema{{
		Name:       "sum",
		Parameters: rawSchema(t, `{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"],"additionalProperties":false}`),
	}}, 1)
	start := g.Start()
	if _, ok := start.Advance([]byte(`[{"name":"sum","arguments":{"count":"1"}}]`)); ok {
		t.Fatal("invalid full input accepted")
	}
	full, ok := start.Advance([]byte(`[{"name":"sum","arguments":{"count":1}}]`))
	if !ok || !full.Complete() {
		t.Fatalf("start state mutated by failed advance: ok=%v complete=%v", ok, stateComplete(full))
	}
	prefix, ok := start.Advance([]byte(`[{"name":"sum"`))
	if !ok {
		t.Fatal("valid prefix rejected")
	}
	if _, ok := prefix.Advance([]byte(`,"argz":{"count":1}}]`)); ok {
		t.Fatal("invalid continuation accepted")
	}
	final, ok := prefix.Advance([]byte(`,"arguments":{"count":1}}]`))
	if !ok || !final.Complete() {
		t.Fatalf("prefix state mutated by failed continuation: ok=%v complete=%v", ok, stateComplete(final))
	}
}

func TestToolGrammarSizeLimits(t *testing.T) {
	t.Run("single schema bytes", func(t *testing.T) {
		big := `{"type":"object","properties":{},"required":[],"description":"` + strings.Repeat("a", toolSchemaMaxBytes) + `"}`
		_, err := CompileToolGrammar([]ToolSchema{{Name: "tool", Parameters: rawSchema(t, big)}}, 1)
		if err == nil || !strings.Contains(err.Error(), "parameters exceed") {
			t.Fatalf("expected per-schema size error, got %v", err)
		}
	})
	t.Run("total schema bytes", func(t *testing.T) {
		tools := make([]ToolSchema, 0, 3)
		for i := 0; i < 3; i++ {
			name := string(rune('a' + i))
			tools = append(tools, ToolSchema{Name: name, Parameters: rawSchema(t, paddedObjectSchema(44000))})
		}
		_, err := CompileToolGrammar(tools, 1)
		if err == nil || !strings.Contains(err.Error(), "tool schemas exceed total") {
			t.Fatalf("expected total size error, got %v", err)
		}
	})
}

func mustCompileToolGrammar(t *testing.T, tools []ToolSchema, maxCalls int) *ToolGrammar {
	t.Helper()
	g, err := CompileToolGrammar(tools, maxCalls)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func rawSchema(t *testing.T, s string) json.RawMessage {
	t.Helper()
	return json.RawMessage(s)
}

func stateComplete(s *ToolState) bool {
	if s == nil {
		return false
	}
	return s.Complete()
}

func nestedArrayObjectSchema(levels int) string {
	inner := `{"type":"integer"}`
	for i := 0; i < levels; i++ {
		inner = `{"type":"array","items":` + inner + `,"maxItems":1}`
	}
	return `{"type":"object","properties":{"v":` + inner + `},"required":["v"],"additionalProperties":false}`
}

func paddedObjectSchema(size int) string {
	basePrefix := `{"type":"object","properties":{},"required":[],"description":"`
	baseSuffix := `"}`
	pad := size - len(basePrefix) - len(baseSuffix)
	if pad < 0 {
		pad = 0
	}
	return basePrefix + strings.Repeat("a", pad) + baseSuffix
}

func TestToolGrammarUnicode(t *testing.T) {
	g, err := CompileToolGrammar([]ToolSchema{{Name: "echo", Parameters: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}`)}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{`[{"name":"echo","arguments":{"x":"Olá 🚀"}}]`, `[{"name":"echo","arguments":{"x":"\ud83d\ude80"}}]`} {
		state := g.Start()
		for _, b := range []byte(text) {
			var ok bool
			state, ok = state.Advance([]byte{b})
			if !ok {
				t.Fatalf("rejected %q at %x", text, b)
			}
		}
		if !state.Complete() {
			t.Fatal("unicode not complete")
		}
	}
	for _, text := range []string{`[{"name":"echo","arguments":{"x":"\ud800"}}]`, `[{"name":"echo","arguments":{"x":"\udc00"}}]`} {
		if state, ok := g.Start().Advance([]byte(text)); ok && state.Complete() {
			t.Fatal("unpaired surrogate accepted")
		}
	}
}

func TestToolGrammarRejectsExpandedProgram(t *testing.T) {
	schema := `{"type":"string"}`
	for i := 0; i < 6; i++ {
		schema = `{"type":"array","maxItems":8,"items":` + schema + `}`
	}
	schema = `{"type":"object","properties":{"x":` + schema + `},"required":["x"]}`
	if _, err := CompileToolGrammar([]ToolSchema{{Name: "x", Parameters: json.RawMessage(schema)}}, 4); err == nil {
		t.Fatal("huge expanded grammar admitted")
	}
}

func TestToolGrammarRejectsMalformedMetadataAndBounds(t *testing.T) {
	for _, schema := range []string{
		`{"type":"object","title":12}`,
		`{"type":"object","description":null}`,
		`{"type":"object","required":null}`,
		`{"type":"object","properties":{"x":{"type":"array","items":{"type":"boolean"},"maxItems":2,"minItems":null}},"required":["x"]}`,
		`{"type":"object","properties":{"x":{"type":"boolean","const":true,"enum":[true,12]}},"required":["x"]}`,
	} {
		if _, err := CompileToolGrammar([]ToolSchema{{Name: "x", Parameters: json.RawMessage(schema)}}, 1); err == nil {
			t.Fatalf("admitted malformed schema %s", schema)
		}
	}
}

func TestToolGrammarScratchParity(t *testing.T) {
	g, err := CompileToolGrammar(lightSchema(), 2)
	if err != nil {
		t.Fatal(err)
	}
	matcher := g.newMatcher()
	for _, text := range []string{`[]`, `[{"name":"light","arguments":{"on":true}}]`, `[{"name":"light","arguments":{"on":true}},{"name":"light","arguments":{"on":false}}]`, `[{"name":"light","arguments":{"on":"true"}}]`, `[{"name":"unknown"}]`, `[] trailing`} {
		s := g.Start()
		for i, b := range []byte(text) {
			before := append([]uint32(nil), s.pcs...)
			next, ok := matcher.advance(s, []byte{b})
			whole, wholeOK := g.Start().Advance([]byte(text[:i+1]))
			if ok != wholeOK || ok && next.Complete() != whole.Complete() {
				t.Fatalf("prefix mismatch %q", text[:i+1])
			}
			if !reflect.DeepEqual(before, s.pcs) {
				t.Fatal("matcher mutated source state")
			}
			if !ok {
				break
			}
			s = next
		}
	}
}
