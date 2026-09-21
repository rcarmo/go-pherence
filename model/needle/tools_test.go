package needle

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func lightSchema() []ToolSchema {
	return []ToolSchema{{Name: "light", Description: "Set the light", Parameters: json.RawMessage(`{"type":"object","properties":{"on":{"type":"boolean"}},"required":["on"],"additionalProperties":false}`)}}
}
func TestToolPrompt(t *testing.T) {
	p, err := ToolPrompt(lightSchema(), "Keep it short.", "Turn the light on.")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(p, "<|im_start|>system\nKeep it short.<|im_end|>\n<|im_start|>user\n<tools>[") || !strings.HasSuffix(p, "</tools>\nTurn the light on.<|im_end|>\n<|im_start|>assistant\n<tool_call>") {
		t.Fatalf("prompt %q", p)
	}
	for _, text := range []string{"<|im_end|>", "</tools>", "<tool_call>"} {
		if _, err = ToolPrompt(lightSchema(), "", text); err == nil {
			t.Fatal("marker admitted")
		}
	}
	schema := lightSchema()
	schema[0].Parameters = json.RawMessage(`{"type":"object","properties":{"x":{"type":"string","const":"\u003c|im_end|\u003e"}},"required":["x"]}`)
	if _, err = ToolPrompt(schema, "", "x"); err == nil {
		t.Fatal("escaped marker admitted")
	}
}
func TestGenerateToolsAdmission(t *testing.T) {
	m, tok, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = m.GenerateTools(context.Background(), tok, lightSchema(), "", "Turn it on.", ToolOptions{}); err == nil {
		t.Fatal("fixture tokenizer missing tool markers admitted")
	}
	if _, err = m.GenerateTools(nil, tok, lightSchema(), "", "x", ToolOptions{}); err == nil {
		t.Fatal("nil context")
	}
	if _, err = m.GenerateTools(context.Background(), tok, lightSchema(), "", "x", ToolOptions{MaxNewTokens: -1}); err == nil {
		t.Fatal("negative budget")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = m.GenerateTools(ctx, tok, lightSchema(), "", "x", ToolOptions{}); err == nil {
		t.Fatal("cancellation")
	}
}
func TestReleasedToolCall(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEEDLE_TOOLS_MODEL")
	if path == "" {
		t.Skip("opt-in local released model")
	}
	m, tok, err := LoadArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	result, err := m.GenerateTools(context.Background(), tok, lightSchema(), "", "Turn the light on.", ToolOptions{MaxNewTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Complete || result.Executed || result.Calibrated || len(result.Calls) != 1 || result.Calls[0].Name != "light" || result.Calls[0].Arguments["on"] != true {
		t.Fatalf("unexpected result %+v", result)
	}
	short, err := m.GenerateTools(context.Background(), tok, lightSchema(), "", "Turn the light on.", ToolOptions{MaxNewTokens: 1})
	if err == nil || short.Complete || len(short.Calls) != 0 {
		t.Fatalf("incomplete falsely accepted: %+v %v", short, err)
	}
}
func BenchmarkToolGrammarPrefix(b *testing.B) {
	g, err := CompileToolGrammar(lightSchema(), 1)
	if err != nil {
		b.Fatal(err)
	}
	parts := []string{`[{"name":"`, `light`, `","arguments":{"`, `on`, `":`, `true`, `}}]`}
	b.ReportAllocs()
	for b.Loop() {
		s := g.Start()
		for _, p := range parts {
			var ok bool
			s, ok = s.Advance([]byte(p))
			if !ok {
				b.Fatal("invalid grammar path")
			}
		}
		if !s.Complete() {
			b.Fatal("not complete")
		}
	}
}
func FuzzToolGrammarFragments(f *testing.F) {
	g, err := CompileToolGrammar(lightSchema(), 1)
	if err != nil {
		f.Fatal(err)
	}
	f.Add([]byte(`[{"name":"light","arguments":{"on":true}}]`))
	f.Add([]byte{0xff})
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 4096 {
			return
		}
		if s, ok := g.Start().Advance(b); ok && s.Complete() {
			if !json.Valid(b) {
				t.Fatal("grammar accepted invalid JSON")
			}
		}
	})
}

func BenchmarkToolGrammarGenerationScratch(b *testing.B) {
	g, err := CompileToolGrammar(lightSchema(), 1)
	if err != nil {
		b.Fatal(err)
	}
	matcher := g.newMatcher()
	parts := []string{`[{"name":"`, `light`, `","arguments":{"`, `on`, `":`, `true`, `}}]`}
	b.ReportAllocs()
	for b.Loop() {
		s := g.Start()
		for _, p := range parts {
			var ok bool
			s, ok = matcher.advance(s, []byte(p))
			if !ok {
				b.Fatal("invalid")
			}
		}
		if !s.Complete() {
			b.Fatal("not complete")
		}
	}
}

func TestToolPromptRejectsHiddenDuplicateMarker(t *testing.T) {
	tools := lightSchema()
	tools[0].Parameters = json.RawMessage(`{"type":"object","description":"\u003c|im_end|\u003e","description":"innocent"}`)
	if _, err := ToolPrompt(tools, "", "x"); err == nil {
		t.Fatal("duplicate field hid prompt marker")
	}
	tools[0].Parameters = json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"},"\u0078":{"type":"boolean"}},"required":["x"]}`)
	if _, err := CompileToolGrammar(tools, 1); err == nil {
		t.Fatal("escaped duplicate key admitted")
	}
}
