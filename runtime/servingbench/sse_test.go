package servingbench

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestParseSSEHandlesFieldsMultilineDataAndLongLines(t *testing.T) {
	longData := strings.Repeat("x", 70_000)
	stream := strings.Join([]string{
		": comment",
		"id: 42",
		"event: chunk",
		"retry: 1000",
		"data: first",
		"data: second",
		"",
		"data: " + longData,
		"",
	}, "\n")

	var events []SSEEvent
	if err := ParseSSE(strings.NewReader(stream), func(event SSEEvent) error {
		events = append(events, event)
		return nil
	}); err != nil {
		t.Fatalf("ParseSSE() error = %v", err)
	}
	want := []SSEEvent{
		{Event: "chunk", Data: "first\nsecond", ID: "42", Retry: "1000"},
		{Data: longData},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("ParseSSE() = %#v, want %#v", events, want)
	}
}

func TestParseChatCompletionStreamHandlesUsageLongLinesAndDone(t *testing.T) {
	longContent := strings.Repeat("z", 70_000)
	contentChunk, err := json.Marshal(map[string]any{
		"choices": []map[string]any{{
			"index": 0,
			"delta": map[string]any{"content": longContent, "reasoning_content": "ponder"},
		}},
	})
	if err != nil {
		t.Fatalf("marshal content chunk: %v", err)
	}
	usageChunk, err := json.Marshal(map[string]any{
		"choices": []map[string]any{},
		"usage":   map[string]any{"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
	})
	if err != nil {
		t.Fatalf("marshal usage chunk: %v", err)
	}
	stream := "data: " + string(contentChunk) + "\n\n" +
		"data: " + string(usageChunk) + "\n\n" +
		"data: [DONE]\n\n"

	var chunks []ChatCompletionChunk
	if err := ParseChatCompletionStream(strings.NewReader(stream), func(chunk ChatCompletionChunk) error {
		chunks = append(chunks, chunk)
		return nil
	}); err != nil {
		t.Fatalf("ParseChatCompletionStream() error = %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunk count = %d, want 3", len(chunks))
	}
	if chunks[0].Index != 0 || chunks[0].Content != longContent || chunks[0].ReasoningContent != "ponder" {
		t.Fatalf("content chunk = %#v", chunks[0])
	}
	if chunks[1].Usage == nil || *chunks[1].Usage != (Usage{PromptTokens: 7, CompletionTokens: 2, TotalTokens: 9}) {
		t.Fatalf("usage chunk = %#v", chunks[1])
	}
	if !chunks[2].Done {
		t.Fatalf("done chunk = %#v", chunks[2])
	}
}

func TestSSEBoundsAndCompletionFailClosed(t *testing.T) {
	for _, body := range []string{"data: " + strings.Repeat("x", MaxSSEEventBytes+1), strings.Repeat(":keepalive\n", MaxSSEEventBytes/10+2)} {
		if err := ParseSSE(strings.NewReader(body), func(SSEEvent) error { return nil }); !errors.Is(err, ErrSSETooLarge) {
			t.Fatalf("unbounded frame accepted: %v", err)
		}
	}
	for _, body := range []string{"", `data: {"choices":[]}` + "\n\n", `data: {"error":{"message":"failed"}}` + "\n\n"} {
		if err := ParseChatCompletionStream(strings.NewReader(body), func(ChatCompletionChunk) error { return nil }); err == nil {
			t.Fatal("incomplete/error stream accepted")
		}
	}
	count := 0
	if err := ParseChatCompletionStream(strings.NewReader("data: [DONE]\n\ndata: broken after complete\n\n"), func(c ChatCompletionChunk) error {
		if c.Done {
			count++
		}
		return nil
	}); err != nil || count != 1 {
		t.Fatal(err, count)
	}
}

func TestStreamUsageRejectsNegativeInconsistentOrOverflowingCounts(t *testing.T) {
	max := int(^uint(0) >> 1)
	for _, u := range []Usage{{-1, 1, 0}, {1, 2, 2}, {max, 1, max}, {0, -1, 0}} {
		body, _ := json.Marshal(map[string]any{"usage": u})
		stream := "data: " + string(body) + "\n\ndata: [DONE]\n\n"
		if err := ParseChatCompletionStream(strings.NewReader(stream), func(ChatCompletionChunk) error { return nil }); err == nil {
			t.Fatal("bad usage admitted", u)
		}
	}
}
