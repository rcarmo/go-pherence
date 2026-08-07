package servingbench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// SSEEvent is a parsed Server-Sent Events frame.
type SSEEvent struct {
	Event string
	Data  string
	ID    string
	Retry string
}

// Usage mirrors OpenAI-compatible token accounting when present.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionChunk is one logical update from a streaming chat completion.
type ChatCompletionChunk struct {
	Index            int
	Content          string
	ReasoningContent string
	FinishReason     string
	Usage            *Usage
	Done             bool
}

// ParseSSE parses an SSE stream without scanner token limits.
func ParseSSE(r io.Reader, fn func(SSEEvent) error) error {
	br := bufio.NewReader(r)
	var event SSEEvent
	var dataLines []string
	haveFields := false
	flush := func() error {
		if !haveFields && len(dataLines) == 0 {
			return nil
		}
		event.Data = strings.Join(dataLines, "\n")
		if err := fn(event); err != nil {
			return err
		}
		event = SSEEvent{}
		dataLines = dataLines[:0]
		haveFields = false
		return nil
	}

	for {
		line, err := readSSELine(br)
		if err != nil && err != io.EOF {
			return err
		}
		if err == io.EOF && len(line) == 0 {
			return flush()
		}
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			if err == io.EOF {
				return nil
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			if err == io.EOF {
				return flush()
			}
			continue
		}

		field, value, ok := strings.Cut(line, ":")
		if ok && strings.HasPrefix(value, " ") {
			value = value[1:]
		}
		if !ok {
			field = line
			value = ""
		}
		switch field {
		case "event":
			event.Event = value
			haveFields = true
		case "data":
			dataLines = append(dataLines, value)
			haveFields = true
		case "id":
			event.ID = value
			haveFields = true
		case "retry":
			event.Retry = value
			haveFields = true
		}
		if err == io.EOF {
			return flush()
		}
	}
}

func readSSELine(br *bufio.Reader) (string, error) {
	var b []byte
	for {
		frag, isPrefix, err := br.ReadLine()
		if err != nil {
			if err == io.EOF && len(b) > 0 {
				return strings.TrimSuffix(string(b), "\r"), io.EOF
			}
			return "", err
		}
		b = append(b, frag...)
		if !isPrefix {
			return strings.TrimSuffix(string(b), "\r"), nil
		}
	}
}

// ParseChatCompletionStream parses an OpenAI-compatible streaming chat
// completion response carried over SSE.
func ParseChatCompletionStream(r io.Reader, fn func(ChatCompletionChunk) error) error {
	type streamChoice struct {
		Index int `json:"index"`
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	}
	type streamChunk struct {
		Choices []streamChoice `json:"choices"`
		Usage   *Usage         `json:"usage"`
	}

	return ParseSSE(r, func(event SSEEvent) error {
		data := strings.TrimSpace(event.Data)
		if data == "" {
			return nil
		}
		if data == "[DONE]" {
			return fn(ChatCompletionChunk{Done: true})
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("decode stream chunk: %w", err)
		}
		if len(chunk.Choices) == 0 {
			if chunk.Usage != nil {
				return fn(ChatCompletionChunk{Usage: chunk.Usage})
			}
			return nil
		}
		for _, choice := range chunk.Choices {
			update := ChatCompletionChunk{
				Index:            choice.Index,
				Content:          choice.Delta.Content,
				ReasoningContent: choice.Delta.ReasoningContent,
				Usage:            chunk.Usage,
			}
			if choice.FinishReason != nil {
				update.FinishReason = *choice.FinishReason
			}
			if err := fn(update); err != nil {
				return err
			}
		}
		return nil
	})
}
