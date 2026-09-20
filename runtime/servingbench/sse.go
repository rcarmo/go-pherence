package servingbench

import (
	"bufio"
	"encoding/json"
	"errors"
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

// MaxSSEEventBytes bounds a single line/frame, including comments and fields.
// This permits large payloads beyond Scanner's default64KiB without unbounded
// memory growth on an unterminated stream. It does not limit total stream bytes.
const MaxSSEEventBytes = 4 << 20

var ErrSSETooLarge = errors.New("SSE event exceeds size limit")

// ParseSSE parses an SSE stream with a bounded frame/line size.
func ParseSSE(r io.Reader, fn func(SSEEvent) error) error {
	if r == nil || fn == nil {
		return fmt.Errorf("nil SSE reader/callback")
	}
	br := bufio.NewReader(r)
	var event SSEEvent
	var dataLines []string
	haveFields := false
	frameBytes := 0
	flush := func() error {
		frameBytes = 0
		if !haveFields && len(dataLines) == 0 {
			return nil
		}
		event.Data = strings.Join(dataLines, "\n")
		if err := fn(event); err != nil {
			return err
		}
		event = SSEEvent{}
		clear(dataLines) // release prior frame strings retained by slice capacity
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
		if len(line)+1 > MaxSSEEventBytes-frameBytes {
			return ErrSSETooLarge
		}
		frameBytes += len(line) + 1
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
		if len(frag) > MaxSSEEventBytes-len(b) {
			return "", ErrSSETooLarge
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
	if fn == nil {
		return fmt.Errorf("nil chat callback")
	}
	type streamChoice struct {
		Index int `json:"index"`
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	}
	type streamChunk struct {
		Choices []streamChoice  `json:"choices"`
		Usage   *Usage          `json:"usage"`
		Error   json.RawMessage `json:"error"`
	}

	done := errors.New("chat stream complete")
	err := ParseSSE(r, func(event SSEEvent) error {
		data := strings.TrimSpace(event.Data)
		if data == "" {
			return nil
		}
		if data == "[DONE]" {
			if err := fn(ChatCompletionChunk{Done: true}); err != nil {
				return err
			}
			return done
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("decode stream chunk: %w", err)
		}
		if event.Event == "error" || len(chunk.Error) > 0 && string(chunk.Error) != "null" {
			return fmt.Errorf("server stream error: %s", data)
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
	if errors.Is(err, done) {
		return nil
	}
	if err == nil {
		return io.ErrUnexpectedEOF
	}
	return err
}
