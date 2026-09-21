package needle

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode/utf8"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

type ToolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}
type ToolResult struct {
	Calls        []ToolCall `json:"function_calls"`
	JSON         string     `json:"json"`
	GeneratedIDs []int      `json:"generated_ids"`
	Complete     bool       `json:"complete"`
	Executed     bool       `json:"executed"`
	Calibrated   bool       `json:"calibrated"`
}
type ToolOptions struct {
	MaxNewTokens int
	MaxCalls     int
	MaxBytes     int
	Decoder      DecoderOptions
}

func containsPromptMarker(s string) bool {
	for _, marker := range []string{"<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tools>", "</tools>", "<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>"} {
		if strings.Contains(s, marker) {
			return true
		}
	}
	return false
}

// ToolPrompt uses the pinned training renderer's chat/tools layout and fixes the
// assistant prefix at <tool_call>. It does not generate free-form reasoning or
// execute calls. User-defined control markers in caller text are rejected.
func ToolPrompt(tools []ToolSchema, system, query string) (string, error) {
	if len(query) > 32<<10 || len(system) > 16<<10 || !utf8.ValidString(query) || !utf8.ValidString(system) {
		return "", fmt.Errorf("needle: tool prompt text invalid or too large")
	}
	if containsPromptMarker(query) || containsPromptMarker(system) {
		return "", fmt.Errorf("needle: caller text contains reserved prompt marker")
	}
	if _, err := CompileToolGrammar(tools, 1); err != nil {
		return "", err
	}
	for _, tool := range tools {
		node, err := decodeJSONValue(tool.Parameters)
		if err != nil {
			return "", err
		}
		if containsPromptMarker(tool.Description) || schemaHasMarker(node) {
			return "", fmt.Errorf("needle: schema contains reserved prompt marker")
		}
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(tools); err != nil {
		return "", err
	}
	schema := strings.TrimSuffix(buf.String(), "\n")
	if len(schema) > 128<<10 {
		return "", fmt.Errorf("needle: serialized tools exceed 128 KiB")
	}
	prefix := ""
	if strings.TrimSpace(system) != "" {
		prefix = "<|im_start|>system\n" + strings.TrimSpace(system) + "<|im_end|>\n"
	}
	return prefix + "<|im_start|>user\n<tools>" + schema + "</tools>\n" + query + "<|im_end|>\n<|im_start|>assistant\n<tool_call>", nil
}

// GenerateTools masks tokens by a bounded schema recognizer. Only complete
// schema-conforming arrays are returned; exhaustion returns partial JSON + error
// with Complete=false and no Calls. The returned calls have never been executed.
func (m *Model) GenerateTools(ctx context.Context, tok *checkpoint.Tokenizer, tools []ToolSchema, system, query string, opts ToolOptions) (result ToolResult, err error) {
	if ctx == nil || tok == nil || m == nil {
		return result, fmt.Errorf("needle: missing context/model/tokenizer")
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	if opts.MaxNewTokens == 0 {
		opts.MaxNewTokens = 128
	}
	if opts.MaxCalls == 0 {
		opts.MaxCalls = 1
	}
	if opts.MaxBytes == 0 {
		opts.MaxBytes = 16 << 10
	}
	if opts.MaxNewTokens < 1 || opts.MaxNewTokens > 1024 || opts.MaxBytes < 2 || opts.MaxBytes > 64<<10 {
		return result, fmt.Errorf("needle: invalid tool generation limits")
	}
	grammar, err := CompileToolGrammar(tools, opts.MaxCalls)
	if err != nil {
		return result, err
	}
	prompt, err := ToolPrompt(tools, system, query)
	if err != nil {
		return result, err
	}
	for _, marker := range []string{"<|im_start|>", "<|im_end|>", "<tools>", "</tools>", "<tool_call>"} {
		if _, ok := tok.MarkerID(marker); !ok {
			return result, fmt.Errorf("needle: tokenizer lacks tool marker %s", marker)
		}
	}
	ids, err := tok.Encode(prompt)
	if err != nil {
		return result, err
	}
	_, _, bos, _ := tok.SpecialIDs()
	ids = append([]int{bos}, ids...)
	if err = m.validateTokens(ids); err != nil {
		return result, err
	}
	bound := m.config.MaxSeq
	if m.deployed && m.archiveWindow > 0 {
		bound = min(bound, m.archiveWindow)
	}
	if len(ids)+opts.MaxNewTokens-1 > bound {
		return result, fmt.Errorf("needle: tool prompt/output budget exceeds context %d", bound)
	}
	if opts.Decoder.Capacity == 0 {
		opts.Decoder.Capacity = len(ids) + opts.MaxNewTokens - 1
	}
	d, err := m.NewDecoder(opts.Decoder)
	if err != nil {
		return result, err
	}
	pieces := make([][]byte, m.config.OutVocab)
	var pieceBytes int
	for i := range pieces {
		b, ok := tok.PieceBytes(i)
		if ok && len(b) > 0 {
			if len(b) > 4096 {
				return result, fmt.Errorf("needle: output token surface exceeds 4096 bytes")
			}
			pieceBytes += len(b)
			if pieceBytes > 4<<20 {
				return result, fmt.Errorf("needle: token surfaces exceed 4 MiB")
			}
			pieces[i] = b
		}
	}
	var logits []float32
	for _, id := range ids {
		logits, err = d.Step(ctx, id)
		if err != nil {
			return result, err
		}
	}
	state := grammar.Start()
	matcher := grammar.newMatcher()
	output := make([]byte, 0, min(opts.MaxBytes, 1024))
	ranks := make([]int, len(logits))
	for i := range ranks {
		ranks[i] = i
	}
	budget := int64(100_000_000)
	for step := 0; step < opts.MaxNewTokens; step++ {
		if err = ctx.Err(); err != nil {
			return result, err
		}
		sort.Slice(ranks, func(i, j int) bool {
			a, b := ranks[i], ranks[j]
			if logits[a] == logits[b] {
				return a < b
			}
			return logits[a] > logits[b]
		})
		chosen := -1
		for _, id := range ranks {
			b := pieces[id]
			if len(b) == 0 || len(output)+len(b) > opts.MaxBytes {
				continue
			}
			budget -= int64(len(b)+1) * int64(len(grammar.prog.Inst))
			if budget < 0 {
				return result, fmt.Errorf("needle: grammar work budget exhausted")
			}
			if err = ctx.Err(); err != nil {
				return result, err
			}
			next, valid := matcher.advance(state, b)
			if valid {
				chosen = id
				state = next
				output = append(output, b...)
				break
			}
		}
		if chosen < 0 {
			return result, fmt.Errorf("needle: no token can continue the admitted schema")
		}
		result.GeneratedIDs = append(result.GeneratedIDs, chosen)
		result.JSON = string(output)
		if state.Complete() {
			dec := json.NewDecoder(bytes.NewReader(output))
			dec.UseNumber()
			var calls []ToolCall
			if err = dec.Decode(&calls); err != nil {
				return result, fmt.Errorf("needle: grammar output invalid JSON: %w", err)
			}
			if err = dec.Decode(new(any)); err != io.EOF {
				return result, fmt.Errorf("needle: grammar output has trailing data")
			}
			result.Calls = calls
			result.Complete = true
			return result, nil
		}
		if step+1 < opts.MaxNewTokens {
			logits, err = d.Step(ctx, chosen)
			if err != nil {
				return result, err
			}
		}
	}
	return result, fmt.Errorf("needle: tool token budget exhausted before complete JSON")
}

func schemaHasMarker(node any) bool {
	switch v := node.(type) {
	case string:
		return containsPromptMarker(v)
	case map[string]any:
		for key, item := range v {
			if containsPromptMarker(key) || schemaHasMarker(item) {
				return true
			}
		}
	case []any:
		for _, item := range v {
			if schemaHasMarker(item) {
				return true
			}
		}
	}
	return false
}
