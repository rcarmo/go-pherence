package webui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/internal/httpinput"
)

const (
	compatMaxRequestBytes = 1 << 20
	compatMaxMessages     = 256
)

// Config configures the llama.cpp-compatible web UI adapter.
type Config struct {
	ModelID     string
	ContextSize int
	MaxTokens   int
	// DefaultMaxTokens defaults to MaxTokens. Unlimited UI requests use this bound.
	DefaultMaxTokens int
	// CurrentModel optionally reports the currently loaded model and context limit.
	CurrentModel func() (string, int)
	ChatHandler  http.Handler
}

// Register attaches the embedded UI, a minimal /props endpoint, and a
// /webui/v1/chat/completions compatibility relay onto mux.
func Register(mux *http.ServeMux, cfg Config) {
	if mux == nil {
		panic("webui: nil mux")
	}
	if cfg.ModelID == "" {
		panic("webui: empty ModelID")
	}
	if cfg.ContextSize < 0 {
		panic("webui: invalid ContextSize")
	}
	if cfg.MaxTokens <= 0 {
		panic("webui: invalid MaxTokens")
	}
	if cfg.DefaultMaxTokens < 0 || cfg.DefaultMaxTokens > cfg.MaxTokens {
		panic("webui: invalid DefaultMaxTokens")
	}
	if cfg.ChatHandler == nil {
		panic("webui: nil ChatHandler")
	}

	mux.Handle("/props", propsHandler{cfg: cfg})
	mux.Handle("/webui/v1/chat/completions", compatChatHandler{cfg: cfg})
	mux.Handle("/", NewHandler())
}

type propsHandler struct{ cfg Config }

func (h propsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.cfg = currentConfig(h.cfg)
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}
	if model := strings.TrimSpace(r.URL.Query().Get("model")); model != "" && model != h.cfg.ModelID {
		http.Error(w, "unknown model", http.StatusBadRequest)
		return
	}
	payload, err := json.Marshal(buildProps(h.cfg))
	if err != nil {
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
	if r.Method == http.MethodHead {
		w.WriteHeader(http.StatusOK)
		return
	}
	_, _ = w.Write(payload)
}

type compatChatHandler struct{ cfg Config }

func (h compatChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.cfg = currentConfig(h.cfg)
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}
	if encoding := strings.TrimSpace(r.Header.Get("Content-Encoding")); encoding != "" && !strings.EqualFold(encoding, "identity") {
		http.Error(w, "unsupported content encoding", http.StatusUnsupportedMediaType)
		return
	}
	defer r.Body.Close()

	var fields map[string]json.RawMessage
	if err := httpinput.DecodeJSON(w, r, &fields, compatMaxRequestBytes, false); err != nil {
		compatError(w, fmt.Sprintf("bad request: %v", err), httpinput.ErrorStatus(err))
		return
	}
	if err := validateCompatFields(fields); err != nil {
		compatError(w, err.Error(), http.StatusBadRequest)
		return
	}
	encoded, _ := json.Marshal(fields)
	var req compatIncomingRequest
	if err := json.Unmarshal(encoded, &req); err != nil {
		compatError(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	out, err := normalizeCompatRequest(req, h.cfg)
	if err != nil {
		compatError(w, err.Error(), http.StatusBadRequest)
		return
	}
	body, err := json.Marshal(out)
	if err != nil {
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return
	}

	proxyReq := cloneChatRequest(r, body)
	h.cfg.ChatHandler.ServeHTTP(w, proxyReq)
}

type compatIncomingRequest struct {
	Model       string                  `json:"model"`
	Messages    []compatIncomingMessage `json:"messages"`
	Stream      *bool                   `json:"stream,omitempty"`
	Temperature *float64                `json:"temperature,omitempty"`
	MaxTokens   *int                    `json:"max_tokens,omitempty"`
	Tools       []json.RawMessage       `json:"tools,omitempty"`
}

type compatIncomingMessage struct {
	Role             string            `json:"role"`
	Content          json.RawMessage   `json:"content"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID       string            `json:"tool_call_id,omitempty"`
}

type compatOutgoingRequest struct {
	Model       string                  `json:"model"`
	Messages    []compatOutgoingMessage `json:"messages"`
	Stream      *bool                   `json:"stream,omitempty"`
	Temperature *float64                `json:"temperature,omitempty"`
	MaxTokens   int                     `json:"max_tokens"`
}

type compatOutgoingMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type compatContentPart struct {
	Type string  `json:"type"`
	Text *string `json:"text,omitempty"`
}

func normalizeCompatRequest(req compatIncomingRequest, cfg Config) (compatOutgoingRequest, error) {
	if req.Model != "" && req.Model != cfg.ModelID {
		return compatOutgoingRequest{}, fmt.Errorf("unknown model %q", req.Model)
	}
	if len(req.Tools) != 0 {
		return compatOutgoingRequest{}, fmt.Errorf("tools are not supported")
	}
	if len(req.Messages) == 0 {
		return compatOutgoingRequest{}, fmt.Errorf("messages must not be empty")
	}
	if len(req.Messages) > compatMaxMessages {
		return compatOutgoingRequest{}, fmt.Errorf("messages must contain at most %d entries", compatMaxMessages)
	}
	if req.Temperature != nil && *req.Temperature != 0 {
		return compatOutgoingRequest{}, fmt.Errorf("temperature: only greedy decoding (0) is supported by this UI adapter")
	}
	maxTokens, err := normalizeMaxTokens(req.MaxTokens, cfg.MaxTokens)
	if err != nil {
		return compatOutgoingRequest{}, err
	}
	if (req.MaxTokens == nil || *req.MaxTokens == 0 || *req.MaxTokens == -1) && cfg.DefaultMaxTokens > 0 {
		maxTokens = cfg.DefaultMaxTokens
	}
	out := compatOutgoingRequest{
		Model:    cfg.ModelID,
		Messages: make([]compatOutgoingMessage, 0, len(req.Messages)),
		Stream:   req.Stream,
		// Neutral temperature is omitted: not all handlers accept this field.
		MaxTokens: maxTokens,
	}
	for i, msg := range req.Messages {
		normalized, err := normalizeCompatMessage(msg)
		if err != nil {
			return compatOutgoingRequest{}, fmt.Errorf("messages[%d]: %w", i, err)
		}
		out.Messages = append(out.Messages, normalized)
	}
	return out, nil
}

func normalizeCompatMessage(msg compatIncomingMessage) (compatOutgoingMessage, error) {
	role := strings.TrimSpace(msg.Role)
	if role != "user" && role != "assistant" && role != "system" && role != "developer" {
		return compatOutgoingMessage{}, fmt.Errorf("role %q is not supported", role)
	}
	if len(msg.ToolCalls) != 0 || msg.ToolCallID != "" {
		return compatOutgoingMessage{}, fmt.Errorf("tool calls are not supported")
	}
	content, err := flattenCompatContent(msg.Content)
	if err != nil {
		return compatOutgoingMessage{}, err
	}
	// Historical reasoning is not a new user-visible prompt and is not re-injected.
	return compatOutgoingMessage{Role: role, Content: content}, nil
}

func flattenCompatContent(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", fmt.Errorf("content is required")
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	var parts []compatContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("content must be a string or text-only content array")
	}
	var b strings.Builder
	for i, part := range parts {
		if part.Type != "text" {
			return "", fmt.Errorf("content part %d: unsupported type %q", i, part.Type)
		}
		if part.Text == nil {
			return "", fmt.Errorf("content part %d: text is required", i)
		}
		b.WriteString(*part.Text)
	}
	return b.String(), nil
}

func normalizeMaxTokens(v *int, limit int) (int, error) {
	if v == nil || *v == 0 || *v == -1 {
		return limit, nil
	}
	if *v < 0 {
		return 0, fmt.Errorf("max_tokens must be -1, 0, or within 1..%d", limit)
	}
	if *v > limit {
		return 0, fmt.Errorf("max_tokens must be within 1..%d", limit)
	}
	return *v, nil
}

func cloneChatRequest(r *http.Request, body []byte) *http.Request {
	clone := r.Clone(r.Context())
	urlCopy := *r.URL
	urlCopy.Path = "/v1/chat/completions"
	urlCopy.RawPath = ""
	urlCopy.RawQuery = ""
	clone.URL = &urlCopy
	clone.RequestURI = ""
	clone.Method = http.MethodPost
	clone.Body = io.NopCloser(bytes.NewReader(body))
	clone.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	clone.ContentLength = int64(len(body))
	clone.TransferEncoding = nil
	clone.Header = clone.Header.Clone()
	clone.Header.Del("Content-Encoding")
	clone.Header.Set("Content-Length", strconv.Itoa(len(body)))
	if clone.Header.Get("Content-Type") == "" {
		clone.Header.Set("Content-Type", "application/json")
	}
	return clone
}

type propsResponse struct {
	Role       string `json:"role"`
	ModelAlias string `json:"model_alias"`
	ModelPath  string `json:"model_path"`
	TotalSlots int    `json:"total_slots"`
	Modalities struct {
		Vision bool `json:"vision"`
		Audio  bool `json:"audio"`
		Video  bool `json:"video"`
	} `json:"modalities"`
	ChatTemplate              string `json:"chat_template"`
	BuildInfo                 string `json:"build_info"`
	DefaultGenerationSettings struct {
		NCtx   int `json:"n_ctx,omitempty"`
		Params struct {
			NPredict    int      `json:"n_predict"`
			MaxTokens   int      `json:"max_tokens"`
			Temperature int      `json:"temperature"`
			TopK        int      `json:"top_k"`
			TopP        int      `json:"top_p"`
			MinP        int      `json:"min_p"`
			Samplers    []string `json:"samplers"`
			Seed        int      `json:"seed"`
			Stream      bool     `json:"stream"`
		} `json:"params"`
	} `json:"default_generation_settings"`
}

func buildProps(cfg Config) propsResponse {
	var props propsResponse
	props.Role = "model"
	props.ModelAlias = cfg.ModelID
	props.ModelPath = cfg.ModelID
	props.TotalSlots = 1
	props.ChatTemplate = ""
	props.BuildInfo = "go-pherence"
	props.DefaultGenerationSettings.NCtx = cfg.ContextSize
	limit := cfg.DefaultMaxTokens
	if limit == 0 {
		limit = cfg.MaxTokens
	}
	props.DefaultGenerationSettings.Params.NPredict = limit
	props.DefaultGenerationSettings.Params.MaxTokens = limit
	props.DefaultGenerationSettings.Params.Temperature = 0
	props.DefaultGenerationSettings.Params.TopK = 1
	props.DefaultGenerationSettings.Params.TopP = 1
	props.DefaultGenerationSettings.Params.MinP = 0
	props.DefaultGenerationSettings.Params.Samplers = []string{}
	props.DefaultGenerationSettings.Params.Seed = -1
	props.DefaultGenerationSettings.Params.Stream = true
	return props
}

func currentConfig(cfg Config) Config {
	if cfg.CurrentModel != nil {
		cfg.ModelID, cfg.ContextSize = cfg.CurrentModel()
	}
	return cfg
}

func compatError(w http.ResponseWriter, message string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
		"message": message, "type": "unsupported_ui_request", "code": status,
	}})
}

// Accept only implemented inputs and neutral llama.cpp defaults. Never silently
// turn a requested sampler/tool/template feature into greedy text generation.
func validateCompatFields(fields map[string]json.RawMessage) error {
	neutral := map[string]string{
		"top_k": "1", "top_p": "1", "min_p": "0", "seed": "-1",
		"dynatemp_range": "0", "dynatemp_exponent": "1", "top_n_sigma": "-1",
		"xtc_probability": "0", "xtc_threshold": "0.1", "typ_p": "1",
		"repeat_last_n": "0", "repeat_penalty": "1", "presence_penalty": "0", "frequency_penalty": "0",
		"dry_multiplier": "0", "dry_base": "1.75", "dry_allowed_length": "2", "dry_penalty_last_n": "-1",
		"mirostat": "0", "mirostat_tau": "5", "mirostat_eta": "0.1",
		"samplers": "[]", "stop": "[]", "logit_bias": "[]", "grammar": `""`,
		"n_keep": "0", "ignore_eos": "false", "continue_final_message": "false", "add_generation_prompt": "true",
		"backend_sampling": "false",
	}
	for key, raw := range fields {
		switch key {
		case "model", "messages", "stream", "temperature", "max_tokens", "tools":
			continue
		// Requests to include optional telemetry/control handles don't promise
		// those handles or progress data. The frontend already tolerates absence.
		case "return_progress", "reasoning_control", "timings_per_token":
			var value bool
			if string(raw) != "null" && json.Unmarshal(raw, &value) == nil {
				continue
			}
		case "reasoning_format":
			if string(raw) == `"auto"` || string(raw) == `"none"` {
				continue
			}
		case "chat_template_kwargs":
			var value map[string]json.RawMessage
			if json.Unmarshal(raw, &value) == nil && len(value) <= 1 {
				if len(value) == 0 || string(value["enable_thinking"]) == "false" {
					continue
				}
			}
		default:
			if _, ok := neutral[key]; !ok {
				return fmt.Errorf("unknown field %q: unsupported by go-pherence's UI adapter", key)
			}
			if expected, ok := neutral[key]; ok {
				var got, want any
				if json.Unmarshal(raw, &got) == nil && json.Unmarshal([]byte(expected), &want) == nil {
					// Compare canonical JSON so e.g. top_p:1.0 equals top_p:1.
					a, _ := json.Marshal(got)
					b, _ := json.Marshal(want)
					if bytes.Equal(a, b) {
						continue
					}
				}
			}
		}
		return fmt.Errorf("%s: unsupported by go-pherence's text-only UI adapter; reset this control to its server default", key)
	}
	return nil
}
