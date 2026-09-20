package webui

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestRegisterAttachesPropsRelayAndStaticRoutes(t *testing.T) {
	cfg := testCompatConfig(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	mux := http.NewServeMux()

	Register(mux, cfg)

	for _, tc := range []struct {
		name, method, target, wantPattern string
	}{
		{name: "props", method: http.MethodGet, target: "https://ui.test/props", wantPattern: "/props"},
		{name: "relay", method: http.MethodPost, target: "https://ui.test/webui/v1/chat/completions", wantPattern: "/webui/v1/chat/completions"},
		{name: "root", method: http.MethodGet, target: "https://ui.test/", wantPattern: "/"},
		{name: "static asset", method: http.MethodGet, target: "https://ui.test/_app/version.json", wantPattern: "/"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.target, nil)
			_, pattern := mux.Handler(req)
			if pattern != tc.wantPattern {
				t.Fatalf("pattern=%q want=%q for %s %s", pattern, tc.wantPattern, tc.method, tc.target)
			}
		})
	}
}

func TestNormalizeCompatRequestDefaultMaxTokensAndBounds(t *testing.T) {
	cfg := testCompatConfig(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	cfg.MaxTokens = 99
	setDefaultMaxTokens(t, &cfg, 7)

	base := compatIncomingRequest{
		Messages: []compatIncomingMessage{{Role: "user", Content: mustRawJSON(`"hello"`)}},
	}

	for _, tc := range []struct {
		name    string
		max     *int
		want    int
		wantErr string
	}{
		{name: "nil uses default", max: nil, want: 7},
		{name: "zero uses default", max: intPtr(0), want: 7},
		{name: "minus one uses default", max: intPtr(-1), want: 7},
		{name: "minimum explicit", max: intPtr(1), want: 1},
		{name: "upper bound explicit", max: intPtr(99), want: 99},
		{name: "above upper bound rejected", max: intPtr(100), wantErr: "1..99"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := base
			req.MaxTokens = tc.max
			out, err := normalizeCompatRequest(req, cfg)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("normalizeCompatRequest() error = %v", err)
				}
				if out.MaxTokens != tc.want {
					t.Fatalf("MaxTokens=%d want=%d", out.MaxTokens, tc.want)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error=%v want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestNormalizeCompatMessageRolesAndReasoningContent(t *testing.T) {
	for _, role := range []string{"user", "assistant", "system", "developer"} {
		t.Run("allow "+role, func(t *testing.T) {
			got, err := normalizeCompatMessage(compatIncomingMessage{
				Role:             role,
				Content:          mustRawJSON(`"visible"`),
				ReasoningContent: "discard me",
			})
			if err != nil {
				t.Fatalf("normalizeCompatMessage(%q) error = %v", role, err)
			}
			if got.Role != role {
				t.Fatalf("Role=%q want=%q", got.Role, role)
			}
			if got.Content != "visible" {
				t.Fatalf("Content=%q want=%q (reasoning_content must be discarded)", got.Content, "visible")
			}
		})
	}

	for _, role := range []string{"tool", "function", "critic"} {
		t.Run("reject "+role, func(t *testing.T) {
			_, err := normalizeCompatMessage(compatIncomingMessage{Role: role, Content: mustRawJSON(`"hi"`)})
			if err == nil || !strings.Contains(err.Error(), "role") {
				t.Fatalf("error=%v want role rejection for %q", err, role)
			}
		})
	}
}

func TestCompatChatHandlerRejectsBadRequests(t *testing.T) {
	cfg := testCompatConfig(t, rejectingChatHandler(t))
	h := compatChatHandler{cfg: cfg}

	for _, tc := range []struct {
		name         string
		method       string
		target       string
		body         string
		headers      map[string]string
		wantCode     int
		wantAllow    string
		wantContains []string
	}{
		{
			name:      "method",
			method:    http.MethodGet,
			target:    "https://ui.test/webui/v1/chat/completions",
			wantCode:  http.StatusMethodNotAllowed,
			wantAllow: "POST",
		},
		{
			name:         "content encoding",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}]}`,
			headers:      map[string]string{"Content-Encoding": "gzip"},
			wantCode:     http.StatusUnsupportedMediaType,
			wantContains: []string{"unsupported content encoding"},
		},
		{
			name:         "trailing json",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}]} {}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"single JSON value"},
		},
		{
			name:         "null messages",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":null}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"messages"},
		},
		{
			name:         "null content",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":null}]}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"content"},
		},
		{
			name:         "unsupported media",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":[{"type":"image_url","text":"x"}]}]}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"unsupported", "image_url"},
		},
		{
			name:         "tools unsupported",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}],"tools":[{}]}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"tools"},
		},
		{
			name:         "unknown top level field",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}],"bogus":true}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"unknown field", "bogus"},
		},
		{
			name:         "nondefault top_p rejected",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}],"top_p":0.5}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"top_p"},
		},
		{
			name:         "nondefault temperature rejected",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}],"temperature":0.5}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"temperature"},
		},
		{
			name:         "thinking budget rejected",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}],"thinking_budget_tokens":12}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"thinking_budget_tokens"},
		},
		{
			name:         "continue final message rejected",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}],"continue_final_message":true}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"continue_final_message"},
		},
		{
			name:         "n_predict zero rejected",
			method:       http.MethodPost,
			target:       "https://ui.test/webui/v1/chat/completions",
			body:         `{"messages":[{"role":"user","content":"ok"}],"n_predict":0}`,
			wantCode:     http.StatusBadRequest,
			wantContains: []string{"n_predict"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var body io.Reader
			if tc.body != "" {
				body = strings.NewReader(tc.body)
			}
			req := httptest.NewRequest(tc.method, tc.target, body)
			if tc.body != "" {
				req.Header.Set("Content-Type", "application/json")
			}
			for k, v := range tc.headers {
				req.Header.Set(k, v)
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, req)
			if w.Code != tc.wantCode {
				t.Fatalf("code=%d want=%d body=%q", w.Code, tc.wantCode, w.Body.String())
			}
			if tc.wantAllow != "" && w.Header().Get("Allow") != tc.wantAllow {
				t.Fatalf("Allow=%q want=%q", w.Header().Get("Allow"), tc.wantAllow)
			}
			for _, needle := range tc.wantContains {
				if !strings.Contains(w.Body.String(), needle) {
					t.Fatalf("body=%q missing %q", w.Body.String(), needle)
				}
			}
		})
	}
}

func TestCompatChatHandlerRejectsOversizeBody(t *testing.T) {
	cfg := testCompatConfig(t, rejectingChatHandler(t))
	h := compatChatHandler{cfg: cfg}

	body := `{"messages":[{"role":"user","content":"` + strings.Repeat("x", compatMaxRequestBytes) + `"}]}`
	req := httptest.NewRequest(http.MethodPost, "https://ui.test/webui/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("code=%d want=%d body=%q", w.Code, http.StatusRequestEntityTooLarge, w.Body.String())
	}
}

func TestCompatChatHandlerAllowsInertDefaultsAndForwardsNormalizedBody(t *testing.T) {
	type ctxKey struct{}

	var (
		forwardedReq  *http.Request
		forwardedBody []byte
	)

	chat := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		forwardedReq = r
		var err error
		forwardedBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll(forwarded body): %v", err)
		}
		if got := r.Context().Value(ctxKey{}); got != "ctx-ok" {
			t.Fatalf("context value=%v want=%q", got, "ctx-ok")
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("forwarded path=%q want=%q", r.URL.Path, "/v1/chat/completions")
		}
		if r.URL.RawQuery != "" {
			t.Fatalf("forwarded RawQuery=%q want empty", r.URL.RawQuery)
		}
		if r.RequestURI != "" {
			t.Fatalf("forwarded RequestURI=%q want empty", r.RequestURI)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("forwarded Method=%q want=%q", r.Method, http.MethodPost)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Fatalf("Authorization=%q", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Fatalf("Accept=%q want=%q", r.Header.Get("Accept"), "text/event-stream")
		}
		if got := r.Header.Get("Content-Encoding"); got != "" {
			t.Fatalf("Content-Encoding=%q want empty", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Fatalf("Content-Type=%q want=%q", got, "application/json")
		}
		if got := r.Header.Get("Content-Length"); got != strconv.Itoa(len(forwardedBody)) {
			t.Fatalf("Content-Length header=%q want=%d", got, len(forwardedBody))
		}
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("writer %T does not implement http.Flusher", w)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: start\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: done\n\n")
	})

	cfg := testCompatConfig(t, chat)
	cfg.MaxTokens = 99
	setDefaultMaxTokens(t, &cfg, 7)
	h := compatChatHandler{cfg: cfg}

	body := `{
		"messages": [
			{"role":"developer","content":"setup","reasoning_content":"discard this"},
			{"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}
		],
		"stream": true,
		"temperature": 0,
		"top_p": 1,
		"top_k": 1,
		"seed": -1,
		"dry_multiplier": 0,
		"return_progress": true,
		"reasoning_control": true
	}`

	req := httptest.NewRequest(http.MethodPost, "https://ui.test/webui/v1/chat/completions?keep=this", strings.NewReader(body))
	req = req.WithContext(context.WithValue(req.Context(), ctxKey{}, "ctx-ok"))
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Authorization", "Bearer test-token")
	req.Header.Set("Content-Encoding", "identity")
	req.Header.Set("Content-Type", "application/json")

	origPath := req.URL.Path
	origQuery := req.URL.RawQuery
	origRequestURI := req.RequestURI
	origEncoding := req.Header.Get("Content-Encoding")
	origAccept := req.Header.Get("Accept")

	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("code=%d want=%d body=%q", w.Code, http.StatusOK, w.Body.String())
	}
	if !w.Flushed {
		t.Fatalf("expected SSE flush; recorder was not flushed")
	}
	if got := w.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("response Content-Type=%q want=%q", got, "text/event-stream")
	}
	if got := w.Body.String(); got != "data: start\n\ndata: done\n\n" {
		t.Fatalf("response body=%q", got)
	}
	if forwardedReq == nil {
		t.Fatalf("ChatHandler was not called")
	}

	if req.URL.Path != origPath || req.URL.RawQuery != origQuery || req.RequestURI != origRequestURI {
		t.Fatalf("original request mutated: path=%q query=%q uri=%q", req.URL.Path, req.URL.RawQuery, req.RequestURI)
	}
	if req.Header.Get("Content-Encoding") != origEncoding || req.Header.Get("Accept") != origAccept {
		t.Fatalf("original headers mutated: Content-Encoding=%q Accept=%q", req.Header.Get("Content-Encoding"), req.Header.Get("Accept"))
	}

	var out compatOutgoingRequest
	if err := json.Unmarshal(forwardedBody, &out); err != nil {
		t.Fatalf("forwarded body is not valid JSON: %v\n%s", err, forwardedBody)
	}
	if out.Model != cfg.ModelID {
		t.Fatalf("Model=%q want=%q", out.Model, cfg.ModelID)
	}
	if out.Stream == nil || !*out.Stream {
		t.Fatalf("Stream=%v want true", out.Stream)
	}
	if out.Temperature != nil {
		t.Fatalf("Temperature=%v want nil when input temperature is zero", *out.Temperature)
	}
	if out.MaxTokens != 7 {
		t.Fatalf("MaxTokens=%d want=%d", out.MaxTokens, 7)
	}
	if len(out.Messages) != 2 {
		t.Fatalf("len(Messages)=%d want=2", len(out.Messages))
	}
	if out.Messages[0].Role != "developer" || out.Messages[0].Content != "setup" {
		t.Fatalf("message[0]=%+v want role=developer content=setup", out.Messages[0])
	}
	if out.Messages[1].Role != "user" || out.Messages[1].Content != "hello world" {
		t.Fatalf("message[1]=%+v want role=user content='hello world'", out.Messages[1])
	}
	if strings.Contains(string(forwardedBody), "discard this") {
		t.Fatalf("forwarded body leaked reasoning_content: %s", forwardedBody)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(forwardedBody, &raw); err != nil {
		t.Fatalf("json.Unmarshal(raw): %v", err)
	}
	wantKeys := map[string]bool{"model": true, "messages": true, "stream": true, "max_tokens": true}
	if len(raw) != len(wantKeys) {
		t.Fatalf("forwarded keys=%v want exactly %v in %s", mapKeys(raw), mapKeysBool(wantKeys), forwardedBody)
	}
	for key := range raw {
		if !wantKeys[key] {
			t.Fatalf("unexpected forwarded key %q in %s", key, forwardedBody)
		}
	}
}

func testCompatConfig(t *testing.T, chat http.Handler) Config {
	t.Helper()
	if chat == nil {
		chat = rejectingChatHandler(t)
	}
	cfg := Config{
		ModelID:     "test-model",
		ContextSize: 4096,
		MaxTokens:   32,
		ChatHandler: chat,
	}
	setDefaultMaxTokens(t, &cfg, 7)
	return cfg
}

func setDefaultMaxTokens(t *testing.T, cfg *Config, v int) {
	t.Helper()
	rv := reflect.ValueOf(cfg).Elem()
	field := rv.FieldByName("DefaultMaxTokens")
	if !field.IsValid() {
		t.Fatalf("Config.DefaultMaxTokens field missing")
	}
	if !field.CanSet() || field.Kind() != reflect.Int {
		t.Fatalf("Config.DefaultMaxTokens must be a settable int field")
	}
	field.SetInt(int64(v))
}

func rejectingChatHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatalf("ChatHandler called unexpectedly")
	})
}

func mustRawJSON(s string) json.RawMessage { return json.RawMessage(s) }

func intPtr(v int) *int { return &v }

func mapKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func mapKeysBool(m map[string]bool) []string { return mapKeys(m) }
