package webui

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestBrowserServer is a loopback-only fixture for Playwright, never inference.
// Normal Go tests skip it; playwright.go.config.ts enables it explicitly.
func TestBrowserServer(t *testing.T) {
	if os.Getenv("WEBUI_BROWSER_TEST") != "1" {
		t.Skip("Playwright fixture only")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{map[string]any{"id": "fixture-model", "object": "model", "created": 1, "owned_by": "go-pherence"}}})
	})
	Register(mux, Config{ModelID: "fixture-model", ContextSize: 4096, MaxTokens: 256, ChatHandler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req compatOutgoingRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			http.Error(w, "bad fixture input", 400)
			return
		}
		if strings.Contains(req.Messages[len(req.Messages)-1].Content, "fixture-error") {
			compatError(w, "Synthetic busy error", http.StatusTooManyRequests)
			return
		}
		if req.Stream == nil || !*req.Stream {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"fixture","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"Synthetic reply from Go."},"finish_reason":"stop"}]}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, text := range []string{"Synthetic ", "reply ", "from Go."} {
			if r.Context().Err() != nil {
				return
			}
			chunk := map[string]any{"id": "fixture", "object": "chat.completion.chunk", "model": "fixture-model", "created": 1, "choices": []any{map[string]any{"index": 0, "delta": map[string]string{"content": text}, "finish_reason": nil}}}
			b, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", b)
			w.(http.Flusher).Flush()
			time.Sleep(25 * time.Millisecond)
		}
		fmt.Fprint(w, "data: {\"id\":\"fixture\",\"model\":\"fixture-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	})})
	listener, err := net.Listen("tcp", "127.0.0.1:18181")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go srv.Serve(listener)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	_ = srv.Close()
}
