package servingbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunWithHTTPTestStreamedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if got := r.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("accept header = %q", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("content-type = %q", got)
		}
		var body chatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
			return
		}
		if body.Model != "demo-model" || !body.Stream || body.MaxTokens != 4 || len(body.Messages) != 1 || body.Messages[0].Content != "hello" {
			t.Errorf("unexpected request body: %+v", body)
		}
		writeTestStreamResponse(t, w, []any{
			map[string]any{"choices": []map[string]any{{"index": 0, "delta": map[string]any{"role": "assistant"}}}},
			map[string]any{"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": "hello"}}}},
			map[string]any{"choices": []map[string]any{{"index": 0, "delta": map[string]any{"reasoning_content": "plan"}}}},
			map[string]any{"choices": []map[string]any{{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}}},
			map[string]any{"choices": []map[string]any{}, "usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
		})
	}))
	defer server.Close()

	report, err := Run(context.Background(), Config{
		Endpoint:     server.URL,
		Model:        "demo-model",
		Prompts:      []string{"hello"},
		RequestCount: 1,
		Concurrency:  1,
		MaxTokens:    4,
		Seed:         1,
		Arrival:      ArrivalConfig{Mode: ArrivalFixed, Rate: 1},
		SLO:          SLOConfig{TTFT: time.Second, ITL: time.Second, E2E: 2 * time.Second},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(report.Requests) != 1 {
		t.Fatalf("request count = %d, want 1", len(report.Requests))
	}
	req := report.Requests[0]
	if !req.Successful() || req.Status != "ok" || req.StatusCode != 200 {
		t.Fatalf("request = %+v", req)
	}
	if req.FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want stop", req.FinishReason)
	}
	if len(req.TokenOffsets) != 2 || req.FirstTokenOffset == nil || req.LastTokenOffset == nil || req.EndOffset == nil {
		t.Fatalf("token timing fields = %+v", req)
	}
	if req.PromptTokens == nil || *req.PromptTokens != 3 || req.CompletionTokens == nil || *req.CompletionTokens != 2 || req.TotalTokens == nil || *req.TotalTokens != 5 {
		t.Fatalf("usage fields = %+v", req)
	}
	if report.Summary.Successful != 1 || report.Summary.GoodRequests != 1 || report.Summary.TotalTokens != 5 {
		t.Fatalf("summary = %+v", report.Summary)
	}
}

func TestRunConcurrencyQueueing(t *testing.T) {
	release := []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
	started := make(chan int, 3)
	var count int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		idx := int(atomic.AddInt32(&count, 1) - 1)
		started <- idx
		<-release[idx]
		writeTestStreamResponse(t, w, []any{
			map[string]any{"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": fmt.Sprintf("token-%d", idx)}}}},
		})
	}))
	defer server.Close()

	type runResult struct {
		report Report
		err    error
	}
	done := make(chan runResult, 1)
	go func() {
		report, err := Run(context.Background(), Config{
			Endpoint:     server.URL,
			Prompts:      []string{"one", "two", "three"},
			RequestCount: 3,
			Concurrency:  1,
			MaxTokens:    1,
			Seed:         1,
			Arrival:      ArrivalConfig{Mode: ArrivalFixed, Rate: 1000},
		})
		done <- runResult{report: report, err: err}
	}()

	if got := <-started; got != 0 {
		t.Fatalf("first started index = %d, want 0", got)
	}
	time.Sleep(20 * time.Millisecond)
	close(release[0])
	if got := <-started; got != 1 {
		t.Fatalf("second started index = %d, want 1", got)
	}
	time.Sleep(20 * time.Millisecond)
	close(release[1])
	if got := <-started; got != 2 {
		t.Fatalf("third started index = %d, want 2", got)
	}
	close(release[2])

	result := <-done
	if result.err != nil {
		t.Fatalf("Run() error = %v", result.err)
	}
	report := result.report
	if report.Summary.Successful != 3 {
		t.Fatalf("summary = %+v", report.Summary)
	}
	q1, ok1 := report.Requests[1].QueueTime()
	q2, ok2 := report.Requests[2].QueueTime()
	if !ok1 || !ok2 || q1 <= 0 || q2 <= q1 {
		t.Fatalf("queue times = %v %v ok=%v/%v", q1, q2, ok1, ok2)
	}
	if report.Requests[2].ArrivalOffset == nil || report.Requests[1].StartOffset == nil {
		t.Fatalf("missing timing fields: %+v", report.Requests)
	}
	if !(*report.Requests[2].ArrivalOffset < *report.Requests[1].StartOffset) {
		t.Fatalf("third arrival %v should precede second start %v", *report.Requests[2].ArrivalOffset, *report.Requests[1].StartOffset)
	}
}

func TestRunCancellationStopsFutureArrivals(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		cancel()
		writeTestStreamResponse(t, w, []any{
			map[string]any{"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": "x"}}}},
		})
	}))
	defer server.Close()

	report, err := Run(ctx, Config{
		Endpoint:     server.URL,
		Prompts:      []string{"one", "two", "three"},
		RequestCount: 3,
		Concurrency:  1,
		MaxTokens:    1,
		Seed:         1,
		Arrival:      ArrivalConfig{Mode: ArrivalFixed, Rate: 20},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if atomic.LoadInt32(&calls) != 1 {
		t.Fatalf("server calls = %d, want 1", calls)
	}
	for i := 1; i < len(report.Requests); i++ {
		if report.Requests[i].ArrivalOffset != nil {
			t.Fatalf("request %d arrival offset = %v, want nil", i, *report.Requests[i].ArrivalOffset)
		}
		if !report.Requests[i].Cancelled || report.Requests[i].Status != "canceled" || !strings.Contains(report.Requests[i].Error, "context canceled") {
			t.Fatalf("request %d = %+v", i, report.Requests[i])
		}
	}
}

func TestRunRequestTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(250 * time.Millisecond):
			// Some HTTP transports do not propagate client cancellation to a
			// handler that has not written headers yet. Bound the test handler so
			// httptest.Server.Close cannot hang even in that case.
		}
	}))
	defer server.Close()

	report, err := Run(context.Background(), Config{
		Endpoint:     server.URL,
		Prompts:      []string{"slow"},
		RequestCount: 1,
		Concurrency:  1,
		MaxTokens:    1,
		Seed:         1,
		Timeout:      20 * time.Millisecond,
		Arrival:      ArrivalConfig{Mode: ArrivalFixed, Rate: 1},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	req := report.Requests[0]
	if !req.Cancelled || req.Status != "canceled" || !strings.Contains(req.Error, "deadline exceeded") {
		t.Fatalf("request = %+v", req)
	}
	if report.Summary.Cancelled != 1 || report.Summary.Successful != 0 {
		t.Fatalf("summary = %+v", report.Summary)
	}
}

func writeTestStreamResponse(t *testing.T, w http.ResponseWriter, chunks []any) {
	t.Helper()
	flusher, ok := w.(http.Flusher)
	if !ok {
		t.Fatal("response writer does not implement http.Flusher")
	}
	w.Header().Set("Content-Type", "text/event-stream")
	for _, chunk := range chunks {
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal stream chunk: %v", err)
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			t.Fatalf("write stream chunk: %v", err)
		}
		flusher.Flush()
	}
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		t.Fatalf("write stream done: %v", err)
	}
	flusher.Flush()
}
