package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/rcarmo/go-pherence/model/diffusiongemma"
)

type countingDenoiser struct {
	calls  atomic.Int32
	onCall func()
}

func (d *countingDenoiser) Denoise(in diffusiongemma.ForwardInput) (diffusiongemma.ForwardOutput, error) {
	d.calls.Add(1)
	if d.onCall != nil {
		d.onCall()
	}
	return diffusiongemma.MockDenoiser{VocabSize: 8, TokenID: 2}.Denoise(in)
}
func testServer(t *testing.T) (*server, *countingDenoiser) {
	t.Helper()
	m := &diffusiongemma.Model{Shape: diffusiongemma.Shape{CanvasLength: 2, VocabSize: 8}, Denoising: diffusiongemma.DefaultDenoisingConfig()}
	d := &countingDenoiser{}
	e, err := diffusiongemma.NewEngine(m, d)
	if err != nil {
		t.Fatal(err)
	}
	return &server{model: m, engine: e, defaultMaxNew: 1}, d
}
func request(s *server, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(body))
	s.handleCompletions(w, r)
	return w
}

func TestAdmissionRejectsBeforeDenoise(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
	}{
		{"oversize", `{"prompt_ids":[1],"ignored":"` + strings.Repeat("x", maxRequestBytes) + `"}`, 413},
		{"oversize whitespace", `{"prompt_ids":[1]}` + strings.Repeat(" ", maxRequestBytes), 413},
		{"trailing value", `{"prompt_ids":[1]} {}`, 400},
		{"negative", `{"prompt_ids":[1],"max_tokens":-1}`, 400},
		{"max", `{"prompt_ids":[1],"max_tokens":9223372036854775807}`, 400},
		{"id", `{"prompt_ids":[8]}`, 400},
		{"canvas", `{"prompt_ids":[1],"canvas_length":257}`, 400},
		{"nested steps", `{"prompt_ids":[1],"denoising":{"max_denoising_steps":257}}`, 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, d := testServer(t)
			w := request(s, tc.body)
			if w.Code != tc.status {
				t.Fatalf("code=%d body=%s", w.Code, w.Body)
			}
			if d.calls.Load() != 0 {
				t.Fatal("rejected request ran inference")
			}
		})
	}
}
func TestAdmissionSuccessAndBusyRecovery(t *testing.T) {
	s, d := testServer(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	d.onCall = func() { once.Do(func() { close(entered) }); <-release }
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- request(s, `{"prompt_ids":[1]}`) }()
	<-entered
	w := request(s, `{"prompt_ids":[1]}`)
	if w.Code != http.StatusTooManyRequests {
		t.Errorf("busy status=%d", w.Code)
	}
	close(release)
	w = <-done
	if w.Code != 200 {
		t.Fatalf("code=%d %s", w.Code, w.Body)
	}
	d.onCall = nil
	w = request(s, `{"prompt_ids":[1],"stream":true}`)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "[DONE]") {
		t.Fatalf("stream: %d %s", w.Code, w.Body)
	}
}
func TestAdmissionCancelledStopsAtStepAndReleasesOwner(t *testing.T) {
	s, d := testServer(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := httptest.NewRequest("POST", "/", strings.NewReader(`{"prompt_ids":[1]}`)).WithContext(ctx)
	w := httptest.NewRecorder()
	s.handleCompletions(w, r)
	if d.calls.Load() != 0 || w.Code != 408 {
		t.Fatal(w.Code, d.calls.Load())
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	d.onCall = cancel
	r = httptest.NewRequest("POST", "/", strings.NewReader(`{"prompt_ids":[1],"max_tokens":16}`)).WithContext(ctx)
	w = httptest.NewRecorder()
	s.handleCompletions(w, r)
	if d.calls.Load() != 1 || w.Code != 408 {
		t.Fatal(w.Code, d.calls.Load())
	}
	d.onCall = nil
	if w = request(s, `{"prompt_ids":[1]}`); w.Code != 200 {
		t.Fatal(w.Code, w.Body)
	}
}
func TestAdmissionBoundsPromptAndDefaultWork(t *testing.T) {
	s, _ := testServer(t)
	ids := make([]int, maxPromptTokens+1)
	body, _ := json.Marshal(completionRequest{PromptIDs: ids})
	if w := request(s, string(body)); w.Code != 400 {
		t.Fatal(w.Code)
	}
	s.defaultMaxNew = maxGeneratedTokens + 1
	if w := request(s, `{"prompt_ids":[1]}`); w.Code != 400 {
		t.Fatal(w.Code)
	}
}
