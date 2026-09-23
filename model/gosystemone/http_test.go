package gosystemone

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/rcarmo/go-pherence/model"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandlerDecisionContract(t *testing.T) {
	h := &Handler{Engine: &Engine{Tokenizer: runeTokenizer{}, Scorer: &fixedScorer{}, BOSToken: 2}, ModelID: "gemma4-12b"}
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/decision", strings.NewReader(`{
		"model":"gemma4-12b",
		"schema":{"urgent":{"type":"boolean","description":"urgent"}},
		"contexts":["one"]
	}`))
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var response Response
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Object != "decision" || response.Model != "gemma4-12b" || response.Created <= 0 || len(response.Results) != 1 {
		t.Fatalf("response=%+v", response)
	}
}

func TestHandlerRejectsMethodBodyModelAndCancellation(t *testing.T) {
	h := &Handler{Engine: &Engine{Tokenizer: runeTokenizer{}, Scorer: &fixedScorer{}, BOSToken: 2}, ModelID: "gemma4-12b"}
	for _, tc := range []struct {
		method string
		body   string
		status int
	}{
		{http.MethodGet, "", http.StatusMethodNotAllowed},
		{http.MethodPost, `{} {}`, http.StatusBadRequest},
		{http.MethodPost, `{"unknown":1}`, http.StatusBadRequest},
		{http.MethodPost, `{"model":"other","schema":{},"contexts":["x"]}`, http.StatusBadRequest},
	} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(tc.method, "/v1/decision", strings.NewReader(tc.body)))
		if w.Code != tc.status {
			t.Fatalf("%s %q status=%d want=%d body=%s", tc.method, tc.body, w.Code, tc.status, w.Body.String())
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/decision", strings.NewReader(`{}`)).WithContext(ctx)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusRequestTimeout {
		t.Fatalf("cancelled status=%d", w.Code)
	}
}

func TestHandlerBusy(t *testing.T) {
	h := &Handler{Engine: &Engine{Tokenizer: runeTokenizer{}, Scorer: &fixedScorer{}, BOSToken: 2}, ModelID: "gemma4-12b"}
	if h.Busy() {
		t.Fatal("idle handler reported busy")
	}
	h.gate.Lock()
	defer h.gate.Unlock()
	if !h.Busy() {
		t.Fatal("locked handler reported idle")
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/decision", strings.NewReader(`{}`)))
	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("status=%d", w.Code)
	}
}

func TestHandlerRejectsUnconfiguredEngine(t *testing.T) {
	w := httptest.NewRecorder()
	(&Handler{}).ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/decision", strings.NewReader(`{"schema":{"x":{"type":"boolean"}},"contexts":["x"]}`)))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

type capacityScorer struct{}

func (capacityScorer) ScoreContext(context.Context, []int, []Branch, bool) ([][]float32, error) {
	return nil, fmt.Errorf("prefill: %w", model.ErrGemma4ContextCapacity)
}
func TestHandlerCapacityReturns422(t *testing.T) {
	h := &Handler{Engine: &Engine{Tokenizer: runeTokenizer{}, Scorer: capacityScorer{}, BOSToken: 2}, ModelID: "fixture"}
	for _, typed := range []bool{false, true} {
		body := `{"schema":{"x":{"type":"boolean","description":"x"}},"contexts":["hello"]}`
		if typed {
			body = `{"state":"hello","questions":{"x":{"type":"noul"}}}`
		}
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
		if typed {
			h.ServeSystemOne(w, r)
		} else {
			h.ServeHTTP(w, r)
		}
		if w.Code != 422 || h.Busy() {
			t.Fatalf("status=%d busy=%v", w.Code, h.Busy())
		}
	}
}
