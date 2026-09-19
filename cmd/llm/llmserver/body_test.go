package main

import (
	"context"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatBodyBoundaryRejectsBeforeRuntimeLookup(t *testing.T) {
	for _, tc := range []struct {
		body   string
		status int
	}{
		{`{} {}`, 400},
		{`{}` + strings.Repeat(" ", 1<<20), 413},
		{`{"unknown":1}`, 400},
	} {
		s := &Server{}
		w := httptest.NewRecorder()
		r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(tc.body))
		s.handleChatCompletions(w, r)
		if w.Code != tc.status {
			t.Fatalf("code=%d want=%d", w.Code, tc.status)
		}
	}
}

func TestAdmissionRejectsBusyCancelledAndUnboundedGeneration(t *testing.T) {
	s := &Server{}
	s.admissionMu.Lock()
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/", strings.NewReader(`{}`))
	s.handleChatCompletions(w, r)
	s.admissionMu.Unlock()
	if w.Code != 429 {
		t.Fatal(w.Code)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w = httptest.NewRecorder()
	r = httptest.NewRequest("POST", "/", strings.NewReader(`{}`)).WithContext(ctx)
	s.handleChatCompletions(w, r)
	if w.Code != 408 {
		t.Fatal(w.Code)
	}
	w = httptest.NewRecorder()
	r = httptest.NewRequest("POST", "/", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"max_tokens":4097}`))
	s.handleChatCompletions(w, r)
	if w.Code != 400 {
		t.Fatal(w.Code)
	}
	s.inferMu.Lock()
	_, _, err := s.snapshotRuntime("")
	s.inferMu.Unlock()
	if !errors.Is(err, errInferenceBusy) {
		t.Fatal(err)
	}
	// No model/factory call is allowed for an already-cancelled request.
	if _, err := s.generate(ctx, serverRuntime{}, nil, 1, nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
