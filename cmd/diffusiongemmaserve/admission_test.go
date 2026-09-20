package main

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatRejectsMalformedOrExcessiveBeforeModel(t *testing.T) {
	for _, tc := range []struct {
		body   string
		status int
	}{
		{`{} {}`, 400}, {`{"messages":[]}`, 400}, {`{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}`, 400},
		{`{"messages":[{"role":"user","content":"hi"}],"max_tokens":4097}`, 400},
		{`{"messages":[{"role":"user","content":"` + strings.Repeat("x", 1<<20) + `"}]}`, 413},
	} {
		s := &server{}
		w := httptest.NewRecorder()
		s.handleChat(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(tc.body)))
		if w.Code != tc.status {
			t.Fatal(w.Code, tc.status)
		}
	}
}
func TestChatBusyDoesNotQueueOrTouchModel(t *testing.T) {
	s := &server{}
	s.mu.Lock()
	defer s.mu.Unlock()
	w := httptest.NewRecorder()
	s.handleChat(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`)))
	if w.Code != 429 {
		t.Fatal(w.Code)
	}
}
