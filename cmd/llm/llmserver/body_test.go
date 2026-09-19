package main

import (
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
