package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebUIRoutes(t *testing.T) {
	s := &server{modelID: "fixture", defaultMaxNew: 3}
	for _, enabled := range []bool{false, true} {
		routes := s.routes(enabled)
		for _, path := range []string{"/", "/props"} {
			w := httptest.NewRecorder()
			routes.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
			want := 404
			if enabled {
				want = 200
			}
			if w.Code != want {
				t.Fatalf("enabled=%v %s: %d", enabled, path, w.Code)
			}
		}
	}
	routes := s.routes(true)
	w := httptest.NewRecorder()
	routes.ServeHTTP(w, httptest.NewRequest("GET", "/props", nil))
	var props struct {
		Defaults struct {
			Params struct {
				MaxTokens int `json:"max_tokens"`
			} `json:"params"`
		} `json:"default_generation_settings"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &props); err != nil {
		t.Fatal(err)
	}
	if props.Defaults.Params.MaxTokens != 3 {
		t.Fatalf("props: %s", w.Body.String())
	}
	// These fail before tokenization or model entry.
	for _, path := range []string{"/v1/chat/completions", "/webui/v1/chat/completions"} {
		w = httptest.NewRecorder()
		routes.ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(`{"model":"fixture","messages":[{"role":"user","content":"hi"}],"unsupported":true}`)))
		if w.Code != 400 {
			t.Fatalf("%s: %d %s", path, w.Code, w.Body.String())
		}
	}
}
