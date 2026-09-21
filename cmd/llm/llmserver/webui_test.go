package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebUIRoutes(t *testing.T) {
	s := &Server{modelID: "fixture", maxCtx: 2048}
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
	// The UI must follow an API-side model switch, not advertise a stale alias.
	s.mu.Lock()
	s.modelID = "switched"
	s.maxCtx = 8192
	s.mu.Unlock()
	w := httptest.NewRecorder()
	routes.ServeHTTP(w, httptest.NewRequest("GET", "/props", nil))
	var props struct {
		ModelAlias string `json:"model_alias"`
		Defaults   struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &props); err != nil {
		t.Fatal(err)
	}
	if props.ModelAlias != "switched" || props.Defaults.NCtx != 8192 {
		t.Fatalf("stale props: %+v", props)
	}
	// UI rejects unsupported fields without inference; existing API remains strict.
	for _, path := range []string{"/v1/chat/completions", "/webui/v1/chat/completions"} {
		w = httptest.NewRecorder()
		routes.ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(`{"model":"switched","messages":[{"role":"user","content":"hi"}],"unsupported":true}`)))
		if w.Code != 400 {
			t.Fatalf("%s: %d %s", path, w.Code, w.Body.String())
		}
	}
}
