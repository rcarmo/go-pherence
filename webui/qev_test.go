package webui

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRegisterQEVPageAndStatus(t *testing.T) {
	mux := http.NewServeMux()
	RegisterQEV(mux, QEVConfig{ModelID: "gemma4-12b", Backend: "nvidia", Device: "RTX 3060", ResidentBytes: 42, MaxContexts: 256, MaxFields: 32, MaxCandidates: 255})
	for _, target := range []string{"/qev", "/qev/"} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, target, nil))
		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "QEV Decision Playground") || w.Header().Get("Content-Security-Policy") == "" || w.Header().Get("Cache-Control") != "no-cache" {
			t.Fatalf("%s: code=%d headers=%v body=%q", target, w.Code, w.Header(), w.Body.String())
		}
	}
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest(http.MethodHead, "/qev/v1/status", nil))
	if w.Code != http.StatusOK || w.Body.Len() != 0 || w.Header().Get("Content-Length") == "" {
		t.Fatalf("HEAD status: code=%d headers=%v body=%q", w.Code, w.Header(), w.Body.String())
	}
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/qev/v1/status", nil))
	var got struct {
		Model         string `json:"model"`
		Backend       string `json:"backend"`
		Device        string `json:"device"`
		ResidentBytes int64  `json:"resident_bytes"`
		Busy          bool   `json:"busy"`
		Limits        struct {
			Contexts int `json:"contexts"`
			Fields   int `json:"fields"`
		} `json:"limits"`
	}
	if w.Code != http.StatusOK || json.Unmarshal(w.Body.Bytes(), &got) != nil || got.Model != "gemma4-12b" || got.Backend != "nvidia" || got.Device != "RTX 3060" || got.ResidentBytes != 42 || got.Busy || got.Limits.Contexts != 256 || got.Limits.Fields != 32 {
		t.Fatalf("status: code=%d body=%s got=%+v", w.Code, w.Body.String(), got)
	}
}

func TestRegisterQEVMethodAndPathBoundaries(t *testing.T) {
	mux := http.NewServeMux()
	RegisterQEV(mux, QEVConfig{ModelID: "m", Backend: "simd", MaxContexts: 1, MaxFields: 1, MaxCandidates: 1})
	for _, target := range []string{"/qev/nope", "/qev/index.html"} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, target, nil))
		if w.Code != http.StatusNotFound {
			t.Fatalf("%s status=%d", target, w.Code)
		}
	}
	for _, target := range []string{"/qev", "/qev/v1/status"} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(http.MethodPost, target, nil))
		if w.Code != http.StatusMethodNotAllowed || w.Header().Get("Allow") != "GET, HEAD" {
			t.Fatalf("%s status=%d headers=%v", target, w.Code, w.Header())
		}
	}
}
