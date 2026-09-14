package httpapi

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func uiHandler(t *testing.T) *Handler {
	t.Helper()
	store, e := speechjob.Open(filepath.Join(t.TempDir(), "jobs"), speechjob.Limits{MaxJobs: 8, MaxUploadBytes: 1024, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20})
	if e != nil {
		t.Fatal(e)
	}
	h, e := New(Config{Store: store, Token: testToken, Hosts: []string{"speech.test"}, Origin: "https://speech.test", EnableUI: true, MaxUploadBytes: 1024, MaxConcurrentRequests: 2, Profiles: []Profile{{ID: "test", Configuration: []byte(`{"hidden_path":"/private/model"}`), Stages: []speechjob.Stage{textStage("transcript", "fixture")}}}})
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { h.Shutdown(context.Background()); store.Close() })
	return h
}
func TestUIStaticBoundaryAndAuthenticatedProfiles(t *testing.T) {
	h := uiHandler(t)
	for _, path := range []string{"/ui", "/ui/", "/ui/app.js", "/ui/style.css"} {
		r := httptest.NewRequest("GET", "https://speech.test"+path, nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 200 || w.Header().Get("Cache-Control") != "no-store" || !strings.Contains(w.Header().Get("Content-Security-Policy"), "frame-ancestors 'none'") || strings.Contains(w.Header().Get("Content-Security-Policy"), "unsafe-inline") || strings.Contains(w.Body.String(), testToken) || strings.Contains(w.Body.String(), "hidden_path") {
			t.Fatal(path, w.Code, w.Header())
		}
		r.Method = "HEAD"
		w = httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 200 || w.Body.Len() != 0 {
			t.Fatal("HEAD", w)
		}
	}
	r := httptest.NewRequest("GET", "https://speech.test/v1/profiles", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 401 {
		t.Fatal(w)
	}
	r.Header.Set("Authorization", "Bearer "+testToken)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	var result struct {
		Profiles []string `json:"profiles"`
		Limit    int64    `json:"max_upload_bytes"`
	}
	json.Unmarshal(w.Body.Bytes(), &result)
	if w.Code != 200 || len(result.Profiles) != 1 || result.Profiles[0] != "test" || result.Limit != 1024 || strings.Contains(w.Body.String(), "private") {
		t.Fatal(w.Body.String())
	}
	for _, tc := range []struct {
		url    string
		origin string
		site   string
		tls    bool
		status int
	}{
		{"https://speech.test/ui/app.js?x=1", "", "", true, 404},
		{"https://speech.test/ui/../app.js", "", "", true, 404},
		{"https://speech.test/ui/missing", "", "", true, 404},
		{"https://speech.test/ui", "https://evil.test", "", true, 403},
		{"https://speech.test/ui", "", "cross-site", true, 403},
		{"http://speech.test/ui", "", "", false, 421},
	} {
		r = httptest.NewRequest("GET", tc.url, nil)
		if tc.tls {
			r.TLS = &tls.ConnectionState{}
		}
		if tc.origin != "" {
			r.Header.Set("Origin", tc.origin)
		}
		if tc.site != "" {
			r.Header.Set("Sec-Fetch-Site", tc.site)
		}
		w = httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != tc.status {
			t.Fatal(tc, w.Code)
		}
	}
	h.Shutdown(context.Background())
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "https://speech.test/ui", nil))
	if w.Code != 503 {
		t.Fatal("UI admission after shutdown", w)
	}
}
func TestUIOptInAndSourceSafety(t *testing.T) {
	h, s, _ := fixture(t, []speechjob.Stage{textStage("transcript", "x")}, 2)
	if w := request(h, "GET", "/ui", nil); w.Code != 404 {
		t.Fatal(w)
	}
	if w := request(h, "GET", "/v1/profiles", nil); w.Code != 200 || !strings.Contains(w.Body.String(), `"profiles":["test"]`) {
		t.Fatal(w)
	}
	for _, origin := range []string{"", "https://other.test", "https://speech.test?"} {
		_, e := New(Config{Store: s, Token: testToken, Hosts: []string{"speech.test"}, Origin: origin, EnableUI: true, MaxUploadBytes: 1024, MaxConcurrentRequests: 2, Profiles: []Profile{{ID: "test", Configuration: []byte(`{}`), Stages: []speechjob.Stage{textStage("transcript", "x")}}}})
		if e == nil {
			t.Fatal("invalid UI origin", origin)
		}
	}
	for _, bad := range []string{"innerHTML", "localStorage", "sessionStorage", "document.cookie", "eval("} {
		if strings.Contains(uiJS, bad) {
			t.Fatal("unsafe JS primitive", bad)
		}
	}
	if !strings.Contains(uiJS, "crypto.subtle.digest") || !strings.Contains(uiJS, "credentials:'omit'") {
		t.Fatal("missing browser download/auth boundary")
	}
	if b, e := os.ReadFile("ui/index.html"); e != nil || string(b) != uiHTML {
		t.Fatal("embedded source differs", e)
	}
}
