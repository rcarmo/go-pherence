package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestOptionsRejectUnsafePathsAndBackend(t *testing.T) {
	if _, err := parseOptions(nil); err == nil {
		t.Fatal("missing required paths accepted")
	}
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"--speechjobserve", file, "--config", file},
		{"--speechjobserve", "/relative", "--config", file},
		{"--speechjobserve", file, "--config", file, "--backend", "http://example.com:80"},
		{"--speechjobserve", file, "--config", file, "--hosts", "user@bad"},
	} {
		if _, err := parseOptions(args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestNoLoginBoundaryInjectsInternalCredential(t *testing.T) {
	const token = "internal-token-that-is-long-enough-for-tests"
	var mu sync.Mutex
	var gotAuth, gotOrigin, gotHost string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotAuth, gotOrigin, gotHost = r.Header.Get("Authorization"), r.Header.Get("Origin"), r.Host
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"jobs":[],"next":""}`)
	}))
	defer backend.Close()
	u, _ := url.Parse(backend.URL)
	h := newHandler(map[string]bool{"sigma.local:8093": true}, u, token)

	static := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "http://sigma.local:8093/", nil)
	h.ServeHTTP(static, r)
	if static.Code != http.StatusOK || !strings.Contains(static.Body.String(), "Transcribe") || static.Header().Get("Content-Security-Policy") == "" {
		t.Fatal(static.Code, static.Header(), static.Body.String())
	}

	api := httptest.NewRecorder()
	r = httptest.NewRequest(http.MethodGet, "http://sigma.local:8093/api/v1/jobs", nil)
	r.Header.Set("Authorization", "Bearer browser-secret")
	r.Header.Set("Origin", "http://sigma.local:8093")
	h.ServeHTTP(api, r)
	if api.Code != http.StatusOK {
		t.Fatal(api.Code, api.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if gotAuth != "Bearer "+token || gotOrigin != "" || gotHost != u.Host {
		t.Fatal(gotAuth, gotOrigin, gotHost)
	}
}

func TestNoLoginBoundaryRejectsHostOriginAndCrossSite(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Fatal("rejected request reached backend") }))
	defer backend.Close()
	u, _ := url.Parse(backend.URL)
	h := newHandler(map[string]bool{"sigma.local:8093": true}, u, strings.Repeat("x", 32))
	cases := []struct {
		host, origin, site string
		status             int
	}{
		{"evil.test", "", "", http.StatusMisdirectedRequest},
		{"sigma.local:8093", "http://evil.test", "", http.StatusForbidden},
		{"sigma.local:8093", "", "cross-site", http.StatusForbidden},
	}
	for _, tc := range cases {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodGet, "http://"+tc.host+"/api/v1/jobs", nil)
		if tc.origin != "" {
			r.Header.Set("Origin", tc.origin)
		}
		if tc.site != "" {
			r.Header.Set("Sec-Fetch-Site", tc.site)
		}
		h.ServeHTTP(w, r)
		if w.Code != tc.status {
			t.Fatal(tc, w.Code)
		}
	}
}

func TestRecoverInterruptedOnly(t *testing.T) {
	const token = "internal-token-that-is-long-enough-for-tests"
	var retried []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+token {
			t.Fatal("missing internal token")
		}
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/queue":
			_ = json.NewEncoder(w).Encode(map[string]any{"entries": []map[string]any{
				{"job_id": strings.Repeat("a", 32), "status": "interrupted"},
				{"job_id": strings.Repeat("b", 32), "status": "failed"},
				{"job_id": strings.Repeat("c", 32), "status": "pending"},
			}})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/retry-queued"):
			retried = append(retried, strings.Split(r.URL.Path, "/")[3])
			w.WriteHeader(http.StatusAccepted)
			_, _ = io.WriteString(w, `{}`)
		default:
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
	}))
	defer backend.Close()
	u, _ := url.Parse(backend.URL)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := recoverInterrupted(ctx, u, token); err != nil {
		t.Fatal(err)
	}
	if len(retried) != 1 || retried[0] != strings.Repeat("a", 32) {
		t.Fatal(retried)
	}
}

func TestReconcileTerminalReleasesSuccessAndDeletesCancellation(t *testing.T) {
	const token = "internal-token-that-is-long-enough-for-tests"
	var actions []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+token {
			t.Fatal("missing token")
		}
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet && r.URL.Path == "/v1/queue" {
			_ = json.NewEncoder(w).Encode(map[string]any{"entries": []map[string]any{
				{"job_id": strings.Repeat("a", 32), "status": "succeeded"},
				{"job_id": strings.Repeat("b", 32), "status": "cancelled"},
				{"job_id": strings.Repeat("c", 32), "status": "failed"},
			}})
			return
		}
		actions = append(actions, r.Method+" "+r.URL.Path)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer backend.Close()
	u, _ := url.Parse(backend.URL)
	if err := reconcileTerminal(context.Background(), u, token); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"POST /v1/jobs/" + strings.Repeat("a", 32) + "/release-media",
		"DELETE /v1/jobs/" + strings.Repeat("a", 32) + "/queue",
		"DELETE /v1/jobs/" + strings.Repeat("b", 32),
	}
	if strings.Join(actions, "\n") != strings.Join(want, "\n") {
		t.Fatalf("actions=%q", actions)
	}
}

func TestBrowserOffersExplicitSpeakerProfileSelection(t *testing.T) {
	html, err := web.ReadFile("web/index.html")
	if err != nil {
		t.Fatal(err)
	}
	js, err := web.ReadFile("web/app.js")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(html), `id="diarize" type="checkbox"`) || !strings.Contains(string(js), `($('diarize').checked?'diar':'asr')+'-'+language+'-'+extension`) {
		t.Fatal("browser does not select separate ASR and diarization profiles")
	}
}

func TestRandomToken(t *testing.T) {
	a, err := randomToken()
	if err != nil || len(a) != 64 {
		t.Fatal(len(a), err)
	}
	b, err := randomToken()
	if err != nil || a == b {
		t.Fatal("token generation repeated", err)
	}
}
