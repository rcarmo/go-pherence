package webui

import (
	"io/fs"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"testing/fstest"
)

func testHandler() http.Handler {
	return NewHandlerFS(fstest.MapFS{
		"index.html":                         {Data: []byte("<!doctype html><title>ui</title>")},
		"favicon.svg":                        {Data: []byte("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>")},
		"_app/immutable/app-abc123.js":       {Data: []byte("console.log('ok')\n")},
		"_app/immutable/app-abc123.css":      {Data: []byte("body{margin:0}\n")},
		"_app/version.json":                  {Data: []byte(`{"ok":true}`)},
		"_app/immutable":                     {Mode: fs.ModeDir},
		"_app/.hidden":                       {Data: []byte("nope")},
		"_app/immutable/app-abc123.js.map":   {Data: []byte("{}")},
		"_app/immutable/.secret/ignored.txt": {Data: []byte("nope")},
	})
}

func serve(h http.Handler, method, target string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, "https://ui.test"+target, nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestRootAndImmutableAssets(t *testing.T) {
	h := testHandler()
	for _, path := range []string{"/", "/index.html"} {
		w := serve(h, http.MethodGet, path)
		if w.Code != http.StatusOK || w.Header().Get("Cache-Control") != "no-cache" || w.Header().Get("X-Content-Type-Options") != "nosniff" || w.Header().Get("X-Frame-Options") != "SAMEORIGIN" || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/html;") || !strings.Contains(w.Body.String(), "<!doctype html>") {
			t.Fatalf("%s: code=%d headers=%v body=%q", path, w.Code, w.Header(), w.Body.String())
		}
	}
	w := serve(h, http.MethodGet, "/_app/immutable/app-abc123.js")
	if w.Code != http.StatusOK || w.Header().Get("Cache-Control") != "public, max-age=31536000, immutable" || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/javascript") || !strings.Contains(w.Body.String(), "console.log") {
		t.Fatalf("immutable js: code=%d headers=%v body=%q", w.Code, w.Header(), w.Body.String())
	}
	w = serve(h, http.MethodGet, "/_app/version.json")
	if w.Code != http.StatusOK || w.Header().Get("Cache-Control") != "no-cache" || !strings.HasPrefix(w.Header().Get("Content-Type"), "application/json") {
		t.Fatalf("version json: code=%d headers=%v body=%q", w.Code, w.Header(), w.Body.String())
	}
	w = serve(h, http.MethodHead, "/favicon.svg")
	if w.Code != http.StatusOK || w.Body.Len() != 0 || w.Header().Get("Cache-Control") != "no-cache" || w.Header().Get("Content-Type") != "image/svg+xml" {
		t.Fatalf("favicon HEAD: code=%d headers=%v body=%q", w.Code, w.Header(), w.Body.String())
	}
}

func TestUnknownDirectoryTraversalAndSourcePaths404(t *testing.T) {
	h := testHandler()
	for _, path := range []string{
		"/v1/chat/completions",
		"/props",
		"/api",
		"/missing",
		"/_app/immutable",
		"/_app/immutable/",
		"/_app/../index.html",
		"/_app/.hidden",
		"/_app/immutable/.secret/ignored.txt",
		"/_app/immutable/app-abc123.js.map",
	} {
		if w := serve(h, http.MethodGet, path); w.Code != http.StatusNotFound {
			t.Fatalf("%s: code=%d want=404", path, w.Code)
		}
	}
}

func TestMethodNotAllowedOnKnownAssets(t *testing.T) {
	h := testHandler()
	w := serve(h, http.MethodPost, "/")
	if w.Code != http.StatusMethodNotAllowed || w.Header().Get("Allow") != "GET, HEAD" {
		t.Fatalf("POST /: code=%d headers=%v", w.Code, w.Header())
	}
	w = serve(h, http.MethodPut, "/_app/immutable/app-abc123.css")
	if w.Code != http.StatusMethodNotAllowed || w.Header().Get("Allow") != "GET, HEAD" {
		t.Fatalf("PUT immutable css: code=%d headers=%v", w.Code, w.Header())
	}
}

// Test the actual embedded output too: a synthetic filesystem alone cannot
// detect a missing bundle or a build pipeline that changed its asset layout.
func TestEmbeddedBuild(t *testing.T) {
	h := NewHandler()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/", nil))
	if w.Code != 200 || !strings.Contains(w.Body.String(), "_app/immutable/") {
		t.Fatalf("missing built app: %d", w.Code)
	}
	matches := regexp.MustCompile(`(?:src|href)="(/_app/[^"?]+)`).FindAllStringSubmatch(w.Body.String(), -1)
	if len(matches) < 2 {
		t.Fatalf("missing script/style references: %v", matches)
	}
	for _, match := range matches {
		res := httptest.NewRecorder()
		h.ServeHTTP(res, httptest.NewRequest(http.MethodGet, match[1], nil))
		if res.Code != 200 || res.Body.Len() == 0 {
			t.Fatalf("asset %s: %d", match[1], res.Code)
		}
	}
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/LICENSE.llama-cpp.txt", nil))
	if w.Code != 200 || !strings.Contains(w.Body.String(), "MIT License") {
		t.Fatal("embedded upstream licence missing")
	}
}
