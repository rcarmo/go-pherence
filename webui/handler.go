package webui

import (
	"bytes"
	"embed"
	"io/fs"
	"mime"
	"net/http"
	"path"
	"strings"
)

//go:embed all:dist
var dist embed.FS

// NewHandler serves the embedded dist/ tree from the package root paths.
func NewHandler() http.Handler {
	sub, err := fs.Sub(dist, "dist")
	if err != nil {
		panic("webui: dist: " + err.Error())
	}
	return NewHandlerFS(sub)
}

// NewHandlerFS serves a built dist/ tree from any fs.FS.
func NewHandlerFS(fsys fs.FS) http.Handler { return &handler{fsys: fsys} }

type handler struct{ fsys fs.FS }

type asset struct {
	name  string
	cache string
}

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	a, ok := resolveAsset(r.URL.Path)
	if !ok {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}
	info, err := fs.Stat(h.fsys, a.name)
	if err != nil || info.IsDir() {
		http.NotFound(w, r)
		return
	}
	body, err := fs.ReadFile(h.fsys, a.name)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	if ctype := contentType(a.name); ctype != "" {
		w.Header().Set("Content-Type", ctype)
	}
	w.Header().Set("Cache-Control", a.cache)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "SAMEORIGIN")
	http.ServeContent(w, r, info.Name(), info.ModTime(), bytes.NewReader(body))
}

func resolveAsset(urlPath string) (asset, bool) {
	switch urlPath {
	case "/", "/index.html":
		return asset{name: "index.html", cache: "no-cache"}, true
	case "/favicon.svg":
		return asset{name: "favicon.svg", cache: "no-cache"}, true
	case "/LICENSE.llama-cpp.txt":
		return asset{name: "LICENSE.llama-cpp.txt", cache: "no-cache"}, true
	}
	if !strings.HasPrefix(urlPath, "/_app/") || strings.HasSuffix(urlPath, "/") || path.Clean(urlPath) != urlPath {
		return asset{}, false
	}
	name := strings.TrimPrefix(urlPath, "/")
	if !safeAssetPath(name) || strings.HasSuffix(strings.ToLower(name), ".map") {
		return asset{}, false
	}
	cache := "no-cache"
	if strings.HasPrefix(name, "_app/immutable/") {
		cache = "public, max-age=31536000, immutable"
	}
	return asset{name: name, cache: cache}, true
}

func safeAssetPath(name string) bool {
	if !fs.ValidPath(name) {
		return false
	}
	for _, part := range strings.Split(name, "/") {
		if part == "" || strings.HasPrefix(part, ".") {
			return false
		}
	}
	return true
}

func contentType(name string) string {
	ext := strings.ToLower(path.Ext(name))
	switch ext {
	case ".html":
		return "text/html; charset=utf-8"
	case ".txt":
		return "text/plain; charset=utf-8"
	case ".css":
		return "text/css; charset=utf-8"
	case ".js", ".mjs":
		return "text/javascript; charset=utf-8"
	case ".json":
		return "application/json"
	case ".svg":
		return "image/svg+xml"
	case ".wasm":
		return "application/wasm"
	case ".woff":
		return "font/woff"
	case ".woff2":
		return "font/woff2"
	}
	return mime.TypeByExtension(ext)
}
