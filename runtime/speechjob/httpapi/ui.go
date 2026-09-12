package httpapi

import (
	_ "embed"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
)

//go:embed ui/index.html
var uiHTML string

//go:embed ui/app.js
var uiJS string

//go:embed ui/style.css
var uiCSS string

// UI assets are the only unauthenticated surface. They are constant embedded
// bytes containing no job/profile/token state. Host, origin, path and request
// admission checks still apply. This does not weaken bearer checks on /v1/*.
func (h *Handler) serveUI(w http.ResponseWriter, r *http.Request) bool {
	if !h.enableUI || (r.URL.Path != "/ui" && !strings.HasPrefix(r.URL.Path, "/ui/")) {
		return false
	}
	if r.URL.RawPath != "" || r.URL.RawQuery != "" || r.URL.ForceQuery || !emptyBody(r) {
		respondError(w, 404, "not_found", nil)
		return true
	}
	u, _ := url.Parse(h.origin)
	if r.Host != u.Host || (u.Scheme == "https") != (r.TLS != nil) {
		respondError(w, 421, "ui_origin_mismatch", nil)
		return true
	}
	origins := r.Header.Values("Origin")
	if len(origins) > 1 || len(origins) == 1 && origins[0] != h.origin {
		respondError(w, 403, "origin_rejected", nil)
		return true
	}
	if site := r.Header.Get("Sec-Fetch-Site"); site != "" && site != "same-origin" && site != "none" {
		respondError(w, 403, "cross_site_rejected", nil)
		return true
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		method(w, "GET, HEAD")
		return true
	}
	var content, mime string
	switch r.URL.Path {
	case "/ui", "/ui/":
		content, mime = uiHTML, "text/html; charset=utf-8"
	case "/ui/app.js":
		content, mime = uiJS, "text/javascript; charset=utf-8"
	case "/ui/style.css":
		content, mime = uiCSS, "text/css; charset=utf-8"
	default:
		respondError(w, 404, "not_found", nil)
		return true
	}
	if !h.admit(false) {
		respondError(w, 503, "unavailable", nil)
		return true
	}
	defer h.release(false)
	w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'")
	w.Header().Set("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Content-Type", mime)
	w.Header().Set("Content-Length", strconv.Itoa(len(content)))
	w.WriteHeader(http.StatusOK)
	if r.Method != http.MethodHead {
		_, _ = io.WriteString(w, content)
	}
	return true
}
func (h *Handler) uiProfiles(w http.ResponseWriter) {
	ids := make([]string, 0, len(h.profiles))
	for id := range h.profiles {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	respond(w, 200, struct {
		Profiles       []string `json:"profiles"`
		MaxUploadBytes int64    `json:"max_upload_bytes"`
	}{ids, h.uploadLimit})
}
