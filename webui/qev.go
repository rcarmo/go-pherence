package webui

import (
	"bytes"
	"embed"
	"encoding/json"
	"net/http"
	"strconv"
)

//go:embed qev/index.html
var qevFiles embed.FS

// QEVConfig describes the independently authored decision playground. The
// decision handler remains the only inference API and is registered separately.
type QEVConfig struct {
	ModelID       string
	Backend       string
	Device        string
	ResidentBytes int64
	MaxContexts   int
	MaxFields     int
	MaxCandidates int
	Busy          func() bool
}

// RegisterQEV attaches the standalone playground and its immutable runtime
// metadata to an existing decision-server mux.
func RegisterQEV(mux *http.ServeMux, cfg QEVConfig) {
	if mux == nil {
		panic("webui: nil QEV mux")
	}
	if cfg.ModelID == "" || cfg.Backend == "" || cfg.MaxContexts <= 0 || cfg.MaxFields <= 0 || cfg.MaxCandidates <= 0 {
		panic("webui: invalid QEV config")
	}
	mux.Handle("/qev", qevPageHandler{})
	mux.Handle("/qev/", qevPageHandler{})
	mux.Handle("/qev/v1/status", qevStatusHandler{cfg: cfg})
}

type qevPageHandler struct{}

func (qevPageHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/qev" && r.URL.Path != "/qev/" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}
	body, err := qevFiles.ReadFile("qev/index.html")
	if err != nil {
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'self'")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "SAMEORIGIN")
	if r.Method == http.MethodGet {
		_, _ = bytes.NewReader(body).WriteTo(w)
	}
}

type qevStatusHandler struct{ cfg QEVConfig }

func (h qevStatusHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}
	busy := false
	if h.cfg.Busy != nil {
		busy = h.cfg.Busy()
	}
	payload, err := json.Marshal(map[string]any{
		"object": "decision.status", "model": h.cfg.ModelID, "backend": h.cfg.Backend,
		"device": h.cfg.Device, "resident_bytes": h.cfg.ResidentBytes, "busy": busy,
		"limits": map[string]int{"contexts": h.cfg.MaxContexts, "fields": h.cfg.MaxFields, "candidates_per_field": h.cfg.MaxCandidates},
	})
	if err != nil {
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if r.Method == http.MethodGet {
		_, _ = w.Write(payload)
	}
}
