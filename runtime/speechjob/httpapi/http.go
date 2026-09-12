// Package httpapi exposes an opt-in, single-tenant HTTP boundary for speechjob.
// It opens no listener, loads no models and runs no automatic/background queue.
package httpapi

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

// Profile is administrator-supplied immutable stage code and semantic settings.
// The handler copies slices/config bytes, not closure-owned models. Callers keep
// models/backend flags immutable and enforce external shared compute admission.
// The stored identity includes ID, exact configuration bytes and stage versions.
type Profile struct {
	ID            string
	Configuration []byte
	Stages        []speechjob.Stage
}

// Config describes one private store and one trusted tenant. Token must have at
// least 32 printable ASCII bytes of caller-generated cryptographic randomness;
// length alone does not prove entropy. TLS, network ACLs, server timeouts and
// request-header limits are provided by the embedding server/reverse proxy.
// Hosts are exact r.Host values, including port. Forwarded headers are ignored.
// Origin, when present, must exactly match Origin; empty config rejects browsers.
// No cookies, CORS preflight or token-in-query authentication is supported.
type Config struct {
	Store                 *speechjob.Store
	Profiles              []Profile
	Token                 string
	Hosts                 []string
	Origin                string
	EnableUI              bool // Opt-in static browser UI; requires exact same Origin/Host.
	MaxUploadBytes        int64
	MaxConcurrentRequests int
}
type boundProfile struct {
	id            string
	configuration []byte
	stages        []speechjob.Stage
}
type Handler struct {
	store         *speechjob.Store
	token         [32]byte
	hosts         map[string]bool
	origin        string
	enableUI      bool
	profiles      map[string]boundProfile
	byConfig      map[string]boundProfile
	uploadLimit   int64
	slots         chan struct{}
	control       chan struct{}
	mutation      chan struct{}
	ctx           context.Context
	cancel        context.CancelFunc
	mu            sync.Mutex
	closed        bool
	active        int
	drained       chan struct{}
	runningID     string
	runningCancel context.CancelFunc
}

// New validates configuration without modifying the store or loading models.
// The store must be exclusively owned by this handler while requests run. The
// caller closes it only after Shutdown succeeds. No default profile is inferred.
func New(cfg Config) (*Handler, error) {
	if cfg.Store == nil || len(cfg.Profiles) < 1 || len(cfg.Profiles) > 32 || len(cfg.Hosts) < 1 || len(cfg.Hosts) > 16 || len(cfg.Token) < 32 || len(cfg.Token) > 256 || cfg.MaxUploadBytes < 1 || cfg.MaxUploadBytes > 512<<20 || cfg.MaxConcurrentRequests < 1 || cfg.MaxConcurrentRequests > 64 {
		return nil, fmt.Errorf("invalid HTTP job configuration")
	}
	for _, b := range []byte(cfg.Token) {
		if b < 33 || b > 126 {
			return nil, fmt.Errorf("invalid bearer token encoding")
		}
	}
	h := &Handler{store: cfg.Store, token: sha256.Sum256([]byte(cfg.Token)), hosts: map[string]bool{}, origin: cfg.Origin, enableUI: cfg.EnableUI, profiles: map[string]boundProfile{}, byConfig: map[string]boundProfile{}, uploadLimit: cfg.MaxUploadBytes, slots: make(chan struct{}, cfg.MaxConcurrentRequests), control: make(chan struct{}, 1), mutation: make(chan struct{}, 1), drained: make(chan struct{})}
	for _, host := range cfg.Hosts {
		u, e := url.Parse("http://" + host)
		if e != nil || host == "" || strings.ContainsAny(host, " /\\?#@\t\r\n") || u.Host != host || u.Hostname() == "" || h.hosts[host] {
			return nil, fmt.Errorf("invalid HTTP host allowlist")
		}
		h.hosts[host] = true
	}
	if cfg.Origin != "" {
		u, e := url.Parse(cfg.Origin)
		if e != nil || u == nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.String() != cfg.Origin {
			return nil, fmt.Errorf("invalid HTTP origin")
		}
	}
	if cfg.EnableUI {
		u, e := url.Parse(cfg.Origin)
		if e != nil || u == nil || u.ForceQuery || cfg.MaxConcurrentRequests < 2 || !h.hosts[u.Host] {
			return nil, fmt.Errorf("browser UI requires an allowed exact origin")
		}
	}
	for _, p := range cfg.Profiles {
		if !slug(p.ID) || h.profiles[p.ID].id != "" || len(p.Configuration) < 2 || len(p.Configuration) > 16<<10 || !utf8.Valid(p.Configuration) || !json.Valid(p.Configuration) || !strings.HasPrefix(strings.TrimSpace(string(p.Configuration)), "{") || len(p.Stages) < 1 || len(p.Stages) > 64 {
			return nil, fmt.Errorf("invalid HTTP profile")
		}
		type stageIdentity struct{ Name, Version string }
		identities := []stageIdentity{}
		names := map[string]bool{}
		for _, s := range p.Stages {
			if !slug(s.Name) || !digest(s.Version) || s.Run == nil || names[s.Name] {
				return nil, fmt.Errorf("invalid HTTP profile stage")
			}
			names[s.Name] = true
			identities = append(identities, stageIdentity{s.Name, s.Version})
		}
		// Config is a string so whitespace and original bytes remain part of identity.
		configuration, e := json.Marshal(struct {
			Schema          int
			Profile, Config string
			Stages          []stageIdentity
		}{1, p.ID, string(p.Configuration), identities})
		if e != nil || len(configuration) > 16<<10 {
			return nil, fmt.Errorf("HTTP profile identity exceeds store bound")
		}
		bound := boundProfile{p.ID, configuration, append([]speechjob.Stage{}, p.Stages...)}
		h.profiles[p.ID] = bound
		h.byConfig[string(configuration)] = bound
	}
	h.ctx, h.cancel = context.WithCancel(context.Background())
	return h, nil
}
func slug(s string) bool {
	if len(s) < 1 || len(s) > 48 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-') {
			return false
		}
	}
	return true
}
func digest(s string) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == 32 && strings.ToLower(s) == s
}
func jobID(s string) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == 16 && strings.ToLower(s) == s
}

// Shutdown refuses new requests, cancels admitted requests and waits for all
// owned handlers/stage callbacks to drain. A context timeout does NOT mean work
// stopped; retry Shutdown and keep store/model resources alive until success.
// It does not close the caller-owned store or HTTP listener.
func (h *Handler) Shutdown(ctx context.Context) error {
	h.mu.Lock()
	if !h.closed {
		h.closed = true
		h.cancel()
		if h.active == 0 {
			close(h.drained)
		}
	}
	done := h.drained
	h.mu.Unlock()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (h *Handler) admit(control bool) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return false
	}
	slot := h.slots
	if control {
		slot = h.control
	}
	select {
	case slot <- struct{}{}:
		h.active++
		return true
	default:
		return false
	}
}
func (h *Handler) release(control bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if control {
		<-h.control
	} else {
		<-h.slots
	}
	h.active--
	if h.closed && h.active == 0 {
		close(h.drained)
	}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'; sandbox")
	if !h.hosts[r.Host] {
		respondError(w, 421, "host_rejected", nil)
		return
	}
	if h.serveUI(w, r) {
		return
	}
	if len(r.Header.Values("Authorization")) != 1 {
		unauthorized(w)
		return
	}
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") || len(auth) > 263 {
		unauthorized(w)
		return
	}
	supplied := sha256.Sum256([]byte(strings.TrimPrefix(auth, "Bearer ")))
	if subtle.ConstantTimeCompare(supplied[:], h.token[:]) != 1 {
		unauthorized(w)
		return
	}
	origins := r.Header.Values("Origin")
	if len(origins) > 1 || len(origins) == 1 && (h.origin == "" || origins[0] != h.origin) {
		respondError(w, 403, "origin_rejected", nil)
		return
	}
	if site := r.Header.Get("Sec-Fetch-Site"); site != "" && site != "same-origin" && site != "none" {
		respondError(w, 403, "cross_site_rejected", nil)
		return
	}
	// Reserve one bounded control slot so a long synchronous run cannot starve
	// its cancel endpoint even when all ordinary request slots are occupied.
	parts := strings.Split(r.URL.Path, "/")
	control := r.Method == http.MethodPost && r.URL.RawPath == "" && len(parts) == 5 && parts[1] == "v1" && parts[2] == "jobs" && jobID(parts[3]) && parts[4] == "cancel"
	if !h.admit(control) {
		respondError(w, 503, "unavailable", nil)
		return
	}
	defer h.release(control)
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	stop := context.AfterFunc(h.ctx, cancel)
	defer stop()
	r = r.WithContext(ctx)
	if e := ctx.Err(); e != nil {
		respondError(w, 409, "cancelled", nil)
		return
	}
	// Canonical routing only: reject escaped separators, aliases, trailing slashes
	// and dot components instead of redirects which could change mutation methods.
	if r.URL.RawPath != "" || strings.Contains(r.URL.Path, "//") || strings.HasSuffix(r.URL.Path, "/") {
		respondError(w, 404, "not_found", nil)
		return
	}
	path := strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")
	query, e := url.ParseQuery(r.URL.RawQuery)
	if e != nil {
		respondError(w, 400, "invalid_query", nil)
		return
	}
	if !(len(path) == 2 && path[0] == "v1" && path[1] == "jobs" && r.Method == http.MethodPost) && !emptyBody(r) {
		respondError(w, 400, "unexpected_body", nil)
		return
	}
	if len(path) == 2 && path[0] == "v1" && path[1] == "jobs" {
		switch r.Method {
		case http.MethodPost:
			if !onlyQuery(query, "profile", "name") || len(query["profile"]) != 1 || len(query["name"]) != 1 {
				respondError(w, 400, "invalid_query", nil)
				return
			}
			h.create(w, r, query.Get("profile"), query.Get("name"))
		case http.MethodGet:
			if !onlyQuery(query, "after") || query.Get("after") != "" && !jobID(query.Get("after")) {
				respondError(w, 400, "invalid_query", nil)
				return
			}
			h.list(w, r, query.Get("after"))
		default:
			method(w, "GET, POST")
		}
		return
	}
	if len(query) != 0 {
		respondError(w, 400, "invalid_query", nil)
		return
	}
	if len(path) == 2 && path[0] == "v1" && path[1] == "profiles" && h.enableUI {
		if r.Method != http.MethodGet {
			method(w, "GET")
			return
		}
		h.uiProfiles(w)
		return
	}
	if len(path) == 2 && path[0] == "v1" && path[1] == "inventory" {
		if r.Method != http.MethodGet {
			method(w, "GET")
			return
		}
		inventory, e := h.store.Inventory()
		if e != nil {
			h.failure(w, e, nil)
			return
		}
		jobs := make([]RetainedJob, 0, len(inventory))
		for _, j := range inventory {
			jobs = append(jobs, RetainedJob{ID: j.ID, Bytes: j.Bytes, ManifestPresent: j.ManifestPresent, Deleting: j.Deleting, Corrupt: j.Corrupt})
		}
		respond(w, 200, struct {
			Jobs []RetainedJob `json:"jobs"`
		}{jobs})
		return
	}
	if len(path) < 3 || len(path) > 5 || path[0] != "v1" || path[1] != "jobs" || !jobID(path[2]) {
		respondError(w, 404, "not_found", nil)
		return
	}
	id := path[2]
	if len(path) == 3 {
		switch r.Method {
		case http.MethodGet:
			m, e := h.store.Get(id)
			if e != nil {
				h.failure(w, e, nil)
				return
			}
			respond(w, 200, h.view(m))
		case http.MethodDelete:
			h.delete(w, r, id)
		default:
			method(w, "GET, DELETE")
		}
		return
	}
	if len(path) == 4 {
		switch path[3] {
		case "run":
			if r.Method != http.MethodPost {
				method(w, "POST")
				return
			}
			h.run(w, r, id)
			return
		case "cancel":
			if r.Method != http.MethodPost {
				method(w, "POST")
				return
			}
			h.cancelRun(w, r, id)
			return
		}
	}
	if len(path) == 5 && path[3] == "artifacts" {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			method(w, "GET, HEAD")
			return
		}
		h.download(w, r, id, path[4])
		return
	}
	respondError(w, 404, "not_found", nil)
}
func onlyQuery(q url.Values, keys ...string) bool {
	for k, v := range q {
		found := false
		for _, key := range keys {
			if k == key {
				found = true
			}
		}
		if !found || len(v) != 1 {
			return false
		}
	}
	return true
}
func emptyBody(r *http.Request) bool {
	return r.ContentLength == 0 && len(r.TransferEncoding) == 0 && r.Header.Get("Content-Encoding") == ""
}
func (h *Handler) mutate(w http.ResponseWriter) bool {
	select {
	case h.mutation <- struct{}{}:
		return true
	default:
		respondError(w, 409, "busy", nil)
		return false
	}
}
func (h *Handler) create(w http.ResponseWriter, r *http.Request, profile, name string) {
	p, ok := h.profiles[profile]
	if !ok {
		respondError(w, 400, "unknown_profile", nil)
		return
	}
	if len(name) < 1 || len(name) > 256 || !utf8.ValidString(name) || strings.ContainsAny(name, "\r\n\x00") {
		respondError(w, 400, "invalid_name", nil)
		return
	}
	mediaType, params, e := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if e != nil || mediaType != "application/octet-stream" || len(params) != 0 || r.Header.Get("Content-Encoding") != "" {
		respondError(w, 415, "unsupported_content_type", nil)
		return
	}
	if r.ContentLength == 0 {
		respondError(w, 400, "empty_upload", nil)
		return
	}
	if r.ContentLength > h.uploadLimit {
		respondError(w, 413, "upload_too_large", nil)
		return
	}
	if !h.mutate(w) {
		return
	}
	defer func() { <-h.mutation }()
	limited := http.MaxBytesReader(w, r.Body, h.uploadLimit)
	// net/http request bodies support Close concurrently with Read. Join this
	// callback before releasing handler admission; no owned closer outlives us.
	var once sync.Once
	closeBody := func() { once.Do(func() { _ = limited.Close() }) }
	closed := make(chan struct{})
	stop := context.AfterFunc(r.Context(), func() { defer close(closed); closeBody() })
	defer func() {
		if !stop() {
			<-closed
		}
		closeBody()
	}()
	m, e := h.store.Create(r.Context(), name, p.configuration, limited)
	if e != nil {
		if r.Context().Err() != nil && !errors.Is(e, speechjob.ErrPersistence) {
			e = r.Context().Err()
		}
		h.failure(w, e, nil)
		return
	}
	w.Header().Set("Location", "/v1/jobs/"+m.ID)
	respond(w, 201, h.view(m))
}
func (h *Handler) list(w http.ResponseWriter, r *http.Request, after string) {
	jobs, next, e := h.store.ListPage(r.Context(), after, 100)
	if e != nil {
		h.failure(w, e, nil)
		return
	}
	views := make([]Job, 0, len(jobs))
	for _, m := range jobs {
		if e = r.Context().Err(); e != nil {
			h.failure(w, e, nil)
			return
		}
		views = append(views, h.view(m))
	}
	respond(w, 200, struct {
		Jobs []Job  `json:"jobs"`
		Next string `json:"next,omitempty"`
	}{views, next})
}
func (h *Handler) delete(w http.ResponseWriter, r *http.Request, id string) {
	if !emptyBody(r) {
		respondError(w, 400, "unexpected_body", nil)
		return
	}
	if !h.mutate(w) {
		return
	}
	defer func() { <-h.mutation }()
	if e := h.store.Delete(r.Context(), id); e != nil {
		h.failure(w, e, nil)
		return
	}
	w.WriteHeader(204)
}
func (h *Handler) run(w http.ResponseWriter, r *http.Request, id string) {
	if !emptyBody(r) {
		respondError(w, 400, "unexpected_body", nil)
		return
	}
	if !h.mutate(w) {
		return
	}
	defer func() { <-h.mutation }()
	m, e := h.store.Get(id)
	if e != nil {
		h.failure(w, e, nil)
		return
	}
	p, ok := h.byConfig[m.Configuration]
	if !ok {
		respondError(w, 409, "profile_changed", nil)
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	h.mu.Lock()
	h.runningID = id
	h.runningCancel = cancel
	h.mu.Unlock()
	defer func() { h.mu.Lock(); h.runningID = ""; h.runningCancel = nil; h.mu.Unlock() }()
	_, e = h.store.Run(ctx, id, p.configuration, p.stages, nil)
	// Return only a fresh persisted snapshot, never the possibly uncertain proposed
	// manifest returned with ErrPersistence. Do not echo callbacks' raw errors.
	stored, readErr := h.store.Get(id)
	if e != nil {
		var view *Job
		if readErr == nil {
			v := h.view(stored)
			view = &v
		}
		h.failure(w, e, view)
		return
	}
	if readErr != nil {
		h.failure(w, readErr, nil)
		return
	}
	respond(w, 200, h.view(stored))
}
func (h *Handler) cancelRun(w http.ResponseWriter, r *http.Request, id string) {
	if !emptyBody(r) {
		respondError(w, 400, "unexpected_body", nil)
		return
	}
	h.mu.Lock()
	accepted := h.runningID == id && h.runningCancel != nil
	if accepted {
		h.runningCancel()
	}
	h.mu.Unlock()
	if !accepted {
		respondError(w, 409, "not_running", nil)
		return
	}
	// A signal acknowledgement is not a durable cancelled-state acknowledgement.
	respond(w, 202, struct {
		ID                    string `json:"id"`
		CancellationRequested bool   `json:"cancellation_requested"`
	}{id, true})
}

// RetainedJob deliberately exports only bounded inventory metadata, never raw
// manifests/configuration, local paths or decoder/intermediate artifact names.
type RetainedJob struct {
	ID              string `json:"id"`
	Bytes           int64  `json:"bytes"`
	ManifestPresent bool   `json:"manifest_present"`
	Deleting        bool   `json:"deleting"`
	Corrupt         bool   `json:"corrupt"`
}

type Artifact struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}
type Job struct {
	ID               string           `json:"id"`
	Name             string           `json:"name"`
	Profile          string           `json:"profile"`
	ProfileAvailable bool             `json:"profile_available"`
	Status           speechjob.Status `json:"status"`
	Attempts         int              `json:"attempts"`
	ActiveStage      string           `json:"active_stage,omitempty"`
	InputBytes       int64            `json:"input_bytes"`
	Created          time.Time        `json:"created"`
	Updated          time.Time        `json:"updated"`
	Artifacts        []Artifact       `json:"artifacts"`
}

var downloadable = map[string]string{"transcript": "application/json", "vtt": "text/vtt; charset=utf-8", "speaker-transcript": "application/json", "speaker-vtt": "text/vtt; charset=utf-8"}

func (h *Handler) view(m speechjob.Manifest) Job {
	j := Job{ID: m.ID, Name: m.Name, Profile: h.byConfig[m.Configuration].id, ProfileAvailable: h.byConfig[m.Configuration].id != "", Status: m.Status, Attempts: m.Attempts, ActiveStage: m.ActiveStage, InputBytes: m.Input.Bytes, Created: m.Created, Updated: m.Updated, Artifacts: []Artifact{}}
	for _, cp := range m.Checkpoints {
		if _, ok := downloadable[cp.Stage]; ok && cp.Blob.Bytes <= 16<<20 {
			j.Artifacts = append(j.Artifacts, Artifact{cp.Stage, cp.Blob.Bytes, cp.Blob.SHA256})
		}
	}
	return j
}
func (h *Handler) download(w http.ResponseWriter, r *http.Request, id, name string) {
	contentType, ok := downloadable[name]
	if !ok {
		respondError(w, 404, "not_found", nil)
		return
	}
	if r.Header.Get("Range") != "" {
		respondError(w, 416, "range_unsupported", nil)
		return
	}
	m, e := h.store.Get(id)
	if e != nil {
		h.failure(w, e, nil)
		return
	}
	var cp *speechjob.Checkpoint
	for _, c := range m.Checkpoints {
		if c.Stage == name {
			copy := c
			cp = &copy
			break
		}
	}
	if cp == nil {
		respondError(w, 404, "not_found", nil)
		return
	}
	if cp.Blob.Bytes > 16<<20 {
		respondError(w, 413, "artifact_too_large", nil)
		return
	}
	reader, e := h.store.OpenCheckpoint(r.Context(), id, name)
	if e != nil {
		h.failure(w, e, nil)
		return
	}
	defer reader.Close()
	extension := ".json"
	if strings.Contains(name, "vtt") {
		extension = ".vtt"
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Disposition", `attachment; filename="`+id+"-"+name+extension+`"`)
	w.Header().Set("Content-Length", fmt.Sprint(cp.Blob.Bytes))
	w.Header().Set("ETag", `"sha256-`+cp.Blob.SHA256+`"`)
	if r.Method == http.MethodHead {
		w.WriteHeader(200)
		return
	}
	// Verified handle; external same-UID payload mutation is outside store trust.
	// Check request cancellation between bounded chunks; blocking network writes
	// need the embedding server's deadlines. Never append JSON after a stream error.
	buf := make([]byte, 32<<10)
	remaining := cp.Blob.Bytes
	for remaining > 0 {
		if r.Context().Err() != nil {
			panic(http.ErrAbortHandler)
		}
		n, e := io.ReadFull(reader, buf[:min(int64(len(buf)), remaining)])
		if e != nil {
			panic(http.ErrAbortHandler)
		}
		written, e := w.Write(buf[:n])
		if e != nil || written != n {
			panic(http.ErrAbortHandler)
		}
		remaining -= int64(n)
	}
}
func unauthorized(w http.ResponseWriter) {
	w.Header().Set("WWW-Authenticate", `Bearer realm="speech-jobs"`)
	respondError(w, 401, "unauthorized", nil)
}
func method(w http.ResponseWriter, allow string) {
	w.Header().Set("Allow", allow)
	respondError(w, 405, "method_not_allowed", nil)
}
func respond(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func respondError(w http.ResponseWriter, status int, code string, job *Job) {
	respond(w, status, struct {
		Error string `json:"error"`
		Job   *Job   `json:"job,omitempty"`
	}{code, job})
}
func (h *Handler) failure(w http.ResponseWriter, e error, job *Job) {
	var tooLarge *http.MaxBytesError
	switch {
	case errors.Is(e, speechjob.ErrPersistence):
		respondError(w, 503, "persistence_uncertain", job)
	case errors.Is(e, speechjob.ErrBusy):
		respondError(w, 409, "busy", job)
	case errors.Is(e, speechjob.ErrLimit) || errors.As(e, &tooLarge):
		respondError(w, 413, "limit_exceeded", job)
	case errors.Is(e, speechjob.ErrConfiguration):
		respondError(w, 409, "profile_changed", job)
	case errors.Is(e, os.ErrNotExist):
		respondError(w, 404, "not_found", job)
	case errors.Is(e, context.Canceled) || errors.Is(e, context.DeadlineExceeded):
		// 408 can cause user agents to retry even a POST on a fresh connection.
		// Application cancellation is an explicit conflict, never replay advice.
		respondError(w, 409, "cancelled", job)
	case errors.Is(e, speechjob.ErrClosed):
		respondError(w, 503, "unavailable", job)
	case errors.Is(e, speechjob.ErrCorrupt):
		respondError(w, 409, "integrity_failure", job)
	default:
		respondError(w, 422, "operation_failed", job)
	}
}
