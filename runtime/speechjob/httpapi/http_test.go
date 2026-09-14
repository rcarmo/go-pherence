package httpapi

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

const testToken = "0123456789abcdef0123456789abcdef"

func testHash(s string) string { d := sha256.Sum256([]byte(s)); return hex.EncodeToString(d[:]) }
func testStage(name string, run func(context.Context, *speechjob.Input, io.Writer) error) speechjob.Stage {
	return speechjob.Stage{Name: name, Version: testHash(name), Run: run}
}
func textStage(name, text string) speechjob.Stage {
	return testStage(name, func(_ context.Context, _ *speechjob.Input, w io.Writer) error {
		_, e := io.WriteString(w, text)
		return e
	})
}
func fixture(t *testing.T, stages []speechjob.Stage, slots int) (*Handler, *speechjob.Store, string) {
	t.Helper()
	root := filepath.Join(t.TempDir(), "jobs")
	s, e := speechjob.Open(root, speechjob.Limits{MaxJobs: 16, MaxUploadBytes: 4096, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20})
	if e != nil {
		t.Fatal(e)
	}
	cfg := Config{Store: s, Profiles: []Profile{{ID: "test", Configuration: []byte(`{"language":"pt","runtime":"private-server-path"}`), Stages: stages}}, Token: testToken, Hosts: []string{"speech.test"}, Origin: "https://speech.test", MaxUploadBytes: 1024, MaxConcurrentRequests: slots}
	h, e := New(cfg)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if e := h.Shutdown(ctx); e != nil {
			t.Error(e)
		}
		if e := s.Close(); e != nil {
			t.Error(e)
		}
	})
	return h, s, root
}
func request(h http.Handler, method, path string, body io.Reader) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, "https://speech.test"+path, body)
	r.Header.Set("Authorization", "Bearer "+testToken)
	if method == http.MethodPost && strings.HasPrefix(path, "/v1/jobs?") {
		r.Header.Set("Content-Type", "application/octet-stream")
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}
func decodeJob(t *testing.T, w *httptest.ResponseRecorder, status int) Job {
	t.Helper()
	if w.Code != status {
		t.Fatalf("status=%d want=%d body=%s", w.Code, status, w.Body.String())
	}
	var j Job
	if e := json.Unmarshal(w.Body.Bytes(), &j); e != nil {
		t.Fatal(e)
	}
	return j
}
func upload(t *testing.T, h *Handler) Job {
	return decodeJob(t, request(h, "POST", "/v1/jobs?profile=test&name=recording.wav", strings.NewReader("fake audio")), 201)
}

func TestLifecycleAndDownloadAllowlist(t *testing.T) {
	transcript := textStage("transcript", `{"plain":"Olá <&>"}`)
	vtt := textStage("vtt", "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nOlá\n")
	h, s, _ := fixture(t, []speechjob.Stage{textStage("decode", "PRIVATE PCM"), textStage("asr-windows", "RAW TOKENS"), transcript, vtt, textStage("diarization", "PRIVATE RAW TURNS"), textStage("speaker-transcript", `{"experimental":true}`), textStage("speaker-vtt", "WEBVTT\n\nNOTE Experimental\n")}, 4)
	j := upload(t, h)
	if j.Status != speechjob.Queued || j.Profile != "test" || !j.ProfileAvailable || j.InputBytes != 10 {
		t.Fatal(j)
	}
	if w := request(h, "GET", "/v1/jobs", nil); w.Code != 200 || strings.Contains(w.Body.String(), "private-server-path") || strings.Contains(w.Body.String(), "Configuration") {
		t.Fatal(w.Code, w.Body.String())
	}
	j = decodeJob(t, request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	if j.Status != speechjob.Complete || len(j.Artifacts) != 4 {
		t.Fatal(j)
	}
	for _, stage := range []string{"input", "source", "decode", "asr-windows", "diarization", "manifest.json", ".lock"} {
		w := request(h, "GET", "/v1/jobs/"+j.ID+"/artifacts/"+stage, nil)
		if w.Code != 404 {
			t.Fatal(stage, w.Code)
		}
	}
	w := request(h, "GET", "/v1/jobs/"+j.ID+"/artifacts/vtt", nil)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "Olá") || w.Header().Get("Cache-Control") != "no-store" || w.Header().Get("X-Content-Type-Options") != "nosniff" || !strings.HasPrefix(w.Header().Get("Content-Disposition"), "attachment;") {
		t.Fatal(w)
	}
	head := request(h, "HEAD", "/v1/jobs/"+j.ID+"/artifacts/vtt", nil)
	if head.Code != 200 || head.Body.Len() != 0 || head.Header().Get("Content-Length") != fmt.Sprint(w.Body.Len()) {
		t.Fatal(head)
	}
	rerun := decodeJob(t, request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	if rerun.Attempts != j.Attempts {
		t.Fatal("completed run repeated")
	}
	if w = request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 204 {
		t.Fatal(w)
	}
	if w = request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 204 {
		t.Fatal(w)
	}
	if _, e := s.Get(j.ID); !errors.Is(e, os.ErrNotExist) {
		t.Fatal(e)
	}
}
func TestPublicFailureCodeAllowlist(t *testing.T) {
	base := speechjob.Manifest{Status: speechjob.Failed}
	for _, tc := range []struct {
		error, code string
	}{
		{"unsupported media input: expected RIFF/WAVE content", "media_type_mismatch"},
		{"unsupported media input: expected ISO BMFF content", "media_type_mismatch"},
		{"secret-file-path /models/private-recording token=not-public", "job_failed"},
	} {
		m := base
		m.Error = tc.error
		if got := publicFailureCode(m); got != tc.code {
			t.Fatal(tc.error, got)
		}
	}
	base.Status = speechjob.Complete
	if got := publicFailureCode(base); got != "" {
		t.Fatal(got)
	}
}

func TestFailedMediaTypeGetsSafeActionableCode(t *testing.T) {
	private := "unsupported media input: expected RIFF/WAVE content"
	st := testStage("decode", func(context.Context, *speechjob.Input, io.Writer) error { return errors.New(private) })
	h, _, _ := fixture(t, []speechjob.Stage{st}, 2)
	j := upload(t, h)
	w := request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil)
	if w.Code != 422 || !strings.Contains(w.Body.String(), `"failure_code":"media_type_mismatch"`) || strings.Contains(w.Body.String(), "RIFF/WAVE") {
		t.Fatal(w.Code, w.Body.String())
	}
	w = request(h, "GET", "/v1/jobs/"+j.ID, nil)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"failure_code":"media_type_mismatch"`) || strings.Contains(w.Body.String(), "RIFF/WAVE") {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestFailedRunKeepsDownloadsAndRedactsErrors(t *testing.T) {
	fail := true
	private := "secret-file-path /models/private-recording token=not-public"
	st := testStage("diarization", func(context.Context, *speechjob.Input, io.Writer) error {
		if fail {
			return errors.New(private)
		}
		return nil
	})
	h, _, _ := fixture(t, []speechjob.Stage{textStage("transcript", "text"), textStage("vtt", "WEBVTT\n\n"), st}, 4)
	j := upload(t, h)
	w := request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil)
	if w.Code != 422 || strings.Contains(w.Body.String(), private) || !strings.Contains(w.Body.String(), `"status":"failed"`) {
		t.Fatal(w.Code, w.Body.String())
	}
	w = request(h, "GET", "/v1/jobs/"+j.ID, nil)
	if strings.Contains(w.Body.String(), private) || !strings.Contains(w.Body.String(), `"failure_code":"job_failed"`) {
		t.Fatal("error leak or missing safe failure code", w.Body.String())
	}
	if w = request(h, "GET", "/v1/jobs/"+j.ID+"/artifacts/transcript", nil); w.Code != 200 || w.Body.String() != "text" {
		t.Fatal(w)
	}
	fail = false
	j = decodeJob(t, request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	if j.Status != speechjob.Complete || j.Attempts != 2 {
		t.Fatal(j)
	}
}

type countingBody struct {
	reads  int
	closed bool
}

func (b *countingBody) Read([]byte) (int, error) {
	b.reads++
	return 0, errors.New("body must not be read")
}
func (b *countingBody) Close() error { b.closed = true; return nil }
func TestAuthenticationOriginRoutingBeforeBody(t *testing.T) {
	h, _, _ := fixture(t, []speechjob.Stage{textStage("transcript", "x")}, 2)
	for _, tc := range []struct {
		name   string
		mutate func(*http.Request)
		code   int
	}{
		{"missing-auth", func(r *http.Request) { r.Header.Del("Authorization") }, 401},
		{"wrong-auth", func(r *http.Request) { r.Header.Set("Authorization", "Bearer wrong") }, 401},
		{"duplicate-auth", func(r *http.Request) { r.Header.Add("Authorization", "Bearer "+testToken) }, 401},
		{"query-token", func(r *http.Request) { r.Header.Del("Authorization"); r.URL.RawQuery += "&token=" + testToken }, 401},
		{"host", func(r *http.Request) { r.Host = "attacker.test"; r.Header.Set("X-Forwarded-Host", "speech.test") }, 421},
		{"origin", func(r *http.Request) { r.Header.Set("Origin", "https://attacker.test") }, 403},
		{"duplicate-origin", func(r *http.Request) {
			r.Header.Add("Origin", "https://speech.test")
			r.Header.Add("Origin", "https://speech.test")
		}, 403},
		{"fetch-site", func(r *http.Request) { r.Header.Set("Sec-Fetch-Site", "cross-site") }, 403},
		{"unknown-profile", func(r *http.Request) { r.URL.RawQuery = "profile=evil&name=x" }, 400},
		{"dup-query", func(r *http.Request) { r.URL.RawQuery += "&profile=test" }, 400},
		{"config-injection", func(r *http.Request) { r.URL.RawQuery += "&configuration=%7B%7D" }, 400},
		{"encoding", func(r *http.Request) { r.Header.Set("Content-Encoding", "gzip") }, 415},
		{"content-type", func(r *http.Request) { r.Header.Set("Content-Type", "application/json") }, 415},
		{"big-declared", func(r *http.Request) { r.ContentLength = 1025 }, 413},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := &countingBody{}
			r := httptest.NewRequest("POST", "https://speech.test/v1/jobs?profile=test&name=x", body)
			r.Header.Set("Authorization", "Bearer "+testToken)
			r.Header.Set("Content-Type", "application/octet-stream")
			tc.mutate(r)
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != tc.code || body.reads != 0 {
				t.Fatal(w.Code, w.Body.String(), body.reads)
			}
		})
	}
	for _, p := range []string{"/v1/jobs/", "/v1//jobs", "/v1/jobs/%2e%2e", "/v1/jobs/../inventory", "/v1/jobs/" + strings.Repeat("a", 32) + "/artifacts/..%2Finput"} {
		if w := request(h, "GET", p, nil); w.Code != 404 {
			t.Fatal(p, w.Code)
		}
	}
	if w := request(h, "OPTIONS", "/v1/jobs", nil); w.Code != 405 || w.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatal(w)
	}
	if w := request(h, "GET", "/v1/jobs", strings.NewReader("unexpected")); w.Code != 400 {
		t.Fatal(w)
	}
}
func TestUploadBoundsAndInventory(t *testing.T) {
	h, s, _ := fixture(t, []speechjob.Stage{textStage("transcript", "x")}, 3)
	r := httptest.NewRequest("POST", "https://speech.test/v1/jobs?profile=test&name=x", strings.NewReader(strings.Repeat("x", 1025)))
	r.ContentLength = -1
	r.TransferEncoding = []string{"chunked"}
	r.Header.Set("Authorization", "Bearer "+testToken)
	r.Header.Set("Content-Type", "application/octet-stream")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 413 {
		t.Fatal(w.Code, w.Body.String())
	}
	inv, e := s.Inventory()
	if e != nil || len(inv) != 1 || inv[0].ManifestPresent {
		t.Fatal(inv, e)
	}
	w = request(h, "GET", "/v1/inventory", nil)
	if w.Code != 200 || strings.Contains(w.Body.String(), "configuration") || strings.Contains(w.Body.String(), "input.wav") {
		t.Fatal(w)
	}
	if w = request(h, "DELETE", "/v1/jobs/"+inv[0].ID, nil); w.Code != 204 {
		t.Fatal(w)
	}
	w = request(h, "POST", "/v1/jobs?profile=test&name=x", strings.NewReader(strings.Repeat("x", 1024)))
	if w.Code != 201 {
		t.Fatal(w)
	}
	for _, name := range []string{"", "bad%0Aname", strings.Repeat("a", 257)} {
		w = request(h, "POST", "/v1/jobs?profile=test&name="+name, strings.NewReader("x"))
		if w.Code != 400 {
			t.Fatal(name, w)
		}
	}
}
func TestCancelReservedSlotAndDrain(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int32
	st := testStage("transcript", func(ctx context.Context, _ *speechjob.Input, w io.Writer) error {
		if calls.Add(1) == 1 {
			close(entered)
			<-ctx.Done()
			<-release
			return ctx.Err()
		}
		_, e := io.WriteString(w, "done")
		return e
	})
	h, s, _ := fixture(t, []speechjob.Stage{st}, 1)
	j := upload(t, h)
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil) }()
	<-entered
	if w := request(h, "GET", "/v1/jobs", nil); w.Code != 503 {
		t.Fatal(w)
	}
	if w := request(h, "POST", "/v1/jobs/"+j.ID+"/cancel", nil); w.Code != 202 {
		t.Fatal(w)
	}
	if e := s.Close(); !errors.Is(e, speechjob.ErrBusy) {
		t.Fatal(e)
	}
	select {
	case w := <-done:
		t.Fatal("returned before owned drain", w)
	default:
	}
	close(release)
	w := <-done
	if w.Code != 409 || !strings.Contains(w.Body.String(), `"status":"cancelled"`) {
		t.Fatal(w.Code, w.Body.String())
	}
	j = decodeJob(t, request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	if j.Status != speechjob.Complete || calls.Load() != 2 {
		t.Fatal(j)
	}
}
func TestShutdownWaitsForHandlers(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	h, s, _ := fixture(t, []speechjob.Stage{testStage("transcript", func(ctx context.Context, _ *speechjob.Input, _ io.Writer) error {
		close(entered)
		<-ctx.Done()
		<-release
		return ctx.Err()
	})}, 2)
	j := upload(t, h)
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil) }()
	<-entered
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e := h.Shutdown(ctx); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if w := request(h, "GET", "/v1/jobs", nil); w.Code != 503 {
		t.Fatal(w)
	}
	if e := s.Close(); !errors.Is(e, speechjob.ErrBusy) {
		t.Fatal(e)
	}
	close(release)
	<-done
	if e := h.Shutdown(context.Background()); e != nil {
		t.Fatal(e)
	}
}
func TestProfileDriftReadDeleteAndIdentityCopy(t *testing.T) {
	h, s, _ := fixture(t, []speechjob.Stage{textStage("transcript", "x")}, 2)
	j := upload(t, h)
	if e := h.Shutdown(context.Background()); e != nil {
		t.Fatal(e)
	}
	cfg := Config{Store: s, Profiles: []Profile{{ID: "test", Configuration: []byte(`{"changed":true}`), Stages: []speechjob.Stage{textStage("transcript", "y")}}}, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1024, MaxConcurrentRequests: 2}
	newer, e := New(cfg)
	if e != nil {
		t.Fatal(e)
	}
	defer newer.Shutdown(context.Background())
	stale := decodeJob(t, request(newer, "GET", "/v1/jobs/"+j.ID, nil), 200)
	if stale.Profile != "" || stale.ProfileAvailable {
		t.Fatal(stale)
	}
	if w := request(newer, "POST", "/v1/jobs/"+j.ID+"/run", nil); w.Code != 409 || !strings.Contains(w.Body.String(), "profile_changed") {
		t.Fatal(w)
	}
	if w := request(newer, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 204 {
		t.Fatal(w)
	}
	// Mutation of caller slices after construction does not alter stored identity/stages.
	p := newer.profiles["test"]
	cfg.Profiles[0].Configuration[0] = '!'
	cfg.Profiles[0].Stages[0].Run = nil
	cfg.Hosts[0] = "evil.test"
	j = upload(t, newer)
	m, e := s.Get(j.ID)
	if e != nil || m.Configuration != string(p.configuration) {
		t.Fatal(e)
	}
	j = decodeJob(t, request(newer, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	if j.Status != speechjob.Complete {
		t.Fatal(j)
	}
}
func TestRecordingTitleAndHumanExportFilename(t *testing.T) {
	h, _, _ := fixture(t, []speechjob.Stage{textStage("vtt", "WEBVTT\n\n")}, 2)
	j := upload(t, h)
	if j.Title != "Recording" || j.Name != "recording.wav" {
		t.Fatal(j)
	}
	// request() fixes mutation bodies as octet-stream; use an explicit JSON request.
	r := httptest.NewRequest("PUT", "https://speech.test/v1/jobs/"+j.ID+"/title", strings.NewReader(`{"title":"Customer Interview"}`))
	r.Header.Set("Authorization", "Bearer "+testToken)
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	j = decodeJob(t, w, 200)
	if j.Title != "Customer Interview" || j.Name != "recording.wav" {
		t.Fatal(j)
	}
	j = decodeJob(t, request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	if len(j.Artifacts) != 1 || j.Artifacts[0].Filename != "customer-interview.transcript.vtt" {
		t.Fatal(j.Artifacts)
	}
	w = request(h, "GET", "/v1/jobs/"+j.ID+"/artifacts/vtt", nil)
	if got := w.Header().Get("Content-Disposition"); got != `attachment; filename="customer-interview.transcript.vtt"` || strings.Contains(got, j.ID) {
		t.Fatal(got)
	}
	for _, payload := range []string{`{"title":""}`, `{"title":" padded "}`, `{"title":"x","extra":1}`} {
		r = httptest.NewRequest("PUT", "https://speech.test/v1/jobs/"+j.ID+"/title", strings.NewReader(payload))
		r.Header.Set("Authorization", "Bearer "+testToken)
		r.Header.Set("Content-Type", "application/json")
		w = httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 400 {
			t.Fatal(payload, w.Code, w.Body.String())
		}
	}
}

func TestArtifactIntegrityHeadersAndMethods(t *testing.T) {
	h, s, root := fixture(t, []speechjob.Stage{textStage("vtt", "WEBVTT\n\n")}, 2)
	j := upload(t, h)
	j = decodeJob(t, request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	r := httptest.NewRequest("GET", "https://speech.test/v1/jobs/"+j.ID+"/artifacts/vtt", nil)
	r.Header.Set("Authorization", "Bearer "+testToken)
	r.Header.Set("Range", "bytes=0-1")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 416 {
		t.Fatal(w)
	}
	if w = request(h, "POST", "/v1/jobs/"+j.ID+"/artifacts/vtt", nil); w.Code != 405 {
		t.Fatal(w)
	}
	m, e := s.Get(j.ID)
	if e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(filepath.Join(root, j.ID, m.Checkpoints[0].Blob.File), []byte("tampered"), 0600); e != nil {
		t.Fatal(e)
	}
	w = request(h, "GET", "/v1/jobs/"+j.ID+"/artifacts/vtt", nil)
	if w.Code != 409 || strings.Contains(w.Body.String(), "tampered") {
		t.Fatal(w)
	}
}
func TestConfigAndErrorMapping(t *testing.T) {
	h, s, _ := fixture(t, []speechjob.Stage{textStage("vtt", "x")}, 2)
	good := Config{Store: s, Profiles: []Profile{{ID: "test", Configuration: []byte(`{}`), Stages: []speechjob.Stage{textStage("vtt", "x")}}}, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1, MaxConcurrentRequests: 1}
	for _, kind := range []string{"token", "host", "origin", "profile", "stage", "config", "limit", "slots"} {
		c := good
		c.Profiles = append([]Profile{}, good.Profiles...)
		switch kind {
		case "token":
			c.Token = "short"
		case "host":
			c.Hosts = []string{"evil.test/path"}
		case "origin":
			c.Origin = "https://speech.test/"
		case "profile":
			c.Profiles[0].ID = "../"
		case "stage":
			c.Profiles[0].Stages = []speechjob.Stage{{Name: "vtt", Version: "bad"}}
		case "config":
			c.Profiles[0].Configuration = []byte(`[]`)
		case "limit":
			c.MaxUploadBytes = 0
		case "slots":
			c.MaxConcurrentRequests = 65
		}
		if _, e := New(c); e == nil {
			t.Fatal(kind)
		}
	}
	for _, tc := range []struct {
		e      error
		status int
		code   string
	}{{speechjob.ErrPersistence, 503, "persistence_uncertain"}, {speechjob.ErrLimit, 413, "limit_exceeded"}, {speechjob.ErrBusy, 409, "busy"}, {speechjob.ErrCorrupt, 409, "integrity_failure"}, {speechjob.ErrClosed, 503, "unavailable"}, {context.Canceled, 409, "cancelled"}, {os.ErrNotExist, 404, "not_found"}, {errors.New("secret"), 422, "operation_failed"}} {
		w := httptest.NewRecorder()
		h.failure(w, tc.e, nil)
		if w.Code != tc.status || !strings.Contains(w.Body.String(), tc.code) || strings.Contains(w.Body.String(), "secret") {
			t.Fatal(w)
		}
	}
}

// Real loopback httptest server only: no production listener or model execution.
func TestHTTPWireTransport(t *testing.T) {
	h, _, _ := fixture(t, []speechjob.Stage{textStage("vtt", "WEBVTT\n\n")}, 2)
	server := httptest.NewServer(h)
	defer server.Close()
	client := server.Client()
	do := func(method, path string, b io.Reader) *http.Response {
		r, e := http.NewRequest(method, server.URL+path, b)
		if e != nil {
			t.Fatal(e)
		}
		r.Host = "speech.test"
		r.Header.Set("Authorization", "Bearer "+testToken)
		if b != nil {
			r.Header.Set("Content-Type", "application/octet-stream")
		}
		res, e := client.Do(r)
		if e != nil {
			t.Fatal(e)
		}
		return res
	}
	res := do("POST", "/v1/jobs?profile=test&name=x", bytes.NewBufferString("test"))
	var j Job
	if e := json.NewDecoder(res.Body).Decode(&j); e != nil {
		t.Fatal(e)
	}
	res.Body.Close()
	if res.StatusCode != 201 {
		t.Fatal(res.Status)
	}
	res = do("POST", "/v1/jobs/"+j.ID+"/run", nil)
	io.Copy(io.Discard, res.Body)
	res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatal(res.Status)
	}
	res = do("GET", "/v1/jobs/"+j.ID+"/artifacts/vtt", nil)
	b, e := io.ReadAll(res.Body)
	res.Body.Close()
	if e != nil || string(b) != "WEBVTT\n\n" || res.ContentLength != 8 {
		t.Fatal(res.Status, string(b), e)
	}
}

func TestUploadDisconnectUnblocksAndDoesNotAcknowledge(t *testing.T) {
	h, s, _ := fixture(t, []speechjob.Stage{textStage("transcript", "x")}, 2)
	pipe, writer := io.Pipe()
	defer writer.Close()
	ctx, cancel := context.WithCancel(context.Background())
	r := httptest.NewRequest("POST", "https://speech.test/v1/jobs?profile=test&name=x", pipe).WithContext(ctx)
	r.Header.Set("Authorization", "Bearer "+testToken)
	r.Header.Set("Content-Type", "application/octet-stream")
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { w := httptest.NewRecorder(); h.ServeHTTP(w, r); done <- w }()
	if _, e := writer.Write([]byte("prefix")); e != nil {
		t.Fatal(e)
	}
	cancel()
	w := <-done
	if w.Code != 409 || strings.Contains(w.Body.String(), `"id"`) {
		t.Fatal(w.Code, w.Body.String())
	}
	jobs, e := s.List()
	if e != nil || len(jobs) != 0 {
		t.Fatal(jobs, e)
	}
	inv, e := s.Inventory()
	if e != nil || len(inv) != 1 || inv[0].ManifestPresent {
		t.Fatal(inv, e)
	}
}
func TestRequestDisconnectCancelsRun(t *testing.T) {
	entered := make(chan struct{})
	h, _, _ := fixture(t, []speechjob.Stage{testStage("transcript", func(ctx context.Context, _ *speechjob.Input, _ io.Writer) error {
		close(entered)
		<-ctx.Done()
		return ctx.Err()
	})}, 2)
	j := upload(t, h)
	ctx, cancel := context.WithCancel(context.Background())
	r := httptest.NewRequest("POST", "https://speech.test/v1/jobs/"+j.ID+"/run", nil).WithContext(ctx)
	r.Header.Set("Authorization", "Bearer "+testToken)
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { w := httptest.NewRecorder(); h.ServeHTTP(w, r); done <- w }()
	<-entered
	cancel()
	w := <-done
	if w.Code != 409 || !strings.Contains(w.Body.String(), `"status":"cancelled"`) {
		t.Fatal(w.Code, w.Body.String())
	}
}
func TestReopenHandlerResume(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jobs")
	limits := speechjob.Limits{MaxJobs: 16, MaxUploadBytes: 4096, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20}
	s, e := speechjob.Open(root, limits)
	if e != nil {
		t.Fatal(e)
	}
	calls, fail := 0, true
	stages := []speechjob.Stage{testStage("transcript", func(_ context.Context, _ *speechjob.Input, w io.Writer) error {
		calls++
		_, e := io.WriteString(w, "saved")
		return e
	}), testStage("later", func(context.Context, *speechjob.Input, io.Writer) error {
		if fail {
			return io.ErrUnexpectedEOF
		}
		return nil
	})}
	cfg := Config{Store: s, Profiles: []Profile{{ID: "test", Configuration: []byte(`{}`), Stages: stages}}, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1024, MaxConcurrentRequests: 2}
	h, e := New(cfg)
	if e != nil {
		t.Fatal(e)
	}
	j := upload(t, h)
	if w := request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil); w.Code != 422 {
		t.Fatal(w)
	}
	if e = h.Shutdown(context.Background()); e != nil {
		t.Fatal(e)
	}
	if e = s.Close(); e != nil {
		t.Fatal(e)
	}
	s, e = speechjob.Open(root, limits)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	cfg.Store = s
	h, e = New(cfg)
	if e != nil {
		t.Fatal(e)
	}
	defer h.Shutdown(context.Background())
	fail = false
	j = decodeJob(t, request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	if calls != 1 || j.Attempts != 2 || j.Status != speechjob.Complete {
		t.Fatal(j, calls)
	}
}
func TestMutationConflictAndAllowedBrowserOrigin(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	h, _, _ := fixture(t, []speechjob.Stage{testStage("transcript", func(context.Context, *speechjob.Input, io.Writer) error { close(entered); <-release; return nil })}, 4)
	j := upload(t, h)
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil) }()
	<-entered
	for _, method := range []string{"POST", "DELETE"} {
		path := "/v1/jobs/" + j.ID
		if method == "POST" {
			path += "/run"
		}
		w := request(h, method, path, nil)
		if w.Code != 409 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	if w := request(h, "POST", "/v1/jobs/"+strings.Repeat("b", 32)+"/cancel", nil); w.Code != 409 {
		t.Fatal(w)
	}
	r := httptest.NewRequest("GET", "https://speech.test/v1/jobs/"+j.ID, nil)
	r.Header.Set("Authorization", "Bearer "+testToken)
	r.Header.Set("Origin", "https://speech.test")
	r.Header.Set("Sec-Fetch-Site", "same-origin")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 200 {
		t.Fatal(w)
	}
	close(release)
	<-done
}
