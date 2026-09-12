package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
	"github.com/rcarmo/go-pherence/runtime/speechjob/httpapi"
)

const token = "0123456789abcdef0123456789abcdef"
const job = "0123456789abcdef0123456789abcdef"

func hashText(text string) string { // fixture helper, same SHA256 contract as wire
	return fmt.Sprintf("%x", sha256.Sum256([]byte(text)))
}
func envFor(endpoint string) func(string) string {
	return func(k string) string {
		switch k {
		case "SPEECHJOB_URL":
			return endpoint
		case "SPEECHJOB_TOKEN":
			return token
		}
		return ""
	}
}
func invoke(ctx context.Context, endpoint string, args ...string) (string, error) {
	var out bytes.Buffer
	e := execute(ctx, append([]string{"--allow-loopback-http"}, args...), envFor(endpoint), &out)
	return out.String(), e
}
func localClient(t *testing.T, h http.Handler) (*client, *httptest.Server) {
	t.Helper()
	s := httptest.NewServer(h)
	t.Cleanup(s.Close)
	c, e := newClient(s.URL, token, true)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(c.close)
	return c, s
}
func assertNoTemp(t *testing.T, dir string) {
	t.Helper()
	entries, e := os.ReadDir(dir)
	if e != nil {
		t.Fatal(e)
	}
	for _, x := range entries {
		if strings.HasPrefix(x.Name(), ".speechjob-download-") {
			t.Fatal("temporary file leaked", x.Name())
		}
	}
}
func downloadServer(body string, mutate func(http.ResponseWriter, *http.Request, string)) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+token {
			w.WriteHeader(401)
			return
		}
		if r.URL.Path == "/v1/jobs/"+job {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"id": job, "artifacts": []artifact{{"vtt", int64(len(body)), hashText(body)}}})
			return
		}
		if mutate != nil {
			mutate(w, r, body)
			return
		}
		w.Header().Set("Content-Type", "text/vtt; charset=utf-8")
		w.Header().Set("Content-Length", fmt.Sprint(len(body)))
		w.Header().Set("ETag", `"sha256-`+hashText(body)+`"`)
		io.WriteString(w, body)
	})
}
func TestDownloadVerifiesAndNeverOverwrites(t *testing.T) {
	text := "WEBVTT\n\n"
	c, _ := localClient(t, downloadServer(text, nil))
	dir := t.TempDir()
	dest := filepath.Join(dir, "result.vtt")
	var out bytes.Buffer
	if e := c.download(context.Background(), job, "vtt", dest, &out); e != nil {
		t.Fatal(e)
	}
	b, e := os.ReadFile(dest)
	if e != nil || string(b) != text {
		t.Fatal(string(b), e)
	}
	st, _ := os.Stat(dest)
	if st.Mode().Perm() != 0600 {
		t.Fatal(st.Mode())
	}
	assertNoTemp(t, dir)
	if e = c.download(context.Background(), job, "vtt", dest, io.Discard); e == nil {
		t.Fatal("replaced output")
	}
	if b, _ = os.ReadFile(dest); string(b) != text {
		t.Fatal("existing changed")
	}
	link := filepath.Join(dir, "symlink")
	if e = os.Symlink(dest, link); e == nil {
		if e = c.download(context.Background(), job, "vtt", link, io.Discard); e == nil {
			t.Fatal("symlink replaced")
		}
	}
}
func TestDownloadRejectsCorruptionAndConflictingPublication(t *testing.T) {
	for _, kind := range []string{"hash", "short", "length", "etag", "type", "encoding", "duplicate-etag", "race-existing"} {
		t.Run(kind, func(t *testing.T) {
			dir := t.TempDir()
			dest := filepath.Join(dir, "result.vtt")
			text := "WEBVTT\n\n"
			c, _ := localClient(t, downloadServer(text, func(w http.ResponseWriter, r *http.Request, b string) {
				w.Header().Set("Content-Type", "text/vtt")
				w.Header().Set("Content-Length", fmt.Sprint(len(b)))
				w.Header().Set("ETag", `"sha256-`+hashText(b)+`"`)
				switch kind {
				case "hash":
					b = "BADVTT\n\n"
				case "short":
					b = b[:3]
				case "length":
					w.Header().Set("Content-Length", "99")
				case "etag":
					w.Header().Set("ETag", `"not-a-hash"`)
				case "type":
					w.Header().Set("Content-Type", "text/html")
				case "encoding":
					w.Header().Set("Content-Encoding", "gzip")
				case "duplicate-etag":
					w.Header().Add("ETag", `"other"`)
				case "race-existing":
					if e := os.WriteFile(dest, []byte("keep"), 0600); e != nil {
						t.Error(e)
					}
				}
				io.WriteString(w, b)
			}))
			if e := c.download(context.Background(), job, "vtt", dest, io.Discard); e == nil {
				t.Fatal("accepted", kind)
			}
			b, e := os.ReadFile(dest)
			if kind == "race-existing" {
				if e != nil || string(b) != "keep" {
					t.Fatal(string(b), e)
				}
			} else if !errors.Is(e, os.ErrNotExist) {
				t.Fatal("published corrupt data", string(b), e)
			}
			assertNoTemp(t, dir)
		})
	}
}
func TestDownloadCancellationCleansScratch(t *testing.T) {
	entered := make(chan struct{})
	c, _ := localClient(t, downloadServer("WEBVTT\n\n", func(w http.ResponseWriter, r *http.Request, b string) {
		w.Header().Set("Content-Type", "text/vtt")
		w.Header().Set("Content-Length", fmt.Sprint(len(b)))
		w.Header().Set("ETag", `"sha256-`+hashText(b)+`"`)
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		close(entered)
		<-r.Context().Done()
	}))
	ctx, cancel := context.WithCancel(context.Background())
	dir := t.TempDir()
	dest := filepath.Join(dir, "result")
	done := make(chan error, 1)
	go func() { done <- c.download(ctx, job, "vtt", dest, io.Discard) }()
	<-entered
	cancel()
	if e := <-done; e == nil {
		t.Fatal("cancel accepted")
	}
	if _, e := os.Stat(dest); !errors.Is(e, os.ErrNotExist) {
		t.Fatal(e)
	}
	assertNoTemp(t, dir)
}
func TestEndpointTokenAndRedirectPolicy(t *testing.T) {
	for _, endpoint := range []string{"", "https://host/path", "https://u:p@host", "https://host?token=secret", "https://host?", "https://host/#x", "http://example.com", "http://localhost", "http://127.0.0.1.evil", "ftp://127.0.0.1"} {
		if c, e := newClient(endpoint, token, true); e == nil {
			c.close()
			t.Fatal("invalid endpoint", endpoint)
		}
	}
	if c, e := newClient("http://127.0.0.1", token, false); e == nil {
		c.close()
		t.Fatal("unapproved HTTP")
	}
	for _, bad := range []string{"short", token + "\n", token + " "} {
		if c, e := newClient("https://speech.test", bad, false); e == nil {
			c.close()
			t.Fatal("bad token")
		}
	}
	var targetCalls atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { targetCalls.Add(1) }))
	defer target.Close()
	c, s := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, target.URL, 307) }))
	if e := c.jsonRequest(context.Background(), "POST", "/v1/jobs/"+job+"/run", nil, 0, 200, io.Discard); e == nil {
		t.Fatal("redirect followed")
	}
	if targetCalls.Load() != 0 {
		t.Fatal("token reached redirect target")
	}
	// Proxy settings cannot redirect authenticated requests. Literal loopback is
	// also the only plain-HTTP endpoint admitted; no DNS rebinding exception.
	t.Setenv("HTTP_PROXY", target.URL)
	t.Setenv("HTTPS_PROXY", target.URL)
	if c.transport.Proxy != nil {
		t.Fatal("proxy enabled")
	}
	_ = s
}
func TestJSONBoundsAndErrorRedaction(t *testing.T) {
	for _, kind := range []string{"huge", "bad-json", "wrong-type", "encoding", "private-error", "known-error"} {
		t.Run(kind, func(t *testing.T) {
			c, _ := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch kind {
				case "huge":
					io.WriteString(w, strings.Repeat(" ", maxJSON+1))
				case "bad-json":
					io.WriteString(w, "not JSON")
				case "wrong-type":
					w.Header().Set("Content-Type", "text/html")
					io.WriteString(w, "<html>secret</html>")
				case "encoding":
					w.Header().Set("Content-Encoding", "gzip")
					io.WriteString(w, "{}")
				case "private-error":
					w.WriteHeader(500)
					io.WriteString(w, `{"error":"`+token+`"}`)
				case "known-error":
					w.WriteHeader(409)
					io.WriteString(w, `{"error":"busy","details":"secret"}`)
				}
			}))
			var out bytes.Buffer
			e := c.jsonRequest(context.Background(), "GET", "/v1/jobs", nil, 0, 200, &out)
			if e == nil || out.Len() != 0 || strings.Contains(e.Error(), token) || strings.Contains(e.Error(), "secret") {
				t.Fatal(out.String(), e)
			}
			if kind == "known-error" && !strings.Contains(e.Error(), "busy") {
				t.Fatal(e)
			}
		})
	}
	c, _ := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"name":"\u001b[2J <script>","number":9007199254740993}`)
	}))
	var out bytes.Buffer
	if e := c.jsonRequest(context.Background(), "GET", "/v1/jobs", nil, 0, 200, &out); e != nil {
		t.Fatal(e)
	}
	if strings.Contains(out.String(), "\x1b") || strings.Contains(out.String(), "<script>") || !strings.Contains(out.String(), "9007199254740993") {
		t.Fatal(out.String())
	}
}

type cancelAtEOF struct{ cancel context.CancelFunc }

func (r cancelAtEOF) Read(b []byte) (int, error) { r.cancel(); copy(b, "{}"); return 2, io.EOF }
func TestEOFAcknowledgementWinsRacingCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	b, e := readBounded(ctx, cancelAtEOF{cancel}, 10)
	if e != nil || string(b) != "{}" {
		t.Fatal(string(b), e)
	}
}
func TestCLIArgumentsAndNoTokenFlags(t *testing.T) {
	for _, args := range [][]string{{"--token", token, "list"}, {"--timeout", "0", "list"}, {"run", "../"}, {"delete", job}, {"get", job, "extra"}, {"list", "--after", "bad"}, {"download", "--job", job, "--artifact", "input", "--out", "x"}, {"unknown"}} {
		var out bytes.Buffer
		e := execute(context.Background(), args, envFor("https://speech.test"), &out)
		if e == nil || strings.Contains(e.Error(), token) {
			t.Fatal(args, e)
		}
	}
	var out bytes.Buffer
	if e := execute(context.Background(), []string{"--help"}, envFor(""), &out); e != nil || !strings.Contains(out.String(), "SPEECHJOB_TOKEN") {
		t.Fatal(out.String(), e)
	}
}
func TestCLIActualHTTPWorkflow(t *testing.T) {
	root := filepath.Join(t.TempDir(), "store")
	store, e := speechjob.Open(root, speechjob.Limits{MaxJobs: 8, MaxUploadBytes: 4096, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20})
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	server := httptest.NewUnstartedServer(nil)
	host := server.Listener.Addr().String()
	stages := []speechjob.Stage{{Name: "vtt", Version: hashText("vtt"), Run: func(_ context.Context, _ *speechjob.Input, w io.Writer) error {
		_, e := io.WriteString(w, "WEBVTT\n\n")
		return e
	}}}
	handler, e := httpapi.New(httpapi.Config{Store: store, Profiles: []httpapi.Profile{{ID: "test", Configuration: []byte(`{}`), Stages: stages}}, Token: token, Hosts: []string{host}, MaxUploadBytes: 1024, MaxConcurrentRequests: 4})
	if e != nil {
		t.Fatal(e)
	}
	server.Config.Handler = handler
	server.Start()
	defer server.Close()
	defer handler.Shutdown(context.Background())
	dir := t.TempDir()
	input := filepath.Join(dir, "áudio.wav")
	if e = os.WriteFile(input, []byte("audio fixture"), 0600); e != nil {
		t.Fatal(e)
	}
	raw, e := invoke(context.Background(), server.URL, "upload", "--profile", "test", "--file", input, "--name", "../../Áudio & ?.wav")
	if e != nil {
		t.Fatal(e)
	}
	var status struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}
	if e = json.Unmarshal([]byte(raw), &status); e != nil || status.Status != "queued" {
		t.Fatal(raw, e)
	}
	for _, command := range []string{"get", "run"} {
		raw, e = invoke(context.Background(), server.URL, command, status.ID)
		if e != nil {
			t.Fatal(command, e)
		}
	}
	if !strings.Contains(raw, `"status": "complete"`) {
		t.Fatal(raw)
	}
	for _, args := range [][]string{{"list"}, {"list", "--after", status.ID}, {"inventory"}} {
		if _, e = invoke(context.Background(), server.URL, args...); e != nil {
			t.Fatal(args, e)
		}
	}
	dest := filepath.Join(dir, "download.vtt")
	if _, e = invoke(context.Background(), server.URL, "download", "--job", status.ID, "--artifact", "vtt", "--out", dest); e != nil {
		t.Fatal(e)
	}
	if _, e = invoke(context.Background(), server.URL, "delete", "--confirm", status.ID); e != nil {
		t.Fatal(e)
	}
	if _, e = store.Get(status.ID); !errors.Is(e, os.ErrNotExist) {
		t.Fatal(e)
	}
}
func TestUploadValidationAndExactWireBytes(t *testing.T) {
	var got []byte
	var query url.Values
	c, _ := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var e error
		got, e = io.ReadAll(r.Body)
		if e != nil {
			t.Error(e)
		}
		query = r.URL.Query()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(201)
		io.WriteString(w, "{}")
	}))
	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.wav")
	data := []byte{0, 1, 2, 255}
	if e := os.WriteFile(path, data, 0600); e != nil {
		t.Fatal(e)
	}
	if e := c.upload(context.Background(), "test", path, "space & ? = Á", io.Discard); e != nil {
		t.Fatal(e)
	}
	if !bytes.Equal(got, data) || query.Get("name") != "space & ? = Á" {
		t.Fatal(got, query)
	}
	for _, name := range []string{"x\n", "x\x00", strings.Repeat("a", 257)} {
		if e := c.upload(context.Background(), "test", path, name, io.Discard); e == nil {
			t.Fatal(name)
		}
	}
	if e := c.upload(context.Background(), "../", path, "", io.Discard); e == nil {
		t.Fatal("profile injection")
	}
	if e := c.upload(context.Background(), "test", dir, "", io.Discard); e == nil {
		t.Fatal("directory")
	}
	link := filepath.Join(dir, "link")
	if e := os.Symlink(path, link); e == nil {
		if e = c.upload(context.Background(), "test", link, "", io.Discard); e == nil {
			t.Fatal("symlink")
		}
	}
}
func TestTimeoutDoesNotRetryMutation(t *testing.T) {
	var calls atomic.Int32
	c, _ := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); <-r.Context().Done() }))
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	e := c.jsonRequest(ctx, "POST", "/v1/jobs/"+job+"/run", nil, 0, 200, io.Discard)
	if e == nil || calls.Load() != 1 || !strings.Contains(e.Error(), "inspect") {
		t.Fatal(calls.Load(), e)
	}
}

func TestMalformedDownloadMetadataFailsBeforeArtifactRequest(t *testing.T) {
	for _, kind := range []string{"id", "missing", "duplicate", "size", "hash"} {
		t.Run(kind, func(t *testing.T) {
			var artifacts atomic.Int32
			c, _ := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if strings.Contains(r.URL.Path, "/artifacts/") {
					artifacts.Add(1)
					t.Error("bad metadata reached download")
					return
				}
				id := job
				rows := []artifact{{"vtt", 8, hashText("WEBVTT\n\n")}}
				switch kind {
				case "id":
					id = strings.Repeat("a", 32)
				case "missing":
					rows = nil
				case "duplicate":
					rows = append(rows, rows[0])
				case "size":
					rows[0].Bytes = maxArtifact + 1
				case "hash":
					rows[0].SHA256 = "bad"
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{"id": id, "artifacts": rows})
			}))
			if e := c.download(context.Background(), job, "vtt", filepath.Join(t.TempDir(), "out"), io.Discard); e == nil || artifacts.Load() != 0 {
				t.Fatal(kind, e)
			}
		})
	}
}

type failingOutput struct{}

func (failingOutput) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
func TestOutputFailureReportsPublishedState(t *testing.T) {
	c, _ := localClient(t, downloadServer("WEBVTT\n\n", nil))
	dest := filepath.Join(t.TempDir(), "result.vtt")
	if e := c.download(context.Background(), job, "vtt", dest, failingOutput{}); e == nil || !strings.Contains(e.Error(), "published and synced") {
		t.Fatal(e)
	}
	if b, e := os.ReadFile(dest); e != nil || string(b) != "WEBVTT\n\n" {
		t.Fatal(string(b), e)
	}
	c2, _ := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, "{}")
	}))
	if e := c2.jsonRequest(context.Background(), "POST", "/v1/jobs/"+job+"/run", nil, 0, 200, failingOutput{}); e == nil || !strings.Contains(e.Error(), "response received") {
		t.Fatal(e)
	}
}
func TestTLSVerificationAndCancelCommand(t *testing.T) {
	tls := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("untrusted TLS request arrived") }))
	defer tls.Close()
	c, e := newClient(tls.URL, token, false)
	if e != nil {
		t.Fatal(e)
	}
	defer c.close()
	if e = c.jsonRequest(context.Background(), "GET", "/v1/jobs", nil, 0, 200, io.Discard); e == nil {
		t.Fatal("untrusted TLS accepted")
	}
	var method, path string
	_, server := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(202)
		io.WriteString(w, `{"cancellation_requested":true}`)
	}))
	if _, e = invoke(context.Background(), server.URL, "cancel", job); e != nil || method != "POST" || path != "/v1/jobs/"+job+"/cancel" {
		t.Fatal(method, path, e)
	}
}

func TestSafeJSONEscapesTerminalAndBidiControls(t *testing.T) {
	source := "Olá\x1b[2J\u009b31m\u007f\u202eevil\U000e0001"
	var out bytes.Buffer
	if e := writeSafeJSON(&out, map[string]string{"name": source}); e != nil {
		t.Fatal(e)
	}
	for _, raw := range []string{"\x1b", "\u009b", "\u007f", "\u202e", "\U000e0001"} {
		if strings.Contains(out.String(), raw) {
			t.Fatal("literal terminal/format control", out.String())
		}
	}
	var decoded map[string]string
	if e := json.Unmarshal(out.Bytes(), &decoded); e != nil || decoded["name"] != source {
		t.Fatal("JSON value changed", out.String(), e)
	}
}

func TestQueueCLIExplicitCommands(t *testing.T) {
	for _, tc := range []struct {
		command, method, path string
		status                int
	}{{"queue", "GET", "/v1/queue", 200}, {"enqueue", "POST", "/v1/jobs/" + job + "/enqueue", 202}, {"retry-queued", "POST", "/v1/jobs/" + job + "/retry-queued", 202}, {"forget-queued", "DELETE", "/v1/jobs/" + job + "/queue", 204}} {
		_, server := localClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != tc.method || r.URL.Path != tc.path {
				t.Error(r.Method, r.URL.Path)
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(tc.status)
			if tc.status != 204 {
				io.WriteString(w, `{}`)
			}
		}))
		args := []string{tc.command}
		if tc.command != "queue" {
			args = append(args, job)
		}
		if _, e := invoke(context.Background(), server.URL, args...); e != nil {
			t.Fatal(tc.command, e)
		}
	}
}
