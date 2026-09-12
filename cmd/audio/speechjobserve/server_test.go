package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/models/whisper"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
	"github.com/rcarmo/go-pherence/runtime/speechjob/httpapi"
)

const serverToken = "0123456789abcdef0123456789abcdef"

type statusWriter struct {
	once  sync.Once
	ready chan struct{}
	mu    sync.Mutex
	b     bytes.Buffer
}

func (w *statusWriter) Write(b []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n, e := w.b.Write(b)
	w.once.Do(func() { close(w.ready) })
	return n, e
}
func TestServerRunCancellationAndOwnedDrain(t *testing.T) {
	testServerCancellationDrain(t, false)
}
func TestServerQueueCancellationAndOwnedDrain(t *testing.T) {
	testServerCancellationDrain(t, true)
}
func testServerCancellationDrain(t *testing.T, queued bool) {
	cfg := baseConfig(t)
	var queue *httpapi.QueueOptions
	if queued {
		queue = &httpapi.QueueOptions{Directory: filepath.Join(t.TempDir(), "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: 5 * time.Second, Admission: speechjob.SerialAdmission()}
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	store, e := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: 8, MaxUploadBytes: 1024, MaxArtifactBytes: 1024, MaxBytes: 1 << 20})
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	listener, e := net.Listen("tcp", "127.0.0.1:0")
	if e != nil {
		t.Fatal(e)
	}
	host := listener.Addr().String()
	handler, e := httpapi.New(httpapi.Config{Queue: queue, Store: store, Token: serverToken, Hosts: []string{host}, MaxUploadBytes: 1024, MaxConcurrentRequests: 2, Profiles: []httpapi.Profile{{ID: "test", Configuration: []byte(`{}`), Stages: []speechjob.Stage{{Name: "transcript", Version: hashBytes([]byte("stage")), Run: func(ctx context.Context, _ *speechjob.Input, _ io.Writer) error {
		close(entered)
		<-ctx.Done()
		<-release
		return ctx.Err()
	}}}}}})
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out := &statusWriter{ready: make(chan struct{})}
	done := make(chan error, 1)
	go func() { done <- serveOwned(ctx, listener, handler, cfg.HTTP, nil, out) }()
	<-out.ready
	client := &http.Client{Timeout: 5 * time.Second}
	defer client.CloseIdleConnections()
	call := func(method, path string, b io.Reader) (*http.Response, error) {
		r, _ := http.NewRequest(method, "http://"+host+path, b)
		r.Header.Set("Authorization", "Bearer "+serverToken)
		if b != nil {
			r.Header.Set("Content-Type", "application/octet-stream")
		}
		return client.Do(r)
	}
	res, e := call("POST", "/v1/jobs?profile=test&name=x", bytes.NewBufferString("fixture"))
	if e != nil {
		t.Fatal(e)
	}
	var job struct {
		ID string `json:"id"`
	}
	json.NewDecoder(res.Body).Decode(&job)
	res.Body.Close()
	if res.StatusCode != 201 {
		t.Fatal(res.Status)
	}
	endpoint := "run"
	if queued {
		endpoint = "enqueue"
		if e := handler.StartQueue(ctx); e != nil {
			t.Fatal(e)
		}
	}
	requestDone := make(chan struct{})
	go func() {
		defer close(requestDone)
		r, _ := call("POST", "/v1/jobs/"+job.ID+"/"+endpoint, nil)
		if r != nil {
			io.Copy(io.Discard, r.Body)
			r.Body.Close()
		}
	}()
	<-entered
	cancel()
	<-requestDone
	if e := store.Close(); !errors.Is(e, speechjob.ErrBusy) {
		t.Fatal("store released before callback", e)
	}
	select {
	case e := <-done:
		t.Fatal("early return", e)
	case <-time.After(1100 * time.Millisecond):
	}
	close(release)
	if e = <-done; e != nil {
		t.Fatal(e)
	}
	jobState, e := store.Get(job.ID)
	if e != nil || jobState.Status != speechjob.Cancelled {
		t.Fatal(jobState, e)
	}
	out.mu.Lock()
	defer out.mu.Unlock()
	if !bytes.Contains(out.b.Bytes(), []byte("resources retained")) {
		t.Fatal(out.b.String())
	}
}

type trivialOwner struct{ handler http.Handler }

func (h *trivialOwner) ServeHTTP(w http.ResponseWriter, r *http.Request) { h.handler.ServeHTTP(w, r) }
func (h *trivialOwner) Shutdown(context.Context) error                   { return nil }
func TestServerTLSAndIdleHeaderDeadline(t *testing.T) {
	// Reuse an ephemeral self-signed test certificate; client explicitly trusts
	// it only in this test. Verify r.TLS survives the connection-cap wrapper.
	source := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	certificate := source.TLS.Certificates[0]
	trusted := source.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	source.Close()
	cfg := baseConfig(t)
	seenTLS := make(chan bool, 1)
	h := &trivialOwner{http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { seenTLS <- r.TLS != nil; w.WriteHeader(204) })}
	listener, e := net.Listen("tcp", "127.0.0.1:0")
	if e != nil {
		t.Fatal(e)
	}
	address := listener.Addr().String()
	ctx, cancel := context.WithCancel(context.Background())
	out := &statusWriter{ready: make(chan struct{})}
	done := make(chan error, 1)
	go func() {
		done <- serveOwned(ctx, listener, h, cfg.HTTP, &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{certificate}}, out)
	}()
	<-out.ready
	tr := &http.Transport{TLSClientConfig: trusted}
	client := &http.Client{Transport: tr, Timeout: 3 * time.Second}
	res, e := client.Get("https://" + address)
	if e != nil {
		t.Fatal(e)
	}
	res.Body.Close()
	if !<-seenTLS || res.StatusCode != 204 {
		t.Fatal("TLS state lost")
	}
	tr.CloseIdleConnections()
	conn, e := net.Dial("tcp", address)
	if e != nil {
		t.Fatal(e)
	}
	conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	var one [1]byte
	if _, e = conn.Read(one[:]); e == nil {
		t.Fatal("idle handshake survived timeout")
	}
	conn.Close()
	cancel()
	if e = <-done; e != nil {
		t.Fatal(e)
	}
}
func TestConnectionCapReleaseAndClose(t *testing.T) {
	listener, e := net.Listen("tcp", "127.0.0.1:0")
	if e != nil {
		t.Fatal(e)
	}
	cap := newCappedListener(listener, 1)
	defer cap.Close()
	peer, e := net.Dial("tcp", listener.Addr().String())
	if e != nil {
		t.Fatal(e)
	}
	defer peer.Close()
	conn, e := cap.Accept()
	if e != nil {
		t.Fatal(e)
	}
	if len(cap.slots) != 1 {
		t.Fatal("slot missing")
	}
	pending := make(chan error, 1)
	go func() { _, e := cap.Accept(); pending <- e }()
	cap.Close()
	if e = <-pending; !errors.Is(e, net.ErrClosed) {
		t.Fatal(e)
	}
	conn.Close()
	conn.Close()
	if len(cap.slots) != 0 {
		t.Fatal("slot leaked")
	}
}
func TestStartRejectsTokenAndTLSBeforeLoading(t *testing.T) {
	cfg := baseConfig(t)
	cfg.AllowExecution = true
	b, _ := json.Marshal(cfg)
	p := cfg.Store + ".json"
	if e := os.WriteFile(p, b, 0600); e != nil {
		t.Fatal(e)
	}
	t.Setenv("SPEECHJOB_TOKEN", "")
	if e := start(context.Background(), p, false, io.Discard); e == nil {
		t.Fatal("missing token")
	}
	cfg.HTTP.TLSCert = "/missing-cert"
	cfg.HTTP.TLSKey = "/missing-key"
	b, _ = json.Marshal(cfg)
	os.WriteFile(p, b, 0600)
	if e := start(context.Background(), p, true, io.Discard); e == nil {
		t.Fatal("bad TLS")
	}
	if _, e := os.Stat(cfg.Store); !os.IsNotExist(e) {
		t.Fatal("created store")
	}
}

func TestStoreLockPrecedesModelAssetReads(t *testing.T) {
	cfg := baseConfig(t)
	cfg.AllowExecution = true
	// Assets deliberately do not exist: a second process must fail the lock
	// before even attempting model verification/materialisation.
	store, e := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: cfg.Limits.Jobs, MaxUploadBytes: cfg.Limits.UploadBytes, MaxArtifactBytes: cfg.Limits.ArtifactBytes, MaxBytes: cfg.Limits.StoreBytes})
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	b, _ := json.Marshal(cfg)
	path := cfg.Store + ".json"
	if e = os.WriteFile(path, b, 0600); e != nil {
		t.Fatal(e)
	}
	t.Setenv("SPEECHJOB_TOKEN", serverToken)
	if e = start(context.Background(), path, false, io.Discard); e == nil || e.Error() != "store open rejected" {
		t.Fatal("assets reached before store exclusion", e)
	}
}

func TestVulkanServerOwnerClosesAfterHandlerDrain(t *testing.T) {
	cfg := vulkanToyConfig(t)
	cfg.AllowExecution = true
	reserve, e := net.Listen("tcp", "127.0.0.1:0")
	if e != nil {
		t.Fatal(e)
	}
	cfg.HTTP.Listen = reserve.Addr().String()
	cfg.HTTP.Hosts = []string{cfg.HTTP.Listen}
	reserve.Close()
	b, _ := json.Marshal(cfg)
	asset := putAsset(t, t.TempDir(), "server.json", b, 0600)
	t.Setenv("SPEECHJOB_TOKEN", serverToken)
	owner := &fakeProfileOwner{stage: speechjob.Stage{Name: "asr-windows", Version: hashBytes([]byte("device-stage")), Run: func(context.Context, *speechjob.Input, io.Writer) error { return nil }}}
	runtime := vulkanProfileRuntime{init: func() bool { return true }, deviceName: func() string { return "fixture-device-1" }, newEncoder: func(context.Context, *whisper.Encoder, int) (*whisper.VulkanEncoder, error) {
		return &whisper.VulkanEncoder{}, nil
	}, newStage: func(*whisper.Whisper, *whisper.Tokenizer, *whisper.VulkanEncoder, speechjob.VulkanWhisperStageConfig) (stageOwner, error) {
		return owner, nil
	}}
	ctx, cancel := context.WithCancel(context.Background())
	out := &statusWriter{ready: make(chan struct{})}
	done := make(chan error, 1)
	go func() { done <- startWithRuntime(ctx, asset.Path, false, out, runtime) }()
	<-out.ready
	if owner.closed != 0 {
		t.Fatal("owner closed while server live")
	}
	cancel()
	if e := <-done; e != nil {
		t.Fatal(e)
	}
	if owner.closed != 1 {
		t.Fatal("owner not closed after handler drain", owner.closed)
	}
	reopened, e := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: cfg.Limits.Jobs, MaxUploadBytes: cfg.Limits.UploadBytes, MaxArtifactBytes: cfg.Limits.ArtifactBytes, MaxBytes: cfg.Limits.StoreBytes})
	if e != nil {
		t.Fatal("store retained after owner close", e)
	}
	reopened.Close()
}
func TestVulkanServerStartupFailureDrainsOwner(t *testing.T) {
	cfg := vulkanToyConfig(t)
	cfg.AllowExecution = true
	occupied, e := net.Listen("tcp", "127.0.0.1:0")
	if e != nil {
		t.Fatal(e)
	}
	defer occupied.Close()
	cfg.HTTP.Listen = occupied.Addr().String()
	cfg.HTTP.Hosts = []string{cfg.HTTP.Listen}
	b, _ := json.Marshal(cfg)
	asset := putAsset(t, t.TempDir(), "server.json", b, 0600)
	t.Setenv("SPEECHJOB_TOKEN", serverToken)
	owner := &fakeProfileOwner{stage: speechjob.Stage{Name: "asr-windows", Version: hashBytes([]byte("device-stage")), Run: func(context.Context, *speechjob.Input, io.Writer) error { return nil }}, failOnce: true}
	runtime := vulkanProfileRuntime{init: func() bool { return true }, deviceName: func() string { return "fixture-device-1" }, newEncoder: func(context.Context, *whisper.Encoder, int) (*whisper.VulkanEncoder, error) {
		return &whisper.VulkanEncoder{}, nil
	}, newStage: func(*whisper.Whisper, *whisper.Tokenizer, *whisper.VulkanEncoder, speechjob.VulkanWhisperStageConfig) (stageOwner, error) {
		return owner, nil
	}}
	if e = startWithRuntime(context.Background(), asset.Path, false, io.Discard, runtime); e == nil {
		t.Fatal("occupied listener")
	}
	if owner.closed != 2 {
		t.Fatal("owner close not retried", owner.closed)
	}
	reopened, e := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: cfg.Limits.Jobs, MaxUploadBytes: cfg.Limits.UploadBytes, MaxArtifactBytes: cfg.Limits.ArtifactBytes, MaxBytes: cfg.Limits.StoreBytes})
	if e != nil {
		t.Fatal(e)
	}
	reopened.Close()
}
func TestVulkanCheckDoesNotInitialiseRuntime(t *testing.T) {
	cfg := vulkanToyConfig(t)
	b, _ := json.Marshal(cfg)
	asset := putAsset(t, t.TempDir(), "server.json", b, 0600)
	calls := 0
	r := vulkanProfileRuntime{init: func() bool { calls++; return false }}
	var out bytes.Buffer
	if e := startWithRuntime(context.Background(), asset.Path, true, &out, r); e != nil {
		t.Fatal(e)
	}
	if calls != 0 {
		t.Fatal("check initialized Vulkan")
	}
	if !bytes.Contains(out.Bytes(), []byte(`"model_loaded":false`)) {
		t.Fatal(out.String())
	}
}
