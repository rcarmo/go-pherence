//go:build linux || darwin || freebsd || openbsd || netbsd || dragonfly

package httpapi

import (
	"bufio"
	"context"
	"io"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func crashHTTP(t *testing.T, root string, child bool, firstCalls *int) (*Handler, *speechjob.Store) {
	t.Helper()
	s, e := speechjob.Open(root, speechjob.Limits{MaxJobs: 16, MaxUploadBytes: 4096, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20})
	if e != nil {
		t.Fatal(e)
	}
	h, e := New(Config{Store: s, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1024, MaxConcurrentRequests: 2, Profiles: []Profile{{ID: "test", Configuration: []byte(`{"test":"crash"}`), Stages: []speechjob.Stage{
		testStage("transcript", func(_ context.Context, _ *speechjob.Input, w io.Writer) error {
			*firstCalls++
			_, e := io.WriteString(w, "preserved text")
			return e
		}),
		testStage("later", func(context.Context, *speechjob.Input, io.Writer) error {
			if child {
				os.Stdout.WriteString("HTTP_CRASH_READY\n")
				time.Sleep(time.Hour)
			}
			return nil
		}),
	}}}})
	if e != nil {
		s.Close()
		t.Fatal(e)
	}
	return h, s
}
func TestHTTPProcessCrashChild(t *testing.T) {
	root := os.Getenv("SPEECH_HTTP_CRASH_ROOT")
	if root == "" {
		return
	}
	calls := 0
	h, s := crashHTTP(t, root, true, &calls)
	defer s.Close()
	defer h.Shutdown(context.Background())
	r := httptest.NewRequest("POST", "https://speech.test/v1/jobs/"+os.Getenv("SPEECH_HTTP_CRASH_ID")+"/run", nil)
	r.Header.Set("Authorization", "Bearer "+testToken)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	t.Fatal("barrier not reached", w.Body.String())
}
func TestHTTPProcessKillReopenAndResume(t *testing.T) {
	root := filepath.Join(t.TempDir(), "store")
	calls := 0
	h, s := crashHTTP(t, root, false, &calls)
	j := upload(t, h)
	if e := h.Shutdown(context.Background()); e != nil {
		t.Fatal(e)
	}
	if e := s.Close(); e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestHTTPProcessCrashChild$", "-test.timeout=30s")
	cmd.Env = append(os.Environ(), "SPEECH_HTTP_CRASH_ROOT="+root, "SPEECH_HTTP_CRASH_ID="+j.ID)
	pipe, e := cmd.StdoutPipe()
	if e != nil {
		t.Fatal(e)
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if e = cmd.Start(); e != nil {
		t.Fatal(e)
	}
	scanner := bufio.NewScanner(pipe)
	ready := false
	for scanner.Scan() {
		if scanner.Text() == "HTTP_CRASH_READY" {
			ready = true
			break
		}
	}
	if !ready {
		cmd.Process.Kill()
		cmd.Wait()
		t.Fatal("missing barrier", stderr.String())
	}
	if e = cmd.Process.Kill(); e != nil {
		t.Fatal(e)
	}
	if e = cmd.Wait(); e == nil {
		t.Fatal("expected process death")
	}
	h, s = crashHTTP(t, root, false, &calls)
	defer s.Close()
	defer h.Shutdown(context.Background())
	stored := decodeJob(t, request(h, "GET", "/v1/jobs/"+j.ID, nil), 200)
	if stored.Status != speechjob.Failed || len(stored.Artifacts) != 1 {
		t.Fatal(stored)
	}
	w := request(h, "GET", "/v1/jobs/"+j.ID+"/artifacts/transcript", nil)
	if w.Code != 200 || w.Body.String() != "preserved text" {
		t.Fatal(w)
	}
	stored = decodeJob(t, request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil), 200)
	if stored.Status != speechjob.Complete || stored.Attempts != 2 || calls != 0 {
		t.Fatal(stored, calls)
	}
}
