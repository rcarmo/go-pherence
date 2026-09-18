//go:build linux || darwin || freebsd || openbsd || netbsd || dragonfly

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestCLIProcessChild(t *testing.T) {
	raw := os.Getenv("SPEECHJOB_TEST_CHILD_ARGS")
	if raw == "" {
		return
	}
	var args []string
	if e := json.Unmarshal([]byte(raw), &args); e != nil {
		t.Fatal(e)
	}
	os.Args = append([]string{"speechjob"}, args...)
	main()
}
func TestSIGTERMDownloadCleansPrivateScratch(t *testing.T) {
	entered := make(chan struct{})
	server := httptest.NewServer(downloadServer("WEBVTT\n\n", func(w http.ResponseWriter, r *http.Request, b string) {
		w.Header().Set("Content-Type", "text/vtt")
		w.Header().Set("Content-Length", fmt.Sprint(len(b)))
		w.Header().Set("ETag", `"sha256-`+hashText(b)+`"`)
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		close(entered)
		<-r.Context().Done()
	}))
	defer server.Close()
	dir := t.TempDir()
	destination := filepath.Join(dir, "result.vtt")
	args, _ := json.Marshal([]string{"--allow-loopback-http", "--timeout", "30s", "download", "--job", job, "--artifact", "vtt", "--out", destination})
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCLIProcessChild$", "-test.timeout=30s")
	cmd.Env = append(os.Environ(), "SPEECHJOB_TEST_CHILD_ARGS="+string(args), "SPEECHJOB_URL="+server.URL, "SPEECHJOB_TOKEN="+token)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if e := cmd.Start(); e != nil {
		t.Fatal(e)
	}
	defer func() { _ = cmd.Process.Kill() }()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal("download did not begin")
	}
	// Wait until headers have been accepted and the child has created the actual
	// scratch file. This tests signal cleanup, not only cancellation before IO.
	deadline := time.Now().Add(3 * time.Second)
	found := false
	for time.Now().Before(deadline) {
		matches, _ := filepath.Glob(filepath.Join(dir, ".speechjob-download-*"))
		if len(matches) > 0 {
			found = true
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !found {
		cmd.Process.Kill()
		cmd.Wait()
		t.Fatal("no download scratch", stderr.String())
	}
	if e := cmd.Process.Signal(syscall.SIGTERM); e != nil {
		t.Fatal(e)
	}
	e := cmd.Wait()
	var exit *exec.ExitError
	if !errors.As(e, &exit) || exit.ExitCode() != 1 {
		t.Fatal("expected controlled error exit", e, stderr.String())
	}
	if state, ok := exit.Sys().(syscall.WaitStatus); ok && state.Signaled() {
		t.Fatal("terminated without cleanup", state)
	}
	assertNoTemp(t, dir)
	if _, e := os.Stat(destination); !errors.Is(e, os.ErrNotExist) {
		t.Fatal("partial output published", e)
	}
	if strings.Contains(stderr.String(), token) {
		t.Fatal("token leaked")
	}
}
