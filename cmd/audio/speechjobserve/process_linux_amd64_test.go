//go:build linux && amd64

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func TestServeProcessChild(t *testing.T) {
	path := os.Getenv("SPEECHSERVE_CHILD_CONFIG")
	if path == "" {
		return
	}
	os.Args = []string{"speechjobserve", "--config", path}
	main()
}
func TestServeProcessSIGTERMClosesListenerAndStore(t *testing.T) {
	testServeProcessSIGTERM(t, false, false)
}
func TestServeQueueProcessSIGTERMClosesLocks(t *testing.T) {
	testServeProcessSIGTERM(t, true, false)
}
func TestServeResourceProcessSIGTERM(t *testing.T) { testServeProcessSIGTERM(t, true, true) }
func testServeProcessSIGTERM(t *testing.T, queued, resources bool) {
	cfg := toyAssets(t)
	cfg.AllowExecution = true
	if resources {
		cfg.Resources = &ResourceSettings{CPUSlots: cfg.Threads, MemoryBytes: 64 << 20, MaxWaiting: 4, LoadBytes: 32 << 20, ResidentBytes: 16 << 20, WorkBytes: 16 << 20}
	}
	if queued {
		cfg.Queue = QueueSettings{Enable: true, StartWorker: true, Directory: filepath.Join(t.TempDir(), "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobSeconds: 5}
	}
	reservation, e := net.Listen("tcp", "127.0.0.1:0")
	if e != nil {
		t.Fatal(e)
	}
	cfg.HTTP.Listen = reservation.Addr().String()
	cfg.HTTP.Hosts = []string{cfg.HTTP.Listen}
	reservation.Close()
	b, _ := json.Marshal(cfg)
	asset := putAsset(t, t.TempDir(), "server.json", b, 0600)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeProcessChild$", "-test.timeout=30s")
	cmd.Env = append(os.Environ(), "SPEECHSERVE_CHILD_CONFIG="+asset.Path, "SPEECHJOB_TOKEN="+serverToken)
	stdout, e := cmd.StdoutPipe()
	if e != nil {
		t.Fatal(e)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if e = cmd.Start(); e != nil {
		t.Fatal(e)
	}
	defer cmd.Process.Kill()
	scanner := bufio.NewScanner(stdout)
	ready := false
	for scanner.Scan() {
		if strings.Contains(scanner.Text(), `"listening"`) {
			ready = true
			break
		}
	}
	if !ready {
		cmd.Process.Kill()
		cmd.Wait()
		t.Fatal("listener never ready", stderr.String())
	}
	request, _ := http.NewRequest("GET", "http://"+cfg.HTTP.Listen+"/v1/jobs", nil)
	request.Header.Set("Authorization", "Bearer "+serverToken)
	client := &http.Client{Timeout: 3 * time.Second}
	response, e := client.Do(request)
	if e != nil {
		t.Fatal(e)
	}
	io.Copy(io.Discard, response.Body)
	response.Body.Close()
	client.CloseIdleConnections()
	if response.StatusCode != 200 {
		t.Fatal(response.Status)
	}
	if e = cmd.Process.Signal(syscall.SIGTERM); e != nil {
		t.Fatal(e)
	}
	if e = cmd.Wait(); e != nil {
		t.Fatal("graceful process exit", e, stderr.String())
	}
	if strings.Contains(stderr.String(), serverToken) {
		t.Fatal("token leak")
	}
	reopened, e := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: cfg.Limits.Jobs, MaxUploadBytes: cfg.Limits.UploadBytes, MaxArtifactBytes: cfg.Limits.ArtifactBytes, MaxBytes: cfg.Limits.StoreBytes})
	if e != nil {
		t.Fatal("store lock not released", e)
	}
	if queued {
		q, e := speechjob.OpenQueue(reopened, speechjob.QueueConfig{Directory: cfg.Queue.Directory, MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: time.Second, Admission: speechjob.SerialAdmission(), Resolve: func(speechjob.Manifest) ([]speechjob.Stage, error) { return nil, speechjob.ErrConfiguration }})
		if e != nil {
			t.Fatal("queue lock not released", e)
		}
		q.Close()
	}
	reopened.Close()
	listener, e := net.Listen("tcp", cfg.HTTP.Listen)
	if e != nil {
		t.Fatal("listener not released", e)
	}
	listener.Close()
	t.Log(fmt.Sprintf("child SIGTERM graceful; no model inference requested"))
}
