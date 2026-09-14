package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

// TestBrowserFixtureServer is a gated model-free fixture for the Playwright
// script. It never starts during ordinary tests and only uses an explicit temp
// store and loopback port; parent owns process termination and fixture deletion.
func TestBrowserFixtureServer(t *testing.T) {
	root := os.Getenv("SPEECHJOB_BROWSER_STORE")
	if root == "" {
		return
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	store, e := speechjob.Open(root, speechjob.Limits{MaxJobs: 100, MaxUploadBytes: 1024, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20})
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	listener, e := net.Listen("tcp", "127.0.0.1:0")
	if e != nil {
		t.Fatal(e)
	}
	origin := "http://" + listener.Addr().String()
	profiles := []Profile{
		{ID: "asr", Configuration: []byte(`{"test":"asr"}`), Stages: []speechjob.Stage{textStage("transcript", `{"text":"Olá <&>"}`), textStage("vtt", "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nOlá &lt;&amp;&gt;\n")}},
		{ID: "fail", Configuration: []byte(`{"test":"failure"}`), Stages: []speechjob.Stage{textStage("transcript", `{"text":"retained"}`), testStage("later", func(context.Context, *speechjob.Input, io.Writer) error {
			return errors.New("private error must not escape")
		})}},
		{ID: "slow", Configuration: []byte(`{"test":"cancel"}`), Stages: []speechjob.Stage{testStage("transcript", func(ctx context.Context, _ *speechjob.Input, _ io.Writer) error { <-ctx.Done(); return ctx.Err() })}},
	}
	h, e := New(Config{Store: store, Profiles: profiles, Token: os.Getenv("SPEECHJOB_BROWSER_TOKEN"), Hosts: []string{listener.Addr().String()}, Origin: origin, EnableUI: true, MaxUploadBytes: 1024, MaxConcurrentRequests: 4})
	if e != nil {
		t.Fatal(e)
	}
	server := &http.Server{Handler: h, ReadHeaderTimeout: time.Second, IdleTimeout: time.Second}
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	json.NewEncoder(os.Stdout).Encode(struct {
		Origin string `json:"origin"`
	}{origin})
	<-ctx.Done()
	server.Close()
	h.Shutdown(context.Background())
	<-done
}
