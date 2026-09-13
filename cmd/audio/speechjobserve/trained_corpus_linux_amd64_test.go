//go:build linux && amd64

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

// TestTrainedCombinedHTTPCorpus runs one arbitrary hash-pinned canonical WAV
// through the real HTTP profile. Diagnostic tie resolution may persist internal
// diarization, but speaker publication must still fail closed. The opt-in output
// directory receives only the retained plain transcript and internal diarization
// checkpoint for offline quality scoring; neither becomes a public HTTP artifact.
func TestTrainedCombinedHTTPCorpus(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_TRAINED_COMBINED_CORPUS") != "1" {
		t.Skip("set GO_PHERENCE_TEST_TRAINED_COMBINED_CORPUS=1 in an authorised CPU/model window")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 6*time.Minute {
		t.Fatal("trained combined corpus test requires -timeout<=6m")
	}
	if os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "1" || os.Getenv("GOMAXPROCS") != "2" || runtime.GOMAXPROCS(0) != 2 {
		t.Fatal("run with GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2")
	}
	corpusPath := os.Getenv("SPEECHJOB_TRAINED_CORPUS_WAV")
	corpusHash := os.Getenv("SPEECHJOB_TRAINED_CORPUS_WAV_SHA256")
	corpusSamples, err := strconv.ParseInt(os.Getenv("SPEECHJOB_TRAINED_CORPUS_SAMPLES"), 10, 64)
	if corpusPath == "" || !validHash(corpusHash) || corpusSamples < 1 || corpusSamples > 128*16000+144000 {
		t.Fatal("explicit corpus WAV, lowercase SHA-256 and bounded sample count required")
	}
	corpus := trainedPinnedAsset(t, corpusPath, corpusHash)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	listen := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	cfg, _, _ := trainedCombinedConfig(t, listen)
	if os.Getenv("SPEECHJOB_TRAINED_CORPUS_OVERLAP") == "1" {
		cfg.Profile.Community.Modes.OverlapBranches = true
	}
	cfg.Store = filepath.Join(t.TempDir(), "store")
	cfg.Limits.Jobs = 1
	cfg.Profile.MaxDurationSeconds = int((corpusSamples + 15999) / 16000)
	cfg.Profile.DecodeBytes = corpusSamples*2 + 4096
	cfg.HTTP.RequestSeconds = 300
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(t.TempDir(), "trained-corpus.json")
	if err := os.WriteFile(configPath, raw, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SPEECHJOB_TOKEN", serverToken)
	ctx, cancel := context.WithCancel(context.Background())
	status := &statusWriter{ready: make(chan struct{})}
	done := make(chan error, 1)
	go func() { done <- start(ctx, configPath, false, status) }()
	stopped := false
	shutdown := func() error {
		if stopped {
			return nil
		}
		stopped = true
		cancel()
		select {
		case err := <-done:
			return err
		case <-time.After(45 * time.Second):
			return fmt.Errorf("trained corpus server did not drain")
		}
	}
	defer func() {
		if err := shutdown(); err != nil {
			t.Error(err)
		}
	}()
	select {
	case <-status.ready:
	case err := <-done:
		done <- err
		t.Fatal("trained corpus startup failed", err)
	case <-time.After(60 * time.Second):
		t.Fatal("trained corpus server did not listen")
	}
	input, err := os.ReadFile(corpus.Path)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Timeout: 310 * time.Second}
	defer client.CloseIdleConnections()
	call := func(method, path string, body io.Reader) (int, []byte) {
		t.Helper()
		request, err := http.NewRequest(method, "http://"+listen+path, body)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+serverToken)
		if body != nil {
			request.Header.Set("Content-Type", "application/octet-stream")
		}
		response, err := client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		data, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
		if err != nil {
			t.Fatal(err)
		}
		return response.StatusCode, data
	}
	code, body := call("POST", "/v1/jobs?profile=trained-combined-en&name=corpus.wav", bytes.NewReader(input))
	if code != http.StatusCreated {
		t.Fatalf("create status=%d body=%s", code, body)
	}
	var created struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &created); err != nil || created.ID == "" {
		t.Fatal(err, string(body))
	}
	runStarted := time.Now()
	code, body = call("POST", "/v1/jobs/"+created.ID+"/run", nil)
	runElapsed := time.Since(runStarted)
	if code != http.StatusConflict || !bytes.Contains(body, []byte(`"status":"failed"`)) || !bytes.Contains(body, []byte(`"name":"transcript"`)) || !bytes.Contains(body, []byte(`"name":"vtt"`)) || bytes.Contains(body, []byte(`"name":"speaker-transcript"`)) {
		t.Fatalf("corpus did not fail closed after plain artifacts: status=%d body=%s", code, body)
	}
	if err := shutdown(); err != nil {
		t.Fatal(err)
	}
	store, err := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: cfg.Limits.Jobs, MaxUploadBytes: cfg.Limits.UploadBytes, MaxArtifactBytes: cfg.Limits.ArtifactBytes, MaxBytes: cfg.Limits.StoreBytes})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manifest, err := store.Get(created.ID)
	if err != nil || manifest.Status != speechjob.Failed || len(manifest.Checkpoints) != 5 || !strings.Contains(manifest.Error, "configuration or stage identity changed") {
		t.Fatal("retained corpus manifest", manifest.Status, len(manifest.Checkpoints), manifest.Error, err)
	}
	readCheckpoint := func(name string) []byte {
		t.Helper()
		reader, err := store.OpenCheckpoint(context.Background(), created.ID, name)
		if err != nil {
			t.Fatal(err)
		}
		data, readErr := io.ReadAll(io.LimitReader(reader, 32<<20))
		closeErr := reader.Close()
		if readErr != nil || closeErr != nil || len(data) == 0 {
			t.Fatal(name, readErr, closeErr, len(data))
		}
		return data
	}
	transcriptBytes := readCheckpoint("transcript")
	diarizationBytes := readCheckpoint("diarization")
	transcript, err := speechjob.ReadTranscriptJSON(context.Background(), bytes.NewReader(transcriptBytes))
	if err != nil || transcript.TotalSamples != corpusSamples || len(transcript.Words) == 0 {
		t.Fatal("retained transcript", transcript.TotalSamples, len(transcript.Words), err)
	}
	diarization, err := speechjob.ReadDiarizationJSON(context.Background(), bytes.NewReader(diarizationBytes))
	if err != nil || diarization.TotalSamples != corpusSamples || len(diarization.AmbiguousFrames) == 0 {
		t.Fatal("retained diarization", diarization.TotalSamples, len(diarization.AmbiguousFrames), err)
	}
	summary := map[string]any{"schema": 1, "job": created.ID, "run_ns": runElapsed.Nanoseconds(), "samples": corpusSamples, "cues": len(transcript.Cues), "words": len(transcript.Words), "clusters": diarization.Clusters, "ambiguous_frames": len(diarization.AmbiguousFrames), "checkpoints": len(manifest.Checkpoints), "overlap_branches": cfg.Profile.Community.Modes.OverlapBranches, "speaker_publication": false, "tie_policy": "lowest-index-diagnostic", "qualified": false}
	out := os.Getenv("SPEECHJOB_TRAINED_CORPUS_OUTPUT")
	if out != "" {
		out, err = filepath.Abs(out)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(out, 0700); err != nil {
			t.Fatal(err)
		}
		write := func(name string, data []byte) {
			t.Helper()
			if err := os.WriteFile(filepath.Join(out, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		write("transcript.json", transcriptBytes)
		write("diarization.json", diarizationBytes)
		summaryJSON, _ := json.Marshal(summary)
		write("summary.json", append(summaryJSON, '\n'))
	}
	for _, data := range [][]byte{transcriptBytes, diarizationBytes} {
		digest := sha256.Sum256(data)
		t.Log("TRAINED_CORPUS_CHECKPOINT_SHA256 " + hex.EncodeToString(digest[:]))
	}
	summaryJSON, _ := json.Marshal(summary)
	t.Log("TRAINED_COMBINED_CORPUS " + string(summaryJSON))
}
