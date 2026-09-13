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
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

const (
	trainedTinyWeightsSHA    = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
	trainedTinyConfigSHA     = "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98"
	trainedTinyGenerationSHA = "a5d5325911f16e74001a72fa13d6e208eee51548f994646de1f4b4cc8b35b512"
	trainedTinyTokenizerSHA  = "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566"
	trainedSegmentationSHA   = "b4913f3f13c3031267db5d99b0daafd251c822a5f12435b581607e1a56b7c488"
	trainedFiltersSHA        = "2d6ecf47b166b2d44a06aea6546788c816c12800fc47fc265a6752b0a085c705"
	trainedEmbeddingSHA      = "ac0257bbcdd97fe8a0acdf9c670d8a2674ad38de90d3572fafadcc14ceb43391"
	trainedXVectorSHA        = "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f"
	trainedPLDASHA           = "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255"
	trainedPublicWAVSHA      = "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e"
	trainedJFKWAVSHA         = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
)

func trainedPinnedAsset(t *testing.T, path, want string) Asset {
	t.Helper()
	path, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.New()
	_, readErr := io.Copy(digest, file)
	closeErr := file.Close()
	if readErr != nil || closeErr != nil {
		t.Fatal(readErr, closeErr)
	}
	got := hex.EncodeToString(digest.Sum(nil))
	if got != want {
		t.Fatalf("asset %s sha256=%s want %s", filepath.Base(path), got, want)
	}
	return Asset{Path: path, SHA256: want}
}

func trainedRequiredDir(t *testing.T, name string) string {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		t.Fatalf("set %s", name)
	}
	absolute, err := filepath.Abs(value)
	if err != nil {
		t.Fatal(err)
	}
	return absolute
}

func trainedRuntimeSHA(t *testing.T) string {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(executable)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.New()
	_, readErr := io.Copy(digest, file)
	closeErr := file.Close()
	if readErr != nil || closeErr != nil {
		t.Fatal(readErr, closeErr)
	}
	return hex.EncodeToString(digest.Sum(nil))
}

func trainedCombinedConfig(t *testing.T, listen string) (ServerConfig, string, string) {
	t.Helper()
	whisperDir := trainedRequiredDir(t, "SPEECHJOB_TRAINED_WHISPER_DIR")
	segDir := trainedRequiredDir(t, "SPEECHJOB_TRAINED_SEGMENTATION_DIR")
	embDir := trainedRequiredDir(t, "SPEECHJOB_TRAINED_EMBEDDING_DIR")
	pldaDir := trainedRequiredDir(t, "SPEECHJOB_TRAINED_PLDA_DIR")
	publicWAV := os.Getenv("SPEECHJOB_TRAINED_PUBLIC_WAV")
	jfkWAV := os.Getenv("SPEECHJOB_TRAINED_JFK_WAV")
	if publicWAV == "" || jfkWAV == "" {
		t.Fatal("set SPEECHJOB_TRAINED_PUBLIC_WAV and SPEECHJOB_TRAINED_JFK_WAV")
	}
	public := trainedPinnedAsset(t, publicWAV, trainedPublicWAVSHA)
	jfk := trainedPinnedAsset(t, jfkWAV, trainedJFKWAVSHA)
	community := CommunitySettings{
		Enable: true, AllowExperimental: true,
		ModelRevision:      "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
		Segmentation:       trainedPinnedAsset(t, filepath.Join(segDir, "segmentation.safetensors"), trainedSegmentationSHA),
		Filters:            trainedPinnedAsset(t, filepath.Join(segDir, "filters.safetensors"), trainedFiltersSHA),
		Embedding:          trainedPinnedAsset(t, filepath.Join(embDir, "embedding.safetensors"), trainedEmbeddingSHA),
		XVectorTransform:   trainedPinnedAsset(t, filepath.Join(pldaDir, "xvec_transform.npz"), trainedXVectorSHA),
		PLDA:               trainedPinnedAsset(t, filepath.Join(pldaDir, "plda.npz"), trainedPLDASHA),
		SegmentationConfig: CommunitySegmentationSettings{SincNetStride: 10, LSTM: CommunityLSTMSettings{InputSize: 60, HiddenSize: 128, NumLayers: 4, Bidirectional: true}, Head: CommunityHeadSettings{InputSize: 256, HiddenSize: 128, NumLayers: 2, Speakers: 3, MaxActive: 2}},
		EmbeddingConfig:    CommunityEmbeddingSettings{BaseChannels: 32, MelBins: 80, EmbedDim: 256},
		EmbeddingPrefix:    "resnet", PLDAConfig: CommunityPLDASettings{InputDim: 256, ProjectedDim: 128, OutputDim: 128},
		PCM:            CommunityPCMSettings{WindowSamples: 160000, StepSamples: 16000, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 64, AHCThreshold: .6, Fa: .07, Fb: .8, Constrained: true, TiePolicy: "lowest-index"},
		Modes:          CommunityModesSettings{SincNet: "simd", LSTM: "simd", Head: "simd", Embedding: "simd"},
		MaxResultBytes: 16 << 20,
	}
	cfg := ServerConfig{
		Schema: 2, AllowExecution: true, Store: filepath.Join(t.TempDir(), "store"), RuntimeSHA256: trainedRuntimeSHA(t), Threads: runtime.GOMAXPROCS(0),
		Resources:   &ResourceSettings{CPUSlots: 2, MemoryBytes: 6 << 30, MaxWaiting: 2, LoadBytes: 5 << 30, ResidentBytes: 4 << 30, WorkBytes: 2 << 30},
		Limits:      Limits{Jobs: 4, UploadBytes: 16 << 20, ArtifactBytes: 64 << 20, StoreBytes: 512 << 20, WeightBytes: 512 << 20, OwnedWeightBytes: 512 << 20},
		HTTP:        HTTPSettings{Listen: listen, Hosts: []string{listen}, AllowLoopbackHTTP: true, RequestSeconds: 180, HeaderSeconds: 10, IdleSeconds: 30, ShutdownSeconds: 30, MaxRequests: 2, MaxConnections: 4},
		Weights:     trainedPinnedAsset(t, filepath.Join(whisperDir, "model.safetensors"), trainedTinyWeightsSHA),
		ModelConfig: trainedPinnedAsset(t, filepath.Join(whisperDir, "config.json"), trainedTinyConfigSHA),
		Tokenizer:   trainedPinnedAsset(t, filepath.Join(whisperDir, "tokenizer.json"), trainedTinyTokenizerSHA),
		Generation:  trainedPinnedAsset(t, filepath.Join(whisperDir, "generation_config.json"), trainedTinyGenerationSHA),
		Profile:     ProfileSettings{ID: "trained-combined-en", Language: "en", Extension: ".wav", MediaBackend: "go264", MaxDurationSeconds: 31, DecodeBytes: 2 << 20, MaxNewTokens: 192, WordTimestamps: true, WindowBytes: 1 << 20, ResultBytes: 16 << 20, Community: &community},
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	return cfg, public.Path, jfk.Path
}

// TestTrainedCombinedHTTPPublicSample is an explicit diagnostic gate for the
// actual seven-stage server profile. It uses Tiny, so transcript quality and
// speaker-attributed WER are not release-qualified. LowestIndexTies is explicit
// because the source-default tie identity is unspecified on this sample.
func TestTrainedCombinedHTTPPublicSample(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_TRAINED_COMBINED") != "1" {
		t.Skip("set GO_PHERENCE_TEST_TRAINED_COMBINED=1 in an authorised CPU/model window")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 6*time.Minute {
		t.Fatal("trained combined service test requires -timeout<=6m")
	}
	if os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "1" || os.Getenv("GOMAXPROCS") != "2" || runtime.GOMAXPROCS(0) != 2 {
		t.Fatal("run with GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	listen := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	cfg, publicWAV, jfkWAV := trainedCombinedConfig(t, listen)
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(t.TempDir(), "trained-combined.json")
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
			return fmt.Errorf("trained combined server did not drain")
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
		t.Fatal("trained combined startup failed", err)
	case <-time.After(60 * time.Second):
		t.Fatal("trained combined server did not listen")
	}
	loadElapsed := time.Since(started)
	client := &http.Client{Timeout: 190 * time.Second}
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
	create := func(name, path string) string {
		t.Helper()
		input, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		code, body := call("POST", "/v1/jobs?profile=trained-combined-en&name="+name, bytes.NewReader(input))
		if code != http.StatusCreated {
			t.Fatalf("create %s status=%d body=%s", name, code, body)
		}
		var created struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(body, &created); err != nil || created.ID == "" {
			t.Fatal(err, string(body))
		}
		return created.ID
	}
	publicID := create("sample.wav", publicWAV)
	code, body := call("POST", "/v1/jobs/"+publicID+"/run", nil)
	if code != http.StatusConflict || !bytes.Contains(body, []byte(`"status":"failed"`)) || !bytes.Contains(body, []byte(`"name":"transcript"`)) || !bytes.Contains(body, []byte(`"name":"vtt"`)) || bytes.Contains(body, []byte(`"name":"speaker-transcript"`)) {
		t.Fatalf("ambiguous public sample did not fail closed with plain artifacts: status=%d body=%s", code, body)
	}
	var createdID string
	var runTimes []int64
	var hashes map[string]string
	var plain speechjob.Transcript
	var speaker speechjob.SpeakerTranscript
	for repeat := 0; repeat < 3; repeat++ {
		createdID = create(fmt.Sprintf("jfk-%d.wav", repeat), jfkWAV)
		runStarted := time.Now()
		code, body = call("POST", "/v1/jobs/"+createdID+"/run", nil)
		runTimes = append(runTimes, time.Since(runStarted).Nanoseconds())
		if code != http.StatusOK {
			if err := shutdown(); err != nil {
				t.Fatal(err)
			}
			store, err := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: cfg.Limits.Jobs, MaxUploadBytes: cfg.Limits.UploadBytes, MaxArtifactBytes: cfg.Limits.ArtifactBytes, MaxBytes: cfg.Limits.StoreBytes})
			if err != nil {
				t.Fatalf("JFK run status=%d body=%s; reopen=%v", code, body, err)
			}
			manifest, getErr := store.Get(createdID)
			closeErr := store.Close()
			t.Fatalf("JFK run status=%d body=%s; private_error=%q checkpoints=%d get=%v close=%v", code, body, manifest.Error, len(manifest.Checkpoints), getErr, closeErr)
		}
		artifacts := map[string][]byte{}
		for _, name := range []string{"transcript", "vtt", "speaker-transcript", "speaker-vtt"} {
			code, artifact := call("GET", "/v1/jobs/"+createdID+"/artifacts/"+name, nil)
			if code != http.StatusOK || len(artifact) == 0 {
				t.Fatalf("artifact %s status=%d bytes=%d", name, code, len(artifact))
			}
			artifacts[name] = artifact
		}
		plain, err = speechjob.ReadTranscriptJSON(context.Background(), bytes.NewReader(artifacts["transcript"]))
		if err != nil || plain.TotalSamples != 176000 || len(plain.Cues) == 0 || len(plain.Words) == 0 {
			t.Fatal("plain transcript", plain.TotalSamples, len(plain.Cues), len(plain.Words), err)
		}
		speaker, err = speechjob.ReadSpeakerTranscriptJSON(context.Background(), bytes.NewReader(artifacts["speaker-transcript"]))
		if err != nil || speaker.Transcript.TotalSamples != plain.TotalSamples || len(speaker.Transcript.Words) != len(plain.Words) || speaker.LabelledWords+speaker.UnlabelledWords != len(plain.Words) {
			t.Fatal("speaker transcript", len(speaker.Transcript.Words), speaker.LabelledWords, speaker.UnlabelledWords, err)
		}
		if !strings.HasPrefix(string(artifacts["vtt"]), "WEBVTT\n\n") || !strings.Contains(string(artifacts["speaker-vtt"]), "NOTE Experimental Community-1") {
			t.Fatal("invalid VTT artifacts")
		}
		current := map[string]string{}
		for name, artifact := range artifacts {
			digest := sha256.Sum256(artifact)
			current[name] = hex.EncodeToString(digest[:])
		}
		if hashes == nil {
			hashes = current
		} else {
			for name, want := range hashes {
				if current[name] != want {
					t.Fatalf("nondeterministic %s artifact: %s != %s", name, current[name], want)
				}
			}
		}
	}
	if err := shutdown(); err != nil {
		t.Fatal(err)
	}
	store, err := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: cfg.Limits.Jobs, MaxUploadBytes: cfg.Limits.UploadBytes, MaxArtifactBytes: cfg.Limits.ArtifactBytes, MaxBytes: cfg.Limits.StoreBytes})
	if err != nil {
		t.Fatal(err)
	}
	publicManifest, publicErr := store.Get(publicID)
	jfkManifest, jfkErr := store.Get(createdID)
	diarReader, diarOpenErr := store.OpenCheckpoint(context.Background(), publicID, "diarization")
	var diar speechjob.DiarizationDocument
	var diarReadErr, diarCloseErr error
	if diarOpenErr == nil {
		diar, diarReadErr = speechjob.ReadDiarizationJSON(context.Background(), diarReader)
		diarCloseErr = diarReader.Close()
	}
	closeErr := store.Close()
	if publicErr != nil || jfkErr != nil || diarOpenErr != nil || diarReadErr != nil || diarCloseErr != nil || closeErr != nil || publicManifest.Status != speechjob.Failed || len(publicManifest.Checkpoints) != 5 || !strings.Contains(publicManifest.Error, "configuration or stage identity changed") || len(diar.AmbiguousFrames) != 84 || jfkManifest.Status != speechjob.Complete || len(jfkManifest.Checkpoints) != 7 {
		t.Fatal("retained manifests", publicManifest.Status, len(publicManifest.Checkpoints), publicManifest.Error, len(diar.AmbiguousFrames), jfkManifest.Status, len(jfkManifest.Checkpoints), publicErr, jfkErr, diarOpenErr, diarReadErr, diarCloseErr, closeErr)
	}
	summary := map[string]any{"schema": 1, "job": createdID, "startup_ready_ns": loadElapsed.Nanoseconds(), "run_ns": runTimes, "repeat_artifacts_identical": true, "total_samples": plain.TotalSamples, "cues": len(plain.Cues), "words": len(plain.Words), "labelled_words": speaker.LabelledWords, "unlabelled_words": speaker.UnlabelledWords, "artifact_sha256": hashes, "public_ambiguous_job": publicID, "public_ambiguous_checkpoints": len(publicManifest.Checkpoints), "public_ambiguous_frames": len(diar.AmbiguousFrames), "internal_diarization_downloadable": false, "tie_policy": "lowest-index", "qualified": false}
	summaryJSON, _ := json.Marshal(summary)
	t.Log("TRAINED_COMBINED_SERVICE " + string(summaryJSON))
	if output := os.Getenv("SPEECHJOB_TRAINED_COMBINED_OUTPUT"); output != "" {
		output, err = filepath.Abs(output)
		if err != nil {
			t.Fatal(err)
		}
		file, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		_, writeErr := fmt.Fprintln(file, string(summaryJSON))
		closeErr := file.Close()
		if writeErr != nil || closeErr != nil {
			t.Fatal(writeErr, closeErr)
		}
	}
}
