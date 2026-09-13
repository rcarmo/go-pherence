//go:build linux && amd64

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/models/whisper"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func putAsset(t *testing.T, dir, name string, b []byte, mode os.FileMode) Asset {
	t.Helper()
	p := filepath.Join(dir, name)
	if e := os.WriteFile(p, b, mode); e != nil {
		t.Fatal(e)
	}
	return Asset{p, hashBytes(b)}
}
func toyAssets(t *testing.T) ServerConfig {
	t.Helper()
	c := baseConfig(t)
	c.Threads = runtime.GOMAXPROCS(0)
	t.Setenv("GOMAXPROCS", fmt.Sprint(c.Threads))
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	dir := t.TempDir()
	model := map[string]any{"model_type": "whisper", "activation_function": "gelu", "is_encoder_decoder": true, "scale_embedding": false, "architectures": []string{"WhisperForConditionalGeneration"}, "num_mel_bins": 80, "d_model": 2, "encoder_layers": 0, "decoder_layers": 0, "encoder_attention_heads": 1, "decoder_attention_heads": 1, "encoder_ffn_dim": 2, "decoder_ffn_dim": 2, "vocab_size": 51865, "max_source_positions": 1, "max_target_positions": 8, "bos_token_id": 50257, "eos_token_id": 50257, "pad_token_id": 50257, "decoder_start_token_id": 50258}
	generation := map[string]any{"bos_token_id": 50257, "eos_token_id": 50257, "pad_token_id": 50257, "decoder_start_token_id": 50258, "no_timestamps_token_id": 50363, "is_multilingual": true, "lang_to_id": map[string]int{"<|en|>": 50259, "<|pt|>": 50267}, "task_to_id": map[string]int{"translate": 50358, "transcribe": 50359}, "max_length": 8, "max_initial_timestamp_index": 0, "suppress_tokens": []int{}, "begin_suppress_tokens": []int{}}
	langs := map[string]int{}
	for id := 50259; id < 50358; id++ {
		name := fmt.Sprintf("<|lang%d|>", id)
		if id == 50259 {
			name = "<|en|>"
		}
		if id == 50267 {
			name = "<|pt|>"
		}
		langs[name] = id
	}
	generation["lang_to_id"] = langs
	b, _ := json.Marshal(model)
	c.ModelConfig = putAsset(t, dir, "config.json", b, 0600)
	b, _ = json.Marshal(generation)
	c.Generation = putAsset(t, dir, "generation.json", b, 0600)
	vocab := map[string]int{}
	for i := 0; i < 51865; i++ {
		vocab[fmt.Sprintf("text%d", i)] = i
	}
	for id, text := range map[int]string{50257: "<|endoftext|>", 50258: "<|startoftranscript|>", 50259: "<|en|>", 50267: "<|pt|>", 50358: "<|translate|>", 50359: "<|transcribe|>", 50363: "<|notimestamps|>"} {
		delete(vocab, fmt.Sprintf("text%d", id))
		vocab[text] = id
	}
	for name, id := range langs {
		delete(vocab, fmt.Sprintf("text%d", id))
		vocab[name] = id
	}
	for i := 0; i <= 1500; i++ {
		id := 50364 + i
		delete(vocab, fmt.Sprintf("text%d", id))
		vocab[fmt.Sprintf("<|%.2f|>", float64(i)/50)] = id
	}
	b, _ = json.Marshal(map[string]any{"model": map[string]any{"vocab": vocab}, "added_tokens": []any{}, "normalizer": nil})
	c.Tokenizer = putAsset(t, dir, "tokenizer.json", b, 0600)
	type tensor struct {
		shape []int
		v     []float32
	}
	tensors := map[string]tensor{}
	add := func(n string, shape ...int) {
		count := 1
		for _, d := range shape {
			count *= d
		}
		tensors[n] = tensor{shape, make([]float32, count)}
	}
	add("model.encoder.conv1.weight", 2, 80, 3)
	add("model.encoder.conv1.bias", 2)
	add("model.encoder.conv2.weight", 2, 2, 3)
	add("model.encoder.conv2.bias", 2)
	add("model.encoder.embed_positions.weight", 1, 2)
	add("model.encoder.layer_norm.weight", 2)
	add("model.encoder.layer_norm.bias", 2)
	add("model.decoder.embed_tokens.weight", 51865, 2)
	add("model.decoder.embed_positions.weight", 8, 2)
	add("model.decoder.layer_norm.weight", 2)
	add("model.decoder.layer_norm.bias", 2)
	for _, n := range []string{"model.encoder.layer_norm.weight", "model.decoder.layer_norm.weight"} {
		for i := range tensors[n].v {
			tensors[n].v[i] = 1
		}
	}
	tensors["model.decoder.embed_tokens.weight"].v[50257*2] = 10
	tensors["model.decoder.embed_tokens.weight"].v[50257*2+1] = -10
	for i := 0; i < 8; i++ {
		tensors["model.decoder.embed_positions.weight"].v[i*2] = 1
		tensors["model.decoder.embed_positions.weight"].v[i*2+1] = -1
	}
	names := []string{}
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]safetensors.TensorInfo{}
	var payload bytes.Buffer
	for _, name := range names {
		v := tensors[name]
		start := payload.Len()
		for _, f := range v.v {
			binary.Write(&payload, binary.LittleEndian, math.Float32bits(f))
		}
		header[name] = safetensors.TensorInfo{DType: "F32", Shape: v.shape, DataOffsets: [2]int{start, payload.Len()}}
	}
	h, _ := json.Marshal(header)
	var file bytes.Buffer
	binary.Write(&file, binary.LittleEndian, uint64(len(h)))
	file.Write(h)
	file.Write(payload.Bytes())
	c.Weights = putAsset(t, dir, "model.safetensors", file.Bytes(), 0600)
	return c
}
func TestProfileCheckDoesNotCreateStoreOrLoadInference(t *testing.T) {
	c := toyAssets(t)
	raw, _ := json.Marshal(c)
	file := putAsset(t, t.TempDir(), "server.json", raw, 0600)
	var out bytes.Buffer
	if e := start(context.Background(), file.Path, true, &out); e != nil {
		t.Fatal(e)
	}
	if !strings.Contains(out.String(), `"model_loaded":false`) {
		t.Fatal(out.String())
	}
	if _, e := os.Stat(c.Store); !os.IsNotExist(e) {
		t.Fatal("check created store", e)
	}
	if e := start(context.Background(), file.Path, false, &out); e == nil {
		t.Fatal("execution implicit")
	}
	profiles, e := buildProfile(context.Background(), c, true)
	if e != nil || len(profiles) != 1 || len(profiles[0].Stages) != 4 {
		t.Fatal(profiles, e)
	}
	if profiles[0].Stages[0].Name != "decode" || profiles[0].Stages[1].Name != "asr-windows" || profiles[0].Stages[2].Name != "transcript" || profiles[0].Stages[3].Name != "vtt" {
		t.Fatal("profile order")
	}
}
func TestProfileMetadataAdmission(t *testing.T) {
	c := toyAssets(t)
	for _, kind := range []string{"hash", "weightcap", "ownedcap", "overlap", "language", "header", "generation-timestamp", "generation-tokens"} {
		bad := c
		switch kind {
		case "hash":
			bad.Tokenizer.SHA256 = hashBytes([]byte("bad"))
		case "weightcap":
			bad.Limits.WeightBytes = 8
		case "ownedcap":
			bad.Limits.OwnedWeightBytes = 100
		case "overlap":
			bad.Profile.OverlapSamples = 161
		case "language":
			bad.Profile.Language = "zz"
		case "generation-timestamp":
			bad.Profile.MaxInitialTimestampIndex = 1
		case "generation-tokens":
			bad.Profile.MaxNewTokens = 6
		case "header":
			b := make([]byte, 8)
			binary.LittleEndian.PutUint64(b, 9<<20)
			bad.Weights = putAsset(t, t.TempDir(), "bad.safetensors", b, 0600)
		}
		if _, e := buildProfile(context.Background(), bad, false); e == nil {
			t.Fatal(kind)
		}
	}
	t.Setenv("WHISPER_INT8", "1")
	if e := validateRuntime(c); e == nil {
		t.Fatal("runtime mode changed")
	}
}
func TestServingProfileSyntheticGo264ToVTT(t *testing.T) {
	c := toyAssets(t)
	profiles, e := buildProfile(context.Background(), c, true)
	if e != nil {
		t.Fatal(e)
	}
	s, e := speechjob.Open(c.Store, speechjob.Limits{MaxJobs: c.Limits.Jobs, MaxUploadBytes: c.Limits.UploadBytes, MaxArtifactBytes: c.Limits.ArtifactBytes, MaxBytes: c.Limits.StoreBytes})
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	const n = 320
	wav := make([]byte, 44+2*n)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], uint32(len(wav)-8))
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 1)
	binary.LittleEndian.PutUint32(wav[24:], 16000)
	binary.LittleEndian.PutUint32(wav[28:], 32000)
	binary.LittleEndian.PutUint16(wav[32:], 2)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 2*n)
	p := profiles[0]
	j, e := s.Create(context.Background(), "toy.wav", p.Configuration, bytes.NewReader(wav))
	if e != nil {
		t.Fatal(e)
	}
	j, e = s.Run(context.Background(), j.ID, p.Configuration, p.Stages, nil)
	if e != nil || j.Status != speechjob.Complete {
		t.Fatal(j, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), j.ID, "vtt")
	if e != nil {
		t.Fatal(e)
	}
	var b bytes.Buffer
	b.ReadFrom(r)
	r.Close()
	if b.String() != "WEBVTT\n\n" {
		t.Fatal(b.String())
	}
}

func TestServingProfileSyntheticFFmpegToVTT(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_FFMPEG") != "1" {
		t.Skip("explicit FFmpeg integration with generated toy model")
	}
	c := toyAssets(t)
	c.Profile.MediaBackend = "ffmpeg"
	for _, target := range []*Asset{&c.FFmpeg, &c.FFprobe} {
		name := "ffmpeg"
		if target == &c.FFprobe {
			name = "ffprobe"
		}
		p, e := exec.LookPath(name)
		if e != nil {
			t.Fatal(e)
		}
		p, _ = filepath.Abs(p)
		b, e := os.ReadFile(p)
		if e != nil {
			t.Fatal(e)
		}
		*target = Asset{p, hashBytes(b)}
	}
	profiles, e := buildProfile(context.Background(), c, true)
	if e != nil {
		t.Fatal(e)
	}
	s, e := speechjob.Open(c.Store, speechjob.Limits{MaxJobs: c.Limits.Jobs, MaxUploadBytes: c.Limits.UploadBytes, MaxArtifactBytes: c.Limits.ArtifactBytes, MaxBytes: c.Limits.StoreBytes})
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	const n = 320
	wav := make([]byte, 44+2*n)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], uint32(len(wav)-8))
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 1)
	binary.LittleEndian.PutUint32(wav[24:], 16000)
	binary.LittleEndian.PutUint32(wav[28:], 32000)
	binary.LittleEndian.PutUint16(wav[32:], 2)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 2*n)
	p := profiles[0]
	j, e := s.Create(context.Background(), "toy.wav", p.Configuration, bytes.NewReader(wav))
	if e != nil {
		t.Fatal(e)
	}
	j, e = s.Run(context.Background(), j.ID, p.Configuration, p.Stages, nil)
	if e != nil || j.Status != speechjob.Complete {
		t.Fatal(j, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), j.ID, "vtt")
	if e != nil {
		t.Fatal(e)
	}
	var b bytes.Buffer
	b.ReadFrom(r)
	r.Close()
	if b.String() != "WEBVTT\n\n" {
		t.Fatal(b.String())
	}
}

func TestStartQueueWorkerConsent(t *testing.T)         { testStartQueueWorkerConsent(t, false) }
func TestStartQueueWithResourceEstimates(t *testing.T) { testStartQueueWorkerConsent(t, true) }
func testStartQueueWorkerConsent(t *testing.T, resources bool) {
	if os.Getenv("GO_PHERENCE_TEST_FFMPEG") != "1" {
		t.Skip("explicit FFmpeg queue integration with generated toy model")
	}
	modes := []string{"false", "true"}
	if resources {
		modes = append(modes, "sync")
	}
	for _, mode := range modes {
		worker := mode == "true"
		t.Run(mode, func(t *testing.T) {
			c := toyAssets(t)
			c.AllowExecution = true
			c.Profile.MediaBackend = "ffmpeg"
			if resources {
				c.Resources = &ResourceSettings{CPUSlots: 2, MemoryBytes: 64 << 20, MaxWaiting: 4, LoadBytes: 32 << 20, ResidentBytes: 16 << 20, WorkBytes: 16 << 20}
			}
			if mode != "sync" {
				c.Queue = QueueSettings{Enable: true, StartWorker: worker, Directory: filepath.Join(t.TempDir(), "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobSeconds: 5}
			}
			for _, pair := range []struct {
				name  string
				asset *Asset
			}{{"ffmpeg", &c.FFmpeg}, {"ffprobe", &c.FFprobe}} {
				p, e := exec.LookPath(pair.name)
				if e != nil {
					t.Fatal(e)
				}
				p, _ = filepath.Abs(p)
				b, e := os.ReadFile(p)
				if e != nil {
					t.Fatal(e)
				}
				*pair.asset = Asset{p, hashBytes(b)}
			}
			reserve, e := net.Listen("tcp", "127.0.0.1:0")
			if e != nil {
				t.Fatal(e)
			}
			c.HTTP.Listen = reserve.Addr().String()
			c.HTTP.Hosts = []string{c.HTTP.Listen}
			reserve.Close()
			config, _ := json.Marshal(c)
			asset := putAsset(t, t.TempDir(), "server.json", config, 0600)
			t.Setenv("SPEECHJOB_TOKEN", serverToken)
			ctx, cancel := context.WithCancel(context.Background())
			out := &statusWriter{ready: make(chan struct{})}
			done := make(chan error, 1)
			go func() { done <- start(ctx, asset.Path, false, out) }()
			defer func() {
				cancel()
				select {
				case e := <-done:
					if e != nil {
						t.Error(e)
					}
				case <-time.After(5 * time.Second):
					t.Error("startup/worker not drained")
				}
			}()
			select {
			case <-out.ready:
			case e := <-done:
				done <- e
				t.Fatal("startup failed", e)
			case <-time.After(5 * time.Second):
				t.Fatal("not listening")
			}
			client := &http.Client{Timeout: 3 * time.Second}
			defer client.CloseIdleConnections()
			call := func(method, path string, body io.Reader) (int, []byte) {
				t.Helper()
				r, _ := http.NewRequest(method, "http://"+c.HTTP.Listen+path, body)
				r.Header.Set("Authorization", "Bearer "+serverToken)
				if body != nil {
					r.Header.Set("Content-Type", "application/octet-stream")
				}
				res, e := client.Do(r)
				if e != nil {
					t.Fatal(e)
				}
				defer res.Body.Close()
				b, e := io.ReadAll(res.Body)
				if e != nil {
					t.Fatal(e)
				}
				return res.StatusCode, b
			}
			const n = 320
			wav := make([]byte, 44+2*n)
			copy(wav, "RIFF")
			binary.LittleEndian.PutUint32(wav[4:], uint32(len(wav)-8))
			copy(wav[8:], "WAVEfmt ")
			binary.LittleEndian.PutUint32(wav[16:], 16)
			binary.LittleEndian.PutUint16(wav[20:], 1)
			binary.LittleEndian.PutUint16(wav[22:], 1)
			binary.LittleEndian.PutUint32(wav[24:], 16000)
			binary.LittleEndian.PutUint32(wav[28:], 32000)
			binary.LittleEndian.PutUint16(wav[32:], 2)
			binary.LittleEndian.PutUint16(wav[34:], 16)
			copy(wav[36:], "data")
			binary.LittleEndian.PutUint32(wav[40:], 2*n)
			status, b := call("POST", "/v1/jobs?profile=asr-pt&name=toy.wav", bytes.NewReader(wav))
			if status != 201 {
				t.Fatal(status, string(b))
			}
			var j struct {
				ID string `json:"id"`
			}
			if json.Unmarshal(b, &j) != nil || j.ID == "" {
				t.Fatal(string(b))
			}
			if mode == "sync" {
				status, b = call("POST", "/v1/jobs/"+j.ID+"/run", nil)
				if status != 200 {
					t.Fatal(status, string(b))
				}
				status, b = call("GET", "/v1/jobs/"+j.ID+"/artifacts/vtt", nil)
				if status != 200 || string(b) != "WEBVTT\n\n" {
					t.Fatal(status, string(b))
				}
				return
			}
			status, b = call("POST", "/v1/jobs/"+j.ID+"/enqueue", nil)
			if status != 202 {
				t.Fatal(status, string(b))
			}
			wanted := speechjob.QueuePending
			if worker {
				wanted = speechjob.QueueSucceeded
			}
			deadline := time.Now().Add(4 * time.Second)
			for {
				status, b = call("GET", "/v1/queue", nil)
				var state struct {
					Entries []speechjob.QueueEntry `json:"entries"`
				}
				if status != 200 || json.Unmarshal(b, &state) != nil || len(state.Entries) != 1 {
					t.Fatal(status, string(b))
				}
				if state.Entries[0].Status == wanted {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("queue did not finish", string(b))
				}
				time.Sleep(time.Millisecond)
			}
			if worker {
				status, b = call("GET", "/v1/jobs/"+j.ID+"/artifacts/vtt", nil)
				if status != 200 || string(b) != "WEBVTT\n\n" {
					t.Fatal(status, string(b))
				}
			} else {
				status, b = call("GET", "/v1/jobs/"+j.ID, nil)
				var job struct {
					Attempts int              `json:"attempts"`
					Status   speechjob.Status `json:"status"`
				}
				if status != 200 || json.Unmarshal(b, &job) != nil || job.Attempts != 0 || job.Status != speechjob.Queued {
					t.Fatal(status, string(b))
				}
			}
		})
	}
}

func noPendingVulkan(context.Context, time.Duration) error { return nil }

type fakeVulkanEncoderCloser struct{ calls int }

func (c *fakeVulkanEncoderCloser) Close() error {
	c.calls++
	if c.calls == 1 {
		return vk.ErrVulkanInFlight
	}
	return nil
}

func TestCloseVulkanEncoderDrainsBeforeRetry(t *testing.T) {
	for _, firstDrain := range []error{nil, context.DeadlineExceeded} {
		t.Run(fmt.Sprint(firstDrain), func(t *testing.T) {
			closer := &fakeVulkanEncoderCloser{}
			drains := 0
			closeVulkanEncoder(closer, 5*time.Millisecond, func(ctx context.Context, poll time.Duration) error {
				drains++
				if ctx == nil || poll != 5*time.Millisecond {
					t.Fatal("invalid drain call", ctx, poll)
				}
				if drains == 1 {
					return firstDrain
				}
				return nil
			})
			if closer.calls != 2 || drains != 1 {
				t.Fatal("cleanup did not drain and retry", closer.calls, drains)
			}
		})
	}
}

func TestVulkanCleanupFatalStateQuarantines(t *testing.T) {
	for _, kind := range []string{"close-panic", "drain-panic", "drain-error", "drain-device", "drain-uncertain"} {
		t.Run(kind, func(t *testing.T) {
			closeCalls, drainCalls, quarantines := 0, 0, 0
			closeResource := func() error {
				closeCalls++
				if kind == "close-panic" {
					panic("fixture")
				}
				return vk.ErrVulkanInFlight
			}
			drain := func(context.Context, time.Duration) error {
				drainCalls++
				switch kind {
				case "drain-panic":
					panic("fixture")
				case "drain-error":
					return io.ErrClosedPipe
				case "drain-device":
					return vk.ErrVulkanDeviceLost
				default:
					return vk.ErrVulkanUncertain
				}
			}
			closeVulkanResource(closeResource, time.Millisecond, drain, func() { quarantines++ })
			if quarantines != 1 || closeCalls != 1 || drainCalls != map[bool]int{true: 0, false: 1}[kind == "close-panic"] {
				t.Fatal(kind, closeCalls, drainCalls, quarantines)
			}
		})
	}
}

func TestVulkanCleanupJoinedTimeoutRetries(t *testing.T) {
	closeCalls, drainCalls, quarantines := 0, 0, 0
	closeVulkanResource(func() error {
		closeCalls++
		if closeCalls == 1 {
			return vk.ErrVulkanInFlight
		}
		return nil
	}, time.Millisecond, func(context.Context, time.Duration) error {
		drainCalls++
		return errors.Join(vk.ErrVulkanInFlight, context.DeadlineExceeded)
	}, func() { quarantines++ })
	if closeCalls != 2 || drainCalls != 1 || quarantines != 0 {
		t.Fatal(closeCalls, drainCalls, quarantines)
	}
}

type fakeProfileOwner struct {
	stage    speechjob.Stage
	closed   int
	failOnce bool
}

func (o *fakeProfileOwner) Stage() speechjob.Stage { return o.stage }
func (o *fakeProfileOwner) Close(context.Context) error {
	o.closed++
	if o.failOnce && o.closed == 1 {
		return io.ErrClosedPipe
	}
	return nil
}
func vulkanToyConfig(t *testing.T) ServerConfig {
	t.Helper()
	c := toyAssets(t)
	c.Resources = &ResourceSettings{CPUSlots: 2, MemoryBytes: 64 << 20, MaxWaiting: 4, LoadBytes: 32 << 20, ResidentBytes: 16 << 20, WorkBytes: 16 << 20}
	c.Profile.Vulkan = &VulkanSettings{Enable: true, AllowExperimental: true, DeviceContains: "fixture-device", BackendSHA256: hashBytes([]byte("fixture-backend")), DrainMilliseconds: 5}
	return c
}
func TestVulkanProfileOwnedConstructionAndIdentity(t *testing.T) {
	c := vulkanToyConfig(t)
	owner := &fakeProfileOwner{stage: speechjob.Stage{Name: "asr-windows", Version: hashBytes([]byte("resident-stage")), Run: func(context.Context, *speechjob.Input, io.Writer) error { return nil }}}
	initCalls, encoderCalls, stageCalls, maxLength := 0, 0, 0, 0
	runtime := vulkanProfileRuntime{init: func() bool { initCalls++; return true }, deviceName: func() string { return "fixture-device-1" }, newEncoder: func(_ context.Context, e *whisper.Encoder, frames int) (*whisper.VulkanEncoder, error) {
		encoderCalls++
		if e == nil || frames < 1 {
			t.Fatal("host encoder missing before copy")
		}
		return &whisper.VulkanEncoder{}, nil
	}, newStage: func(m *whisper.Whisper, _ *whisper.Tokenizer, _ *whisper.VulkanEncoder, cfg speechjob.VulkanWhisperStageConfig) (stageOwner, error) {
		stageCalls++
		maxLength = m.Config.MaxLength
		if m.Encoder != nil {
			t.Fatal("host encoder retained after resident copy")
		}
		if !cfg.AllowExperimental || cfg.BackendSHA256 != c.Profile.Vulkan.BackendSHA256 || cfg.DrainPoll != 5*time.Millisecond {
			t.Fatal(cfg)
		}
		return owner, nil
	}, drain: noPendingVulkan}
	built, e := buildProfileOwned(context.Background(), c, true, runtime)
	if e != nil {
		t.Fatal(e)
	}
	if initCalls != 1 || encoderCalls != 1 || stageCalls != 1 || len(built.owners) != 1 || built.owners[0] != owner || len(built.Profiles) != 1 {
		t.Fatal(initCalls, encoderCalls, stageCalls, built)
	}
	stages := built.Profiles[0].Stages
	if len(stages) != 4 || stages[1].Version != owner.stage.Version {
		t.Fatal(stages)
	}
	expected, e := speechjob.NewTranscriptStage(speechjob.TranscriptStageConfig{ASRVersion: owner.stage.Version, Language: c.Profile.Language, WindowSamples: 160 * int64(maxLength), OverlapSamples: c.Profile.OverlapSamples})
	if e != nil || stages[2].Version != expected.Version {
		t.Fatal("transcript uses CPU ASR identity", e)
	}
	if e = built.Close(context.Background()); e != nil || owner.closed != 1 {
		t.Fatal(e, owner.closed)
	}
	// Legacy helper cannot discard a Vulkan owner.
	owner = &fakeProfileOwner{stage: owner.stage}
	runtime.newStage = func(*whisper.Whisper, *whisper.Tokenizer, *whisper.VulkanEncoder, speechjob.VulkanWhisperStageConfig) (stageOwner, error) {
		return owner, nil
	}
	if _, e = buildProfile(context.Background(), c, true); e == nil {
		t.Fatal("default helper unexpectedly bypassed owned builder")
	}
}
func TestVulkanProfileRejectsRuntimeAndClosesEncoder(t *testing.T) {
	for _, kind := range []string{"init", "device", "encoder", "stage", "stage-nil", "stage-partial", "drain"} {
		t.Run(kind, func(t *testing.T) {
			c := vulkanToyConfig(t)
			drain := noPendingVulkan
			partialOwner := &fakeProfileOwner{stage: speechjob.Stage{Name: "asr-windows", Version: hashBytes([]byte("partial"))}}
			if kind == "drain" {
				drain = nil
			}
			r := vulkanProfileRuntime{init: func() bool { return kind != "init" }, deviceName: func() string {
				if kind == "device" {
					return "other"
				}
				return "fixture-device"
			}, newEncoder: func(context.Context, *whisper.Encoder, int) (*whisper.VulkanEncoder, error) {
				if kind == "encoder" {
					return &whisper.VulkanEncoder{}, io.ErrClosedPipe
				}
				return &whisper.VulkanEncoder{}, nil
			}, newStage: func(*whisper.Whisper, *whisper.Tokenizer, *whisper.VulkanEncoder, speechjob.VulkanWhisperStageConfig) (stageOwner, error) {
				switch kind {
				case "stage":
					return nil, io.ErrClosedPipe
				case "stage-nil":
					return nil, nil
				case "stage-partial":
					return partialOwner, io.ErrClosedPipe
				}
				return &fakeProfileOwner{stage: speechjob.Stage{Name: "asr-windows", Version: hashBytes([]byte("x")), Run: func(context.Context, *speechjob.Input, io.Writer) error { return nil }}}, nil
			}, drain: drain}
			if _, e := buildProfileOwned(context.Background(), c, true, r); e == nil {
				t.Fatal(kind)
			}
			if kind == "stage-partial" && partialOwner.closed != 1 {
				t.Fatal("partial stage owner not closed", partialOwner.closed)
			}
		})
	}
}
