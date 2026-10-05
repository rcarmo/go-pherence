package nemotronasr

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio"
)

func TestReleasedASRSharedProjectionParity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_ASR_SHARED") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_ASR_SHARED=1 with authorised Vulkan access")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	pin := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_SHA256")
	if len(pin) != 64 || path == "" {
		t.Fatal("require pinned model path and GO_PHERENCE_NEMOTRON_ASR_SHA256")
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	_, err = io.Copy(h, file)
	closeErr := file.Close()
	if err != nil || closeErr != nil || hex.EncodeToString(h.Sum(nil)) != pin {
		t.Fatal("model pin mismatch", err, closeErr)
	}
	device := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if device == "" || !vulkan.VulkanInit() || !strings.Contains(strings.ToLower(vulkan.VulkanDeviceName()), strings.ToLower(device)) {
		t.Fatal("authorised Vulkan device unavailable/mismatch")
	}
	model := releasedPCMGenerationModel(t)
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		t.Fatal(err, rate)
	}
	type result struct {
		Backend    string                  `json:"backend"`
		Seconds    float64                 `json:"seconds"`
		Tokens     []int                   `json:"tokens"`
		Frames     []int64                 `json:"frames"`
		Stats      ProjectionTransferStats `json:"stats"`
		sub, tower []float32
	}
	run := func(backend string) result {
		t.Helper()
		before := vulkan.VulkanMemoryStats()
		prompt := 0
		s := &PCMGenerationStream{Model: model, PromptID: &prompt}
		var p *DeviceSubsamplingProjector
		if backend != "cpu" {
			p = &DeviceSubsamplingProjector{Backend: "vulkan", SharedMemory: backend == "shared"}
			s.Projector = p
		}
		r := result{Backend: backend}
		s.onStage = func(stage string, v []float32) {
			if stage == "subsampling" {
				r.sub = append(r.sub, v...)
			} else if stage == "tower" {
				r.tower = append(r.tower, v...)
			}
		}
		started := time.Now()
		for offset := 0; offset < len(pcm); offset += 80000 {
			end := min(offset+80000, len(pcm))
			tokens, frames, err := s.AppendPCM(context.Background(), pcm[offset:end])
			if err != nil {
				t.Fatal(err)
			}
			r.Tokens = append(r.Tokens, tokens...)
			r.Frames = append(r.Frames, frames...)
		}
		tokens, frames, err := s.Finish(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		r.Tokens = append(r.Tokens, tokens...)
		r.Frames = append(r.Frames, frames...)
		if p != nil {
			r.Stats = p.TransferStats()
			if err := p.Close(); err != nil {
				t.Fatal(err)
			}
		}
		r.Seconds = time.Since(started).Seconds()
		after := vulkan.VulkanMemoryStats()
		if before.Bytes != after.Bytes || before.Allocations != after.Allocations || after.InFlight || after.DeviceLost {
			t.Fatal("native resources leaked", before, after)
		}
		if backend == "shared" {
			if r.Stats.Dispatches < 1 || r.Stats.InputCopyBytes != 0 || r.Stats.OutputCopyBytes != 0 || r.Stats.CPUReads != uint64(r.Stats.Dispatches) || r.Stats.CPUWrites != uint64(r.Stats.Dispatches) {
				t.Fatal("shared boundary copied/no native borrow", r.Stats)
			}
		}
		if backend == "copied" && (r.Stats.InputCopyBytes != uint64(r.Stats.Dispatches)*4*4352*4 || r.Stats.OutputCopyBytes != uint64(r.Stats.Dispatches)*4*1024*4) {
			t.Fatal("copied accounting", r.Stats)
		}
		return r
	}
	compare := func(got, want result) {
		t.Helper()
		if !reflect.DeepEqual(got.Tokens, want.Tokens) || !reflect.DeepEqual(got.Frames, want.Frames) {
			t.Fatalf("%s/%s decision mismatch", got.Backend, want.Backend)
		}
		for _, v := range []struct {
			name           string
			got, want      []float32
			abs, rel, mean float64
		}{{"subsampling", got.sub, want.sub, 3e-3, 4e-5, 1e-4}, {"tower", got.tower, want.tower, 3e-4, 2e-5, 2e-5}} {
			if len(v.got) != len(v.want) || len(v.got) == 0 {
				t.Fatal("empty/length")
			}
			var max, sum float64
			for i, x := range v.got {
				d := math.Abs(float64(x - v.want[i]))
				max = math.Max(max, d)
				sum += d
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || d > v.abs+v.rel*math.Abs(float64(v.want[i])) {
					t.Fatalf("%s/%s %s drift[%d] %g", got.Backend, want.Backend, v.name, i, d)
				}
			}
			mean := sum / float64(len(v.got))
			if mean > v.mean {
				t.Fatal("mean drift", mean)
			}
			t.Logf("%s/%s %s max=%g mean=%g", got.Backend, want.Backend, v.name, max, mean)
		}
	}
	var results []result
	for repeat := 0; repeat < 5; repeat++ {
		order := []string{"cpu", "copied", "shared"}
		if repeat%2 == 1 {
			order = []string{"shared", "copied", "cpu"}
		}
		round := map[string]result{}
		for _, backend := range order {
			r := run(backend)
			round[backend] = r
			results = append(results, r)
			t.Logf("repeat=%d backend=%s seconds=%.6f stats=%+v", repeat, backend, r.Seconds, r.Stats)
		}
		compare(round["shared"], round["cpu"])
		compare(round["copied"], round["cpu"])
		compare(round["shared"], round["copied"])
	}
	// Cancellation after a completed chunk must close the ASR stream and drain
	// confirmed resources; a fresh request repeats safely with new shared storage.
	p := &DeviceSubsamplingProjector{Backend: "vulkan", SharedMemory: true}
	prompt := 0
	s := &PCMGenerationStream{Model: model, PromptID: &prompt, Projector: p}
	ctx := &cancelAfterASRChunk{stream: s}
	tokens, frames, err := s.AppendPCM(ctx, pcm[:80000])
	if !errors.Is(err, context.Canceled) || tokens != nil || frames != nil || !s.closed {
		t.Fatal("cancel contract", err)
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	fresh := run("shared")
	compare(fresh, results[0])
	if output := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_SHARED_REPORT"); output != "" {
		data, err := json.MarshalIndent(struct {
			Device             string   `json:"device"`
			ModelSHA256        string   `json:"model_sha256"`
			Results            []result `json:"results"`
			Fresh              result   `json:"fresh"`
			CancellationPassed bool     `json:"cancellation_passed"`
		}{vulkan.VulkanDeviceName(), pin, results, fresh, true}, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(output, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
}
