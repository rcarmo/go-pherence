package whisper

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
)

func TestVulkanEncoderPlanConstructorAdmission(t *testing.T) {
	if e, err := newVulkanEncoder(context.Background(), nil, 1, nil); err == nil || e != nil {
		t.Fatal("nil constructor admitted")
	}
	called := false
	_, err := newVulkanEncoder(context.Background(), nil, 1, func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error) { called = true; return nil, nil })
	if err == nil || called {
		t.Fatal("invalid source reached plan builder")
	}
}

// Diagnostic only: the public constructor still uses unmodified whole plans.
// No timing callbacks or alternate stage dispatch are exposed to inference APIs.
func TestVulkanTurboProfile(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO_PROFILE") != "1" {
		t.Skip("explicit full Turbo profiling window required")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 300*time.Second {
		t.Fatal("test timeout<=300s required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 270*time.Second)
	defer cancel()
	root := os.Getenv("GO_PHERENCE_MINDS_FIXTURE_DIR")
	if root == "" {
		t.Fatal("MINDS directory required")
	}
	fixture := mindsSpeechFixtures[0]
	path := filepath.Join(root, fixture.File)
	pinnedSpeechFile(t, path, fixture.SHA256)
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := media.NewFFmpeg(media.Config{FFmpegPath: ffmpeg, FFprobePath: ffprobe})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := adapter.DecodeToFile(ctx, path, filepath.Join(t.TempDir(), "canonical.wav"))
	if err != nil {
		t.Fatal(err)
	}
	if int64(decoded.Timeline.Samples) != fixture.Samples || decoded.Timeline.SampleRate != 16000 {
		t.Fatal("PCM extent")
	}
	reader, err := media.OpenCanonicalPCM(ctx, decoded.Path)
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, fixture.Samples)
	n, readErr := reader.ReadSamplesAt(ctx, pcm, 0)
	closeErr := reader.Close()
	if n != len(pcm) || readErr != nil || closeErr != nil {
		t.Fatal(n, readErr, closeErr)
	}
	model, _, _ := pinnedTurboSpeechModel(t, ctx)
	padded := make([]float32, 480000)
	copy(padded, pcm)
	mel, frames, err := MelFlatFromSamplesCheckedContext(ctx, padded, model.Config)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := describeVulkanEncoder(ctx, model.Encoder, frames)
	if err != nil {
		t.Fatal(err)
	}
	if !vk.VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	name := vk.VulkanDeviceName()
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if want == "" || !strings.Contains(name, want) || strings.Contains(strings.ToLower(name), "llvmpipe") {
		t.Fatal("unexpected device", name)
	}
	before := vk.VulkanMemoryStats()
	if before.Allocations != 0 || before.Bytes != 0 {
		t.Fatal("profile requires isolated process")
	}
	if err := vk.VulkanSetMemoryBudget(vk.VulkanMemoryBudget{MaxBytes: 4 << 30, MaxAllocations: 40}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := vk.VulkanSetMemoryBudget(before.Budget); err != nil {
			t.Error(err)
		}
	})
	t.Cleanup(func() { nativeEncoderMemory(t, before) })
	// Capture the very stages used to construct normal plans. Single-stage plans
	// have private descriptor sets/fences but share exactly the same kernel/tensors.
	var groups [][]vk.VkF32Stage
	factory := func(ctx context.Context, stages []vk.VkF32Stage) (*vk.VkF32Plan, error) {
		p, err := vk.NewVkF32Plan(ctx, stages)
		if err == nil {
			groups = append(groups, append([]vk.VkF32Stage(nil), stages...))
		}
		return p, err
	}
	gpu, err := newVulkanEncoder(ctx, model.Encoder, frames, factory)
	if gpu != nil {
		t.Cleanup(func() {
			if err := gpu.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	// Retain only the graph's symbolic operations, not source slices. Both the
	// original model and layout weight descriptions otherwise keep host weights.
	symbolic := layout.plans
	layout = nil
	model = nil
	runtime.GC()
	if len(groups) != 34 || len(symbolic) != len(groups) || gpu.Stats().Stages != 390 {
		t.Fatal("profile graph topology")
	}
	type opRecord struct {
		Plan   int       `json:"plan"`
		Index  int       `json:"index"`
		Kind   string    `json:"kind"`
		Output []int     `json:"output_shape"`
		Inputs [][]int   `json:"input_shapes"`
		Groups [3]uint32 `json:"groups"`
		FLOPs  uint64    `json:"estimated_flops"`
		run    *vk.VkF32Plan
	}
	var operations []opRecord
	for i, stages := range groups {
		if len(stages) != len(symbolic[i]) {
			t.Fatal("symbolic plan mismatch")
		}
		for j, stage := range stages {
			p, err := vk.NewVkF32Plan(ctx, []vk.VkF32Stage{stage})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := p.Close(); err != nil {
					t.Error(err)
				}
			})
			op := symbolic[i][j]
			record := opRecord{Plan: i, Index: j, Kind: op.op, Groups: stage.Groups, run: p}
			for k, tensor := range stage.Tensors {
				shape := tensor.Shape()
				if k == len(stage.Tensors)-1 {
					record.Output = shape
				} else {
					record.Inputs = append(record.Inputs, shape)
				}
			}
			if op.op == "linear" {
				m, k, n := record.Inputs[0][0], record.Inputs[0][1], record.Inputs[1][0]
				record.FLOPs = 2 * uint64(m) * uint64(k) * uint64(n)
			}
			if op.op == "conv" {
				rows, out := record.Output[0], record.Output[1]
				in := record.Inputs[1][1]
				record.FLOPs = 2 * uint64(rows) * uint64(out) * uint64(in) * 3
			}
			if op.op == "attention" {
				rows, width := record.Inputs[0][0], record.Inputs[0][1]
				keys := record.Inputs[1][0]
				record.FLOPs = 4 * uint64(rows) * uint64(keys) * uint64(width)
			} // QK+PV only, no softmax
			operations = append(operations, record)
		}
	}
	start := time.Now()
	reference, err := gpu.Forward(ctx, mel)
	if err != nil {
		t.Fatal(err)
	}
	forwardTime := time.Since(start).Nanoseconds()
	bytes := make([]byte, len(reference)*4)
	for i, v := range reference {
		binary.LittleEndian.PutUint32(bytes[4*i:], math.Float32bits(v))
	}
	hash := sha256.Sum256(bytes)
	info, _ := json.Marshal(map[string]any{"device": name, "stats": gpu.Stats(), "native_memory": vk.VulkanMemoryStats(), "reference_forward_ns": forwardTime, "output_sha256": hex.EncodeToString(hash[:]), "fixture": fixture.Name, "normal_plans": len(groups), "single_stage_plans": len(operations), "gpu_timestamps": false})
	t.Log("PROFILE_META " + string(info))
	for _, record := range operations {
		raw, _ := json.Marshal(record)
		t.Log("PROFILE_OPERATION " + string(raw))
	}
	// All paths start from the same uploaded input and overwrite scratch in order.
	// Two alternating rounds reduce order bias; timings remain attribution clues,
	// not standalone-kernel throughput or a speedup qualification.
	for round := 0; round < 2; round++ {
		modes := []string{"layer_plans", "single_stage"}
		if round == 1 {
			modes = []string{"single_stage", "layer_plans"}
		}
		for _, mode := range modes {
			if err := gpu.s.input.Upload(ctx, mel); err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			var samples []int64
			if mode == "layer_plans" {
				for _, p := range gpu.s.plans {
					begin := time.Now()
					if err := p.Run(ctx); err != nil {
						t.Fatal(err)
					}
					samples = append(samples, time.Since(begin).Nanoseconds())
				}
			} else {
				for _, op := range operations {
					begin := time.Now()
					if err := op.run.Run(ctx); err != nil {
						t.Fatal(err)
					}
					samples = append(samples, time.Since(begin).Nanoseconds())
				}
			}
			total := time.Since(start).Nanoseconds()
			out := make([]float32, len(reference))
			if err := gpu.s.output.Download(ctx, out); err != nil {
				t.Fatal(err)
			}
			equalContextFloats(t, reference, out)
			for index, ns := range samples {
				raw, _ := json.Marshal(map[string]any{"round": round, "mode": mode, "index": index, "wall_ns": ns})
				t.Log("PROFILE_SAMPLE " + string(raw))
			}
			raw, _ := json.Marshal(map[string]any{"round": round, "mode": mode, "wall_ns": total, "samples": len(samples), "bit_exact": true})
			t.Log("PROFILE_RUN " + string(raw))
		}
	}
	// The mode comparison retains no output beyond this test and doesn't change
	// production kernels, weights or execution defaults.
}
