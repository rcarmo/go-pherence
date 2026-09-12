package whisper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Pinned public Tiny checkpoint only. No download/credentials in the test.
// Disabled by default; select a physical GPU ICD and expected device explicitly.
func TestVulkanEncoderTrainedTiny(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TRAINED") != "1" {
		t.Skip("explicit trained Vulkan compute opt-in required")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 3*time.Minute {
		t.Fatal("use test timeout<=3m")
	}
	dir := os.Getenv("GO_PHERENCE_WHISPER_TINY_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_WHISPER_TINY_DIR")
	}
	path := filepath.Join(dir, "model.safetensors")
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	_, err = io.Copy(hash, f)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	wantHash := "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
	if hex.EncodeToString(hash.Sum(nil)) != wantHash {
		t.Fatal("unpinned checkpoint")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfgHash := sha256.Sum256(raw)
	if hex.EncodeToString(cfgHash[:]) != "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98" {
		t.Fatal("unpinnedconfig")
	}
	cfg, err := ParseModelConfigChecked(raw)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 110*time.Second)
	defer cancel()
	source, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadModelSourceChecked(ctx, source, cfg)
	closeErr = source.Close()
	if err != nil || closeErr != nil {
		t.Fatal("checkedtrainedload", err, closeErr)
	}
	if !vk.VulkanInit() {
		t.Fatal("native Vulkan unavailable")
	}
	device := vk.VulkanDeviceName()
	expected := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if expected == "" || !strings.Contains(device, expected) || strings.Contains(strings.ToLower(device), "llvmpipe") {
		t.Fatal("unexpecteddevice", device)
	}
	t.Logf("TRAINED_CHECKPOINT revision=169d4a4341b33bc18d8881c4b69c2e104e1cc0af sha256=%s device=%s", wantHash, device)
	before := vk.VulkanMemoryStats()
	// Full trained width/depth, shortened sequence for affordable scalar boundary
	// and decoder comparison. No reduced-width/synthetic-weight substitution.
	if !t.Run("trained-short-boundaries-decoder", func(t *testing.T) {
		const frames = 34
		mel := make([]float32, cfg.NumMelBins*frames)
		for i := range mel {
			mel[i] = float32(math.Sin(float64(i)*.023) * .5)
		}
		refs := vulkanScalarBoundaries(model.Encoder, mel, frames)
		gpu, err := NewVulkanEncoder(ctx, model.Encoder, frames)
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
		if err := gpu.s.input.Upload(ctx, mel); err != nil {
			t.Fatal(err)
		}
		for i, p := range gpu.s.plans {
			if err := p.Run(ctx); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(refs[i]))
			if err := gpu.s.output.Download(ctx, got); err != nil {
				t.Fatal(err)
			}
			trainedCompare(t, fmt.Sprintf("boundary%d", i), got, refs[i], 1e-3, 2e-4)
		}
		out, err := gpu.Forward(ctx, mel)
		if err != nil {
			t.Fatal(err)
		}
		trainedCompare(t, "short-final", out, refs[len(refs)-1], 1e-3, 2e-4)
		scalarState, err := NewDecoderStateContext(ctx, cfg, refs[len(refs)-1], frames/2, model.Decoder)
		if err != nil {
			t.Fatal(err)
		}
		gpuState, err := NewDecoderStateContext(ctx, cfg, out, frames/2, model.Decoder)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < cfg.DecoderLayers; i++ {
			trainedCompare(t, fmt.Sprintf("crossK%d", i), gpuState.CrossK[i], scalarState.CrossK[i], 2e-3, 3e-4)
			trainedCompare(t, fmt.Sprintf("crossV%d", i), gpuState.CrossV[i], scalarState.CrossV[i], 2e-3, 3e-4)
		}
		for i, token := range []int{50258, 50259, 50359} {
			cpu := append([]float32(nil), model.Decoder.ForwardToken(token, scalarState)...)
			got := model.Decoder.ForwardToken(token, gpuState)
			trainedCompare(t, fmt.Sprintf("prompt-logits%d", i), got, cpu, 1e-2, 5e-4)
			argmax := func(a []float32) int {
				idx := 0
				for i, v := range a {
					if v > a[idx] {
						idx = i
					}
				}
				return idx
			}
			if argmax(got) != argmax(cpu) {
				t.Fatal("prompttop1", i, argmax(got), argmax(cpu))
			}
			t.Logf("TRAINED_TOP1 step=%d token=%d", i, argmax(got))
		}
	}) {
		return
	}
	nativeEncoderMemory(t, before)
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_FULL_TINY") != "1" {
		return
	}
	if !t.Run("trained-full-shape-repeat", func(t *testing.T) {
		// Correctly shaped synthetic features, not speech or a quality/RTF corpus.
		mel := make([]float32, cfg.NumMelBins*cfg.MaxLength)
		for i := range mel {
			mel[i] = float32(math.Sin(float64(i)*.023) * .5)
		}
		start := time.Now()
		gpu, err := NewVulkanEncoder(ctx, model.Encoder, cfg.MaxLength)
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
		construction := time.Since(start)
		var first []float32
		durations := []int64{}
		for i := 0; i < 3; i++ {
			start = time.Now()
			out, err := gpu.Forward(ctx, mel)
			if err != nil {
				t.Fatal(err)
			}
			durations = append(durations, time.Since(start).Nanoseconds())
			if len(out) != 1500*384 {
				t.Fatal("fullshapeextent")
			}
			if i == 0 {
				first = out
			} else {
				equalContextFloats(t, first, out)
			}
		}
		h := sha256.New()
		var word [4]byte
		for _, v := range first {
			b := math.Float32bits(v)
			word = [4]byte{byte(b), byte(b >> 8), byte(b >> 16), byte(b >> 24)}
			h.Write(word[:])
		}
		data, _ := json.Marshal(map[string]any{"stats": gpu.Stats(), "construct_ns": construction.Nanoseconds(), "forward_ns": durations, "output_sha256": hex.EncodeToString(h.Sum(nil)), "values": len(first), "repeat_bit_exact": true, "scalar_full_reference": false, "speech_input": false})
		t.Log("TRAINED_FULL " + string(data))
	}) {
		return
	}
	nativeEncoderMemory(t, before)
}
func trainedCompare(t *testing.T, name string, got, want []float32, abs, rel float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("length", name)
	}
	maximum := 0.
	for i, v := range got {
		err := math.Abs(float64(v) - float64(want[i]))
		maximum = math.Max(maximum, err)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || err > abs+rel*math.Abs(float64(want[i])) {
			t.Fatalf("%s[%d] got%g want%g error%g budget%g", name, i, v, want[i], err, abs+rel*math.Abs(float64(want[i])))
		}
	}
	data, _ := json.Marshal(map[string]any{"name": name, "values": len(got), "max_abs": maximum, "abs_budget": abs, "rel_budget": rel})
	t.Log("TRAINED_METRIC " + string(data))
}
