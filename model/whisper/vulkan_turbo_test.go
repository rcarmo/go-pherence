package whisper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func pinnedTurboSpeechModel(t *testing.T, ctx context.Context) (*Whisper, *Tokenizer, *CheckedGenerationConfig) {
	t.Helper()
	dir := os.Getenv("GO_PHERENCE_WHISPER_TURBO_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_WHISPER_TURBO_DIR")
	}
	f, err := os.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	_, err = io.Copy(hash, f)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	if hex.EncodeToString(hash.Sum(nil)) != "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1" {
		t.Fatal("unpinnedTurbo")
	}
	config := pinnedSpeechFile(t, filepath.Join(dir, "config.json"), "c5b526b3e3cd64cd8940dabb45e8ba726629e22d8ed389c29b552f9140daf04a")
	generation := pinnedSpeechFile(t, filepath.Join(dir, "generation_config.json"), "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c")
	tokenPath := filepath.Join(dir, "tokenizer.json")
	pinnedSpeechFile(t, tokenPath, "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd")
	tok, err := LoadTokenizer(tokenPath)
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	info := source.TensorInfos()
	if len(info) != 587 {
		source.Close()
		t.Fatal("unexpected Turbo tensor count", len(info))
	}
	for name, i := range info {
		if i.DType != "F16" {
			source.Close()
			t.Fatal("unexpected dtype", name, i.DType)
		}
	}
	start := time.Now()
	model, policy, err := LoadConfiguredModelSourceChecked(ctx, source, config, generation, tok)
	closeErr = source.Close()
	if err != nil || closeErr != nil {
		t.Fatal("checked Turbo load", err, closeErr)
	}
	t.Logf("TURBO_LOAD ns=%d tensors=587 checkpoint_f16=true inference_f32=true", time.Since(start).Nanoseconds())
	return model, tok, policy
}

// Explicitly heavier test: full 32-layer trained F32 Turbo encoder. First run
// only the bounded short reference subtest; speech requires a second opt-in.
// Per-fixture selection allows qualification to stop without replaying success.
func TestVulkanTurbo(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO") != "1" {
		t.Skip("explicit full Turbo compute opt-in required")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 5*time.Minute {
		t.Fatal("test timeout<=5m required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 270*time.Second)
	defer cancel()
	model, tok, policy := pinnedTurboSpeechModel(t, ctx)
	if !vk.VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	device := vk.VulkanDeviceName()
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if want == "" || !strings.Contains(device, want) || strings.Contains(strings.ToLower(device), "llvmpipe") {
		t.Fatal("unexpected device", device)
	}
	before := vk.VulkanMemoryStats()
	// Package allocation caps are not total process/device RAM limits. Host
	// snapshots remain necessary; this bounds only explicitly owned native memory.
	if before.Allocations != 0 || before.Bytes != 0 {
		t.Fatal("run Turbo in isolated test process", before)
	}
	if err := vk.VulkanSetMemoryBudget(vk.VulkanMemoryBudget{MaxBytes: 4 << 30, MaxAllocations: 40}); err != nil {
		t.Fatal(err)
	}
	// Separate cleanup callbacks: t.Fatal during leak checking cannot prevent
	// the earlier registered budget restoration callback from running.
	t.Cleanup(func() {
		if err := vk.VulkanSetMemoryBudget(before.Budget); err != nil {
			t.Error(err)
		}
	})
	t.Cleanup(func() { nativeEncoderMemory(t, before) })
	if !t.Run("short-reference", func(t *testing.T) {
		const frames = 4
		mel := make([]float32, model.Config.NumMelBins*frames)
		for i := range mel {
			mel[i] = float32(i%17-8) / 32
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
			out := make([]float32, len(refs[i]))
			if err := gpu.s.output.Download(ctx, out); err != nil {
				t.Fatal(err)
			}
			trainedCompare(t, fmt.Sprintf("turbo-boundary%d", i), out, refs[i], 2e-3, 5e-4)
		}
		result, _ := json.Marshal(map[string]any{"device": device, "stats": gpu.Stats(), "memory": vk.VulkanMemoryStats(), "full_trained_width_depth": true, "frames": frames})
		t.Log("TURBO_SHORT " + string(result))
	}) {
		return
	}
	nativeEncoderMemory(t, before)
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO_SPEECH") != "1" {
		return
	}
	if !t.Run("public-speech", func(t *testing.T) {
		root := os.Getenv("GO_PHERENCE_MINDS_FIXTURE_DIR")
		jfk := os.Getenv("GO_PHERENCE_WHISPER_JFK_PATH")
		fixtures, err := turboPublicFixtures(os.Getenv("GO_PHERENCE_TURBO_FIXTURE"), root, jfk)
		if err != nil {
			t.Fatal(err)
		}
		// Validate selected files before allocating the full resident encoder.
		for _, f := range fixtures {
			pinnedSpeechFile(t, f.File, f.SHA256)
		}
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
		start := time.Now()
		gpu, err := NewVulkanEncoder(ctx, model.Encoder, 3000)
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
		// Decoder retained on host, encoder weights owned by Vulkan. Release the
		// host encoder before inference to reduce steady-state resident memory.
		model.Encoder = nil
		runtime.GC()
		meta, _ := json.Marshal(map[string]any{"construct_ns": time.Since(start).Nanoseconds(), "device": device, "stats": gpu.Stats(), "native_memory": vk.VulkanMemoryStats()})
		t.Log("TURBO_RESIDENT " + string(meta))
		for i, f := range fixtures {
			decoded, err := adapter.DecodeToFile(ctx, f.File, filepath.Join(t.TempDir(), fmt.Sprintf("pcm%d.wav", i)))
			if err != nil {
				t.Fatal(err)
			}
			if int64(decoded.Timeline.Samples) != f.Samples || decoded.Timeline.SampleRate != 16000 {
				t.Fatal("PCM extent")
			}
			reader, err := media.OpenCanonicalPCM(ctx, decoded.Path)
			if err != nil {
				t.Fatal(err)
			}
			out := speechCorpusResult{Fixture: f.Name, Language: f.Language, Path: "vulkan-turbo-f32", Samples: f.Samples, Reference: f.Reference}
			start := time.Now()
			err = model.TranscribePCMWindows(ctx, reader, f.Samples, tok, PCMTranscribeOptions{Language: f.Language, Generation: policy, MaxNewTokens: 96, VulkanEncoder: gpu}, func(w WindowTranscript) error { out.Windows = append(out.Windows, w); return nil })
			out.InferenceNS = time.Since(start).Nanoseconds()
			closeErr := reader.Close()
			if err != nil {
				out.Error = err.Error()
			}
			for _, w := range out.Windows {
				for _, s := range w.Segments {
					out.Text += " " + s.Text
				}
			}
			out.Text = strings.TrimSpace(out.Text)
			out.Edits, out.Words = speechFixtureWER(f.Reference, out.Text)
			wer := float64(out.Edits) / float64(out.Words)
			out.WER = &wer
			out.EmptyOutput = out.Text == ""
			raw, _ := json.Marshal(out)
			t.Log("TURBO_RESULT " + string(raw))
			for _, w := range out.Windows {
				raw, _ := json.Marshal(map[string]any{"fixture": f.Name, "window": w})
				t.Log("TURBO_WINDOW " + string(raw))
			}
			if err != nil || closeErr != nil || len(out.Windows) != 1 || out.EmptyOutput {
				t.Fatal("incomplete Turbo inference", err, closeErr)
			}
			// WER is reported independently: completion is not quality acceptance.
			if i == 0 && os.Getenv("GO_PHERENCE_TEST_TURBO_STAGES") == "1" {
				turboStageDiagnostic(t, ctx, model, tok, policy, gpu, f, decoded.Path, out.Windows)
			}
		}

	}) {
		return
	}
}

func turboPublicFixtures(selected, root, jfk string) ([]publicSpeechFixture, error) {
	all := append([]publicSpeechFixture(nil), mindsSpeechFixtures...)
	all = append(all, publicSpeechFixture{Name: "jfk", Language: "en", File: jfk, SHA256: "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e", Reference: "And so my fellow Americans ask not what your country can do for you ask what you can do for your country", Samples: 176000})
	var result []publicSpeechFixture
	for _, f := range all {
		if selected != "" && selected != f.Name {
			continue
		}
		if f.Name == "jfk" {
			if jfk == "" {
				return nil, fmt.Errorf("JFK path required")
			}
		} else {
			if root == "" {
				return nil, fmt.Errorf("MINDS directory required")
			}
			f.File = filepath.Join(root, f.File)
		}
		result = append(result, f)
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("unknown Turbo fixture %q", selected)
	}
	return result, nil
}
func TestTurboFixtureSelection(t *testing.T) {
	for _, c := range []struct {
		selected, root, jfk string
		count               int
	}{{"jfk", "", "/public/jfk.wav", 1}, {"minds-pt-0", "/public", "", 1}, {"", "/public", "/public/jfk.wav", 4}, {"jfk", "/public", "", 0}, {"minds-pt-0", "", "/public/jfk.wav", 0}, {"unknown", "/public", "/public/jfk.wav", 0}} {
		f, err := turboPublicFixtures(c.selected, c.root, c.jfk)
		if c.count == 0 {
			if err == nil {
				t.Fatal("acceptedbadselector", c)
			}
		} else if err != nil || len(f) != c.count {
			t.Fatal(c, f, err)
		}
	}
}

// Separate equal-output diagnostic, excluded from end-to-end fixture timing.
func turboStageDiagnostic(t *testing.T, ctx context.Context, model *Whisper, tok *Tokenizer, policy *CheckedGenerationConfig, gpu *VulkanEncoder, f publicSpeechFixture, path string, windows []WindowTranscript) {
	t.Helper()
	r, err := media.OpenCanonicalPCM(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	samples := make([]float32, 480000)
	n, readErr := r.ReadSamplesAt(ctx, samples, 0)
	closeErr := r.Close()
	if int64(n) != f.Samples || readErr != io.EOF || closeErr != nil {
		t.Fatal("stagePCM", n, readErr, closeErr)
	}
	start := time.Now()
	mel, frames, err := MelFlatFromSamplesCheckedContext(ctx, samples, model.Config)
	if err != nil {
		t.Fatal(err)
	}
	front := time.Since(start).Nanoseconds()
	start = time.Now()
	encoded, err := gpu.Forward(ctx, mel)
	if err != nil {
		t.Fatal(err)
	}
	encoder := time.Since(start).Nanoseconds()
	start = time.Now()
	state, err := NewDecoderStateContext(ctx, model.Config, encoded, (frames+1)/2, model.Decoder)
	if err != nil {
		t.Fatal(err)
	}
	cross := time.Since(start).Nanoseconds()
	vocab, err := checkedTimestampVocabulary(model.Config, tok, f.Language)
	if err != nil {
		t.Fatal(err)
	}
	opts, suppress, begin, err := resolvePCMGeneration(model.Config, vocab, PCMTranscribeOptions{Language: f.Language, Generation: policy, MaxNewTokens: 96}, model.Decoder.SuppressTokens, model.Decoder.BeginSuppressTokens)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	start = time.Now()
	segments, err := decodeCheckedTimestamps(ctx, model.Config, tok, vocab, opts, suppress, begin, func(token int) ([]float32, error) { calls++; return model.Decoder.ForwardToken(token, state), nil })
	if err != nil {
		t.Fatal(err)
	}
	decode := time.Since(start).Nanoseconds()
	segments, err = canonicalWindowSegments(windows[0].Window, segments)
	if err != nil || !reflect.DeepEqual(segments, windows[0].Segments) {
		t.Fatal("stageoutput differs", err)
	}
	raw, _ := json.Marshal(map[string]any{"fixture": f.Name, "frontend_ns": front, "encoder_ns": encoder, "cross_kv_ns": cross, "decoder_ns": decode, "decoder_calls": calls, "matches_checked_pcm": true, "separate_diagnostic": true})
	t.Log("TURBO_STAGES " + string(raw))
}
