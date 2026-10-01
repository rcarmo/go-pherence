package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
	vadassets "github.com/rcarmo/go-pherence/loader/silero"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	vad "github.com/rcarmo/go-pherence/model/silero"
)

type whisperGoalOptions struct {
	backend    string
	repeats    int
	vad, words bool
}

func parseWhisperGoalOptions(backend, repeats, vadFlag, wordsFlag string) (whisperGoalOptions, error) {
	opts := whisperGoalOptions{backend: backend}
	if backend != "cpu" && backend != "vulkan-f32" && backend != "vulkan-f32-tile64" && backend != "vulkan-f32-tile64-key32" && backend != "vulkan-f32-tile64-key32-scoreilp" && backend != "vulkan-original-q5-mlp" && backend != "vulkan-original-q5-source" && backend != "vulkan-q8-mlp" && backend != "vulkan-q8-kv-mlp" {
		return opts, fmt.Errorf("select explicit CPU/F32/selective-Q8 backend")
	}
	n, err := strconv.Atoi(repeats)
	if err != nil || n < 5 || n > 10 {
		return opts, fmt.Errorf("benchmark requires5..10 repeats")
	}
	opts.repeats = n
	for _, item := range []struct {
		text        string
		destination *bool
	}{{vadFlag, &opts.vad}, {wordsFlag, &opts.words}} {
		switch item.text {
		case "0":
			*item.destination = false
		case "1":
			*item.destination = true
		default:
			return opts, fmt.Errorf("VAD and word timing require explicit0/1 settings")
		}
	}
	return opts, nil
}
func TestWhisperGoalOptionsAdmission(t *testing.T) {
	for _, backend := range []string{"cpu", "vulkan-f32", "vulkan-f32-tile64", "vulkan-f32-tile64-key32", "vulkan-f32-tile64-key32-scoreilp", "vulkan-original-q5-mlp", "vulkan-original-q5-source", "vulkan-q8-mlp", "vulkan-q8-kv-mlp"} {
		opts, err := parseWhisperGoalOptions(backend, "5", "1", "0")
		if err != nil || opts.repeats != 5 || !opts.vad || opts.words {
			t.Fatal(opts, err)
		}
	}
	for _, args := range [][4]string{{"vulkan", "5", "1", "0"}, {"cpu", "1", "1", "0"}, {"cpu", "11", "1", "0"}, {"cpu", "five", "1", "0"}, {"cpu", "5", "true", "0"}, {"cpu", "5", "0", ""}} {
		if _, err := parseWhisperGoalOptions(args[0], args[1], args[2], args[3]); err == nil {
			t.Fatal("ambiguous benchmark settings")
		}
	}
}

type whisperGoalSample struct {
	Index                                                                          int     `json:"index"`
	RequestSeconds                                                                 float64 `json:"request_seconds"`
	AllocatedBytes, Allocations                                                    uint64
	DecoderSelfSeconds, DecoderCrossSeconds, DecoderMLPSeconds, DecoderHeadSeconds float64
	Windows                                                                        []WindowTranscript    `json:"windows,omitempty"`
	VADWindows                                                                     []VADWindowTranscript `json:"vad_windows,omitempty"`
}

// Explicit release-model benchmark. No original engine is invoked here and no
// local cached model is loaded by ordinary go test. Run one arm per isolated
// process, alternate original/Go order externally, and preserve every raw sample.
// Load/preparation are charged separately; returned native state is closed before
// writing evidence. Current Go precision and timestamp/VAD choices stay explicit.
func TestWhisperPerformanceGoalArm(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_WHISPER_GOAL") != "1" {
		t.Skip("explicit matched native Whisper performance window required")
	}
	opts, err := parseWhisperGoalOptions(os.Getenv("GO_PHERENCE_WHISPER_BENCH_BACKEND"), os.Getenv("GO_PHERENCE_WHISPER_BENCH_REPEATS"), os.Getenv("GO_PHERENCE_WHISPER_BENCH_VAD"), os.Getenv("GO_PHERENCE_WHISPER_BENCH_WORDS"))
	if err != nil {
		t.Fatal(err)
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 30*time.Minute {
		t.Fatal("bounded test timeout<=30min required")
	}
	input, pin := os.Getenv("GO_PHERENCE_WHISPER_BENCH_INPUT"), os.Getenv("GO_PHERENCE_WHISPER_BENCH_INPUT_SHA256")
	if input == "" || len(pin) != 64 || os.Getenv("GO_PHERENCE_WHISPER_BENCH_REPORT") == "" {
		t.Fatal("require pinned input and explicit report path")
	}
	if runtime.GOMAXPROCS(0) != 4 || (os.Getenv("WHISPER_THREADS") != "" && os.Getenv("WHISPER_THREADS") != "4") {
		t.Fatal("matched baseline requires GOMAXPROCS=4 and four Whisper threads")
	}
	info, err := os.Stat(input)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 16<<20 {
		t.Fatal("benchmark fixture must be a bounded regular file", err)
	}
	pinnedSpeechFile(t, input, pin)
	language := os.Getenv("GO_PHERENCE_WHISPER_BENCH_LANGUAGE")
	if language == "" {
		t.Fatal("require explicit benchmark language")
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline.Add(-time.Second))
	defer cancel()
	var loadBefore, loadAfter, prepareAfter runtime.MemStats
	runtime.ReadMemStats(&loadBefore)
	started := time.Now()
	var model *Whisper
	var tok *Tokenizer
	var policy *CheckedGenerationConfig
	var packedFile *legacy.File
	modelPin := "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"
	precision := "F16 checkpoint widened toF32; selectiveQ8 explicitbackend only"
	if os.Getenv("GO_PHERENCE_WHISPER_BENCH_LEGACY_VALUES") == "1" {
		modelPin = os.Getenv("GO_PHERENCE_WHISPER_GGML_SHA256")
		if modelPin != "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2" && modelPin != "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69" {
			t.Fatal("retained original model pin required")
		}
		if opts.backend != "cpu" && opts.backend != "vulkan-f32-tile64-key32" && opts.backend != "vulkan-f32-tile64-key32-scoreilp" && opts.backend != "vulkan-original-q5-mlp" && opts.backend != "vulkan-original-q5-source" {
			t.Fatal("legacy-value diagnostic must preserve stored values without extra quantisation")
		}
		if opts.backend == "vulkan-original-q5-source" {
			model, tok, policy, packedFile = pinnedLegacyWhisperModelMode(t, ctx, true)
		} else if opts.backend == "vulkan-original-q5-mlp" {
			model, tok, policy, packedFile = pinnedLegacyWhisperModelOpen(t, ctx)
		} else {
			model, tok, policy = pinnedLegacyWhisperModel(t, ctx)
		}
		precision = "Explicit legacy GGML stored values widened to F32; original storage identified by ModelPin; no packed inference"
	} else {
		model, tok, policy = pinnedTurboSpeechModel(t, ctx)
	}
	if (opts.backend == "vulkan-original-q5-mlp" || opts.backend == "vulkan-original-q5-source") && os.Getenv("GO_PHERENCE_WHISPER_BENCH_LEGACY_VALUES") != "1" {
		t.Fatal("packed mode requires original stored-value model")
	}
	loadSeconds := time.Since(started).Seconds()
	runtime.ReadMemStats(&loadAfter)
	reader, err := media.OpenCanonicalPCM(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	total := int64(reader.Timeline().Samples)
	if total < 160 || total > 60*16000 {
		t.Fatal("first benchmark arms require10ms..60s canonical16k fixture")
	}
	var vadModel *vad.Model
	var encoder *VulkanEncoder
	prepareStarted := time.Now()
	if opts.vad {
		file, err := vadassets.LoadPath(ctx, os.Getenv("GO_PHERENCE_SILERO_MODEL"), vadassets.RevisionSHA256)
		if err != nil {
			t.Fatal(err)
		}
		vadModel, err = vad.New(file)
		if err != nil {
			t.Fatal(err)
		}
	}
	var memoryBefore vk.VulkanMemoryUsage
	if opts.backend != "cpu" {
		device := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
		if device == "" || !vk.VulkanInit() || !strings.Contains(strings.ToLower(vk.VulkanDeviceName()), strings.ToLower(device)) || strings.Contains(strings.ToLower(vk.VulkanDeviceName()), "llvmpipe") || strings.Contains(strings.ToLower(vk.VulkanDeviceName()), "lavapipe") {
			t.Fatal("authorised physical Vulkan device required")
		}
		memoryBefore = vk.VulkanMemoryStats()
		if memoryBefore.Bytes != 0 || memoryBefore.Allocations != 0 {
			t.Fatal("isolated native process required")
		}
		if err := vk.VulkanSetMemoryBudget(vk.VulkanMemoryBudget{MaxBytes: 4 << 30, MaxAllocations: 40}); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := vk.VulkanSetMemoryBudget(memoryBefore.Budget); err != nil {
				t.Error(err)
			}
		}()
		switch opts.backend {
		case "vulkan-f32":
			encoder, err = NewVulkanEncoder(ctx, model.Encoder, model.Config.MaxLength)
		case "vulkan-f32-tile64":
			encoder, err = NewVulkanEncoderRegTile64(ctx, model.Encoder, model.Config.MaxLength)
		case "vulkan-f32-tile64-key32":
			encoder, err = NewVulkanEncoderTile64Key32(ctx, model.Encoder, model.Config.MaxLength)
		case "vulkan-f32-tile64-key32-scoreilp":
			encoder, err = NewVulkanEncoderTile64Key32ScoreILP(ctx, model.Encoder, model.Config.MaxLength)
		case "vulkan-original-q5-mlp":
			precision = "Original Q5_0 FC1/FC2 kept packed on Vulkan; other original stored values widened to F32; activations/attention/decoder F32"
			encoder, err = NewVulkanEncoderOriginalQ5MLP(ctx, model.Encoder, model.Config.MaxLength, packedFile)
			if closeErr := packedFile.Close(); err == nil {
				err = closeErr
			}
		case "vulkan-original-q5-source":
			precision = "Original Q5_0 encoder FC1/FC2 never CPU-widened; native packed FFN plus F32 other weights/attention/decoder"
			encoder, err = newVulkanEncoderPackedOnlyMode(ctx, model.Encoder, model.Config.MaxLength, vk.NewVkF32Plan, vulkanLinearOriginalQ5MLP, packedFile, true)
			if closeErr := packedFile.Close(); err == nil {
				err = closeErr
			}
			if err == nil {
				model.Encoder = nil
			}
		case "vulkan-q8-mlp":
			encoder, err = NewVulkanEncoderQ8MLPWeight(ctx, model.Encoder, model.Config.MaxLength)
		case "vulkan-q8-kv-mlp":
			encoder, err = NewVulkanEncoderQ8KVMLPWeight(ctx, model.Encoder, model.Config.MaxLength)
		}
		if encoder != nil {
			defer func() {
				if err := encoder.Close(); err != nil {
					t.Error(err)
				}
			}()
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	prepareSeconds := time.Since(prepareStarted).Seconds()
	runtime.ReadMemStats(&prepareAfter)
	decodeOptions := PCMTranscribeOptions{Language: language, Generation: policy, WordTimestamps: opts.words, VulkanEncoder: encoder}
	var samples []whisperGoalSample
	for repeat := 0; repeat < opts.repeats; repeat++ {
		if err := ctx.Err(); err != nil {
			t.Fatal(err)
		}
		// Existing timers are process-local and checked Whisper requests serialize.
		// Timing counters include the alignment decoder when word timing is enabled.
		decSelfNs, decCrossNs, decMlpNs, decLmNs = 0, 0, 0, 0
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		sample := whisperGoalSample{Index: repeat}
		requestStarted := time.Now()
		if opts.vad {
			err = model.TranscribePCMWindowsWithVAD(ctx, reader, total, tok, decodeOptions, PCMVADOptions{Model: vadModel, Segmentation: vad.DefaultSegmentOptions(), AllowExperimental: true, PreserveWindowGaps: os.Getenv("GO_PHERENCE_WHISPER_BENCH_VAD_KEEP_GAPS") == "1"}, func(w VADWindowTranscript) error { sample.VADWindows = append(sample.VADWindows, w); return nil })
		} else {
			err = model.TranscribePCMWindows(ctx, reader, total, tok, decodeOptions, func(w WindowTranscript) error { sample.Windows = append(sample.Windows, w); return nil })
		}
		sample.RequestSeconds = time.Since(requestStarted).Seconds()
		runtime.ReadMemStats(&after)
		if err != nil {
			t.Fatal("request", repeat, err)
		}
		sample.AllocatedBytes = after.TotalAlloc - before.TotalAlloc
		sample.Allocations = after.Mallocs - before.Mallocs
		sample.DecoderSelfSeconds = float64(decSelfNs) / 1e9
		sample.DecoderCrossSeconds = float64(decCrossNs) / 1e9
		sample.DecoderMLPSeconds = float64(decMlpNs) / 1e9
		sample.DecoderHeadSeconds = float64(decLmNs) / 1e9
		samples = append(samples, sample)
		t.Logf("WHISPER_GOAL backend=%s VAD=%t words=%t repeat=%d seconds=%.6f bytes=%d allocations=%d", opts.backend, opts.vad, opts.words, repeat, sample.RequestSeconds, sample.AllocatedBytes, sample.Allocations)
	}
	var encoderStats VulkanEncoderStats
	if encoder != nil {
		encoderStats = encoder.Stats()
	}
	cleanupStarted := time.Now()
	if encoder != nil {
		if err := encoder.Close(); err != nil {
			t.Fatal(err)
		}
		after := vk.VulkanMemoryStats()
		if after.Bytes != memoryBefore.Bytes || after.Allocations != memoryBefore.Allocations || after.InFlight || after.DeviceLost {
			t.Fatal("native resource leak", after)
		}
	}
	closeErr := reader.Close()
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	cleanupSeconds := time.Since(cleanupStarted).Seconds()
	report := struct {
		Schema                                                               int `json:"schema"`
		EncoderStats                                                         VulkanEncoderStats
		ModelLoadAllocatedBytes, PreparationAllocatedBytes                   uint64
		Backend, Language                                                    string
		VAD, WordTimestamps, VADPreserveWindowGaps                           bool
		ModelPin, InputPin                                                   string
		ModelLoadSeconds, PreparationSeconds, CleanupSeconds, FullArmSeconds float64
		InputSamples                                                         int64
		CPUThreads                                                           int
		Precision                                                            string
		GoMemoryLimit                                                        string
		Samples                                                              []whisperGoalSample
	}{1, encoderStats, loadAfter.TotalAlloc - loadBefore.TotalAlloc, prepareAfter.TotalAlloc - loadAfter.TotalAlloc, opts.backend, language, opts.vad, opts.words, os.Getenv("GO_PHERENCE_WHISPER_BENCH_VAD_KEEP_GAPS") == "1", modelPin, pin, loadSeconds, prepareSeconds, cleanupSeconds, time.Since(started).Seconds(), total, runtime.GOMAXPROCS(0), precision, os.Getenv("GOMEMLIMIT"), samples}
	if err := ctx.Err(); err != nil {
		t.Fatal(err)
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	output, err := os.OpenFile(os.Getenv("GO_PHERENCE_WHISPER_BENCH_REPORT"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal("report path must be fresh; preserved baseline evidence is never overwritten", err)
	}
	_, writeErr := output.Write(append(data, '\n'))
	closeReportErr := output.Close()
	if writeErr != nil || closeReportErr != nil {
		t.Fatal("benchmark report write", writeErr, closeReportErr)
	}
}
