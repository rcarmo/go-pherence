// omnivoice is a native CPU development probe, not yet an end-to-end TTS CLI.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"runtime"
	"syscall"
	"time"

	audio "github.com/rcarmo/go-264/audio"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	model "github.com/rcarmo/go-pherence/models/omnivoice"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(args []string) error {
	commandStarted := time.Now()
	flags := flag.NewFlagSet("omnivoice", flag.ContinueOnError)
	path := flags.String("model", "", "local OmniVoice model directory (required)")
	mode := flags.String("mode", "inspect", "inspect, block, stack, logits, prepare, synthesize, synthesize-long, plan-chunks, encode-reference, generate (prepared prompt), export-gguf, serve (NDJSON stdin/stdout), capabilities, or audio")
	serveDir := flags.String("output-dir", "", "serve: existing directory for private per-process chunk files")
	cacheMiB := flags.Int64("cache-mib", 0, "serve: bounded process-local phrase cache payload MiB (0..256)")
	firstFrames := flags.Int("first-frames", 0, "serve: shorter first-chunk frame limit (0 uses frames)")
	ggufPath := flags.String("weights-gguf", "", "GGUF backbone override for generate/logits/block/stack; -model still supplies codec assets and matching config")
	ggufFormat := flags.String("gguf-format", "f16", "export-gguf storage: f16, f32, q8_0 or q8_0_f16")
	directQ8 := flags.Bool("direct-q8", false, "experimental direct Q8 SIMD projections for generate; excludes resident/prepack")
	input := flags.String("input", "", "pretokenized input JSON for logits/generate")
	output := flags.String("output", "", "new synthetic WAV path for generate mode")
	shared := flags.Bool("shared-traversal", false, "experimental guided streamed CFG layer traversal; excludes resident/prepack/direct-q8")
	guidanceFlag := flags.Float64("guidance", 2, "experimental CFG scale (0 skips unconditional inference; default 2)")
	steps := flags.Int("steps", 16, "generation steps")
	residentMiB := flags.Int64("resident-mib", 0, "opt-in decoder cache budget in MiB including optional packed panels; 0 streams layers (excludes other memory)")
	prepacked := flags.Bool("prepack", false, "prepack resident decoder projections; requires budget covering raw weights plus panels")
	text := flags.String("text", "", "target speech text for prepare/synthesize")
	cachedReference := flags.String("reference-tokens", "", "cached reference codes and transcript JSON (not raw audio)")
	frames := flags.Int("frames", 75, "target frames, or maximum per chunk in long modes, 25/sec (1..250)")
	language := flags.String("language", "en", "prompt language")
	instruct := flags.String("instruct", "", "optional style instruction")
	denoise := flags.Bool("denoise", true, "request denoised voice reference")
	preprocess := flags.Bool("preprocess-reference", false, "normalise then trim reference silence like upstream (raw input only)")
	postprocess := flags.Bool("postprocess", false, "trim output silence, restore loudness, fade and pad like upstream")
	ref := flags.String("reference", "", "reference audio path (WAV or supported MP4/AAC)")
	transcript := flags.String("transcript", "", "exact transcript for native reference encoding")
	layer := flags.Int("layer", 0, "decoder layer to evaluate")
	tokens := flags.Int("tokens", 3, "synthetic token count for block probe (1..256)")
	threads := flags.Int("threads", 2, "Go execution threads")
	workers := flags.Int("gemm-workers", 0, "persistent SIMD GEMM workers (0 keeps serial path; may exceed threads for oversubscription tests)")
	backend := flags.String("backend", "cpu", "cpu, auto, or vulkan (Vulkan dispatch not implemented)")
	iterations := flags.Int("iterations", 1, "resident block repetitions (1..1000)")
	cpuProfile := flags.String("cpuprofile", "", "exclusive-create CPU profile for the whole command")
	memProfile := flags.String("memprofile", "", "exclusive-create allocation profile for the whole command")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if err := validateGuidance(*guidanceFlag, *shared); err != nil {
		return err
	}
	guidance := float32(*guidanceFlag)
	guidanceSet := false
	flags.Visit(func(f *flag.Flag) {
		if f.Name == "guidance" {
			guidanceSet = true
		}
	})
	if guidanceSet && *mode != "generate" && *mode != "synthesize" && *mode != "synthesize-long" && *mode != "serve" {
		return fmt.Errorf("guidance applies only to synthesis/generation")
	}
	if *shared && (*residentMiB != 0 || *prepacked || *directQ8 || (*mode != "generate" && *mode != "synthesize" && *mode != "synthesize-long" && *mode != "serve")) {
		return fmt.Errorf("shared-traversal requires streamed synthesis/generation without resident/prepack/direct-q8")
	}
	if *directQ8 && (*mode != "generate" || *ggufPath == "" || *residentMiB != 0 || *prepacked) {
		return fmt.Errorf("direct-q8 requires generate with weights-gguf and no resident/prepack")
	}
	if *residentMiB < 0 || *residentMiB > 65536 {
		return fmt.Errorf("resident-mib must be 0..65536")
	}
	if *residentMiB != 0 && *mode != "synthesize" && *mode != "synthesize-long" && *mode != "generate" && *mode != "serve" {
		return fmt.Errorf("resident-mib applies only to synthesize, synthesize-long or generate")
	}
	if *prepacked && !model.DiscoverBackend(model.BackendCPU).CPU.Capabilities.HasSGEMM {
		return fmt.Errorf("prepack requires active SIMD GEMM on this host")
	}
	if *prepacked && *residentMiB == 0 {
		return fmt.Errorf("prepack requires a positive resident-mib budget")
	}
	residentBytes := *residentMiB << 20
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments")
	}
	if *workers < 0 || *workers > 64 {
		return fmt.Errorf("gemm-workers must be 0..64")
	}
	if *workers > 0 && *mode != "synthesize" && *mode != "synthesize-long" && *mode != "generate" && *mode != "serve" {
		return fmt.Errorf("gemm-workers applies only to synthesis/generation")
	}
	if *threads < 1 || *threads > runtime.NumCPU() {
		return fmt.Errorf("threads out of range")
	}
	runtime.GOMAXPROCS(*threads)
	if *mode == "vulkan-probe-internal" {
		available := vk.VulkanInit()
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"available": available, "name": vk.VulkanDeviceName()})
	}
	if *iterations < 1 || *iterations > 1000 {
		return fmt.Errorf("iterations must be 1..1000")
	}
	stop, err := startProfiles(*cpuProfile, *memProfile)
	if err != nil {
		return err
	}
	defer stop()
	backendMode, err := model.ParseBackendMode(*backend)
	if err != nil {
		return err
	}
	if *mode == "capabilities" {
		return json.NewEncoder(os.Stdout).Encode(model.DiscoverBackend(backendMode))
	}
	if *mode == "audio" {
		if *ref == "" {
			return fmt.Errorf("reference path required")
		}
		return inspectAudio(*ref)
	}
	if *mode == "serve" {
		if *ref != "" || *preprocess || *input != "" || *text != "" || *output != "" {
			return fmt.Errorf("serve uses cached reference and stdin requests; no raw reference/input/text/output flags")
		}
		if *cacheMiB < 0 || *cacheMiB > 256 {
			return fmt.Errorf("cache-mib must be 0..256")
		}
		if _, err := model.SelectBackend(backendMode); err != nil {
			return err
		}
		return runServe(serveOptions{guidance: &guidance, shared: *shared, modelPath: *path, weightsPath: *ggufPath, reference: *cachedReference, outputDir: *serveDir, language: *language, instruct: *instruct, frames: *frames, firstFrames: *firstFrames, steps: *steps, workers: *workers, residentBytes: residentBytes, cacheBytes: *cacheMiB << 20, prepacked: *prepacked, denoise: *denoise, postprocess: *postprocess})
	}
	if *serveDir != "" || *cacheMiB != 0 || *firstFrames != 0 {
		return fmt.Errorf("output-dir/cache-mib/first-frames apply only to serve")
	}
	if *mode == "export-gguf" {
		if *path == "" || *output == "" || filepath.Ext(*output) != ".gguf" || *ggufPath != "" {
			return fmt.Errorf("export-gguf requires -model and new -output .gguf, without -weights-gguf")
		}
		w, err := loader.OpenWeights(*path)
		if err != nil {
			return err
		}
		defer w.Close()
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()
		if err := loader.ExportGGUF(ctx, w, *output, *ggufFormat); err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"output": *output, "storage": *ggufFormat, "backbone_only": true, "seconds": time.Since(commandStarted).Seconds()})
	}
	if *ggufPath != "" && *mode != "generate" && *mode != "logits" && *mode != "block" && *mode != "stack" {
		return fmt.Errorf("weights-gguf applies only to generate/logits/block/stack")
	}
	if *mode != "inspect" && *mode != "block" && *mode != "stack" && *mode != "logits" && *mode != "generate" && *mode != "prepare" && *mode != "synthesize" && *mode != "encode-reference" && *mode != "plan-chunks" && *mode != "synthesize-long" {
		return fmt.Errorf("unknown mode %q", *mode)
	}
	if *mode == "plan-chunks" || *mode == "synthesize-long" {
		if *path == "" || *input != "" {
			return fmt.Errorf("chunk mode requires -model and text/reference flags, not -input")
		}
		if _, err := model.SelectBackend(backendMode); err != nil {
			return err
		}
		return runChunked(*path, *mode, *output, *text, *ref, *transcript, *cachedReference, *language, *instruct, *frames, *steps, *denoise, *preprocess, *postprocess, residentBytes, *prepacked, *workers, *shared, guidance)
	}

	if *preprocess && (*ref == "" || (*mode != "synthesize" && *mode != "prepare" && *mode != "encode-reference")) {
		return fmt.Errorf("preprocess-reference requires raw reference preparation")
	}
	if (*mode == "generate" || *mode == "synthesize") && (*steps < 1 || *steps > 128) {
		return fmt.Errorf("steps must be 1..128")
	}
	if *postprocess && *mode != "generate" && *mode != "synthesize" {
		return fmt.Errorf("postprocess requires generate/synthesize")
	}
	if err := validatePreparationFlags(*mode, *input, *output, *text, *ref, *transcript, *cachedReference, *frames, *steps); err != nil {
		return err
	}
	if *path == "" {
		return fmt.Errorf("model directory required")
	}
	cfg, err := loader.LoadConfig(*path)
	if err != nil {
		return err
	}
	if err := model.ValidateConfig(cfg.LLMConfig); err != nil {
		return err
	}
	if *mode == "inspect" {
		metadata, err := loader.ValidateCheckpoint(*path, cfg)
		if err != nil {
			return err
		}
		if err = json.NewEncoder(os.Stdout).Encode(metadata); err != nil {
			return err
		}
		if !metadata.Valid {
			return fmt.Errorf("invalid checkpoint")
		}
		return nil
	}
	if *mode == "encode-reference" {
		if *output == "" || filepath.Ext(*output) != ".json" {
			return fmt.Errorf("encode-reference requires new -output .json")
		}
		if *ref == "" || *transcript == "" || *cachedReference != "" {
			return fmt.Errorf("encode-reference requires -reference and -transcript")
		}
		if _, err := model.SelectBackend(backendMode); err != nil {
			return err
		}
		if _, err := os.Lstat(*output); !os.IsNotExist(err) {
			return fmt.Errorf("output exists or cannot be checked")
		}
		encoded, err := encodeReference(*path, *ref, *transcript, *preprocess)
		if err != nil {
			return err
		}
		return writeReference(*output, encoded)
	}
	var prompt loader.PreparedPrompt
	if *mode == "prepare" || *mode == "synthesize" {
		if *input != "" {
			return fmt.Errorf("prepare/synthesize use -text and -reference-tokens, not -input")
		}
		if *ref != "" {
			if *cachedReference != "" || *transcript == "" {
				return fmt.Errorf("raw reference requires -transcript and no -reference-tokens")
			}
			if _, err := model.SelectBackend(backendMode); err != nil {
				return err
			}
			prompt, err = prepareRawPrompt(*path, *ref, *transcript, *text, *frames, *language, *instruct, *denoise, *preprocess)
		} else {
			prompt, err = prepareCachedPrompt(*path, *cachedReference, *text, *frames, *language, *instruct, *denoise)
		}
		if err != nil {
			return err
		}
		if *mode == "prepare" {
			return writePreparedPrompt(*output, prompt)
		}
	}
	selection, err := model.SelectBackend(backendMode)
	if err != nil {
		return err
	}
	if *tokens < 1 || *tokens > 256 {
		return fmt.Errorf("tokens must be 1..256")
	}
	start := time.Now()
	weightPath := *path
	if *ggufPath != "" {
		weightPath = *ggufPath
	}
	weights, err := loader.OpenWeights(weightPath)
	if err != nil {
		return err
	}
	defer weights.Close()
	if !reflect.DeepEqual(cfg, weights.Config) {
		return fmt.Errorf("weights config differs from model config")
	}
	if *mode == "synthesize" {
		p := generationPrompt(prompt)
		p.NativeReference = *ref != ""
		p.PreprocessedReference = *preprocess
		p.Postprocess = *postprocess
		p.CommandStarted = commandStarted
		p.Reference = *ref
		return generatePrompt(weights, p, *output, filepath.Join(*path, "audio_tokenizer"), *steps, false, residentBytes, *prepacked, *workers, false, *shared, guidance)
	}
	if *mode == "generate" {
		return runGenerate(weights, *input, *output, filepath.Join(*path, "audio_tokenizer"), *steps, *postprocess, residentBytes, *prepacked, *workers, *directQ8, *shared, guidance)
	}
	if *mode == "logits" {
		return runLogits(weights, *input)
	}
	if *mode == "stack" {
		return profileStack(weights, *tokens, selection)
	}
	w, err := weights.Layer(*layer)
	if err != nil {
		return err
	}
	b, err := model.NewBlock(cfg.LLMConfig, w)
	if err != nil {
		return err
	}
	x := make([]float32, *tokens*cfg.LLMConfig.HiddenSize)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.01) * 0.1)
	}
	scratch, err := b.NewWorkspace(*tokens)
	if err != nil {
		return err
	}
	out := make([]float32, len(x))
	loaded := time.Now()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for i := 0; i < *iterations; i++ {
		if err = b.ForwardInto(out, x, *tokens, nil, nil, scratch); err != nil {
			return err
		}
	}
	elapsed := time.Since(loaded)
	runtime.ReadMemStats(&after)
	peak := float32(0)
	sum := float64(0)
	for _, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("nonfinite output")
		}
		sum += float64(v)
		peak = max(peak, float32(math.Abs(float64(v))))
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"mode": "block", "speech_generation": false, "layer": *layer, "tokens": *tokens, "hidden_size": cfg.LLMConfig.HiddenSize, "sgemm_asm": simd.HasSgemmAsm, "load_seconds": loaded.Sub(start).Seconds(), "forward_seconds": elapsed.Seconds() / float64(*iterations), "iterations": *iterations, "hot_allocations": after.Mallocs - before.Mallocs, "hot_allocated_bytes": after.TotalAlloc - before.TotalAlloc, "scratch_bytes": scratch.ScratchBytes(), "backend": selection.Backend, "sum": sum, "peak": peak, "first_values": out[:min(8, len(out))]})
}
func inspectAudio(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return err
	}
	decoder, err := audio.Open(context.Background(), f, stat.Size(), audio.Options{TargetRate: 24000, TargetChannels: 1})
	if err != nil {
		return err
	}
	defer decoder.Close()
	buf := make([]int16, 4096)
	count := 0
	sum := float64(0)
	peak := 0
	const maxSamples = 20 * 24000
	for {
		n, _, err := decoder.ReadPCM(context.Background(), buf)
		count += n
		if count > maxSamples {
			return fmt.Errorf("reference exceeds 20 seconds")
		}
		for _, v := range buf[:n] {
			a := int(v)
			if a < 0 {
				a = -a
			}
			peak = max(peak, a)
			s := float64(v) / 32768
			sum += s * s
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrNoProgress
		}
	}
	if count < 2*24000 {
		return fmt.Errorf("reference shorter than 2 seconds")
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"mode": "audio", "backend": "go-264/audio", "sample_rate": 24000, "channels": 1, "samples": count, "seconds": float64(count) / 24000, "peak": float64(peak) / 32768, "rms": math.Sqrt(sum / float64(count)), "codec_encoding": false})
}

func validateGuidance(value float64, shared bool) error {
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > math.MaxFloat32 || (value > 0 && float32(value) == 0) {
		return fmt.Errorf("guidance must be finite, nonnegative and representable as float32")
	}
	if shared && value == 0 {
		return fmt.Errorf("shared-traversal requires positive guidance")
	}
	return nil
}
