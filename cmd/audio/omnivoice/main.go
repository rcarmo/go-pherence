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
	"runtime"
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
	flags := flag.NewFlagSet("omnivoice", flag.ContinueOnError)
	path := flags.String("model", "", "local OmniVoice model directory (required)")
	mode := flags.String("mode", "inspect", "inspect, block, stack, capabilities, or audio; no speech generation yet")
	ref := flags.String("reference", "", "reference audio path for audio mode (WAV or supported MP4/AAC)")
	layer := flags.Int("layer", 0, "decoder layer to evaluate")
	tokens := flags.Int("tokens", 3, "synthetic token count for block probe (1..256)")
	threads := flags.Int("threads", 2, "Go execution threads")
	backend := flags.String("backend", "cpu", "cpu, auto, or vulkan (Vulkan dispatch not implemented)")
	iterations := flags.Int("iterations", 1, "resident block repetitions (1..1000)")
	cpuProfile := flags.String("cpuprofile", "", "exclusive-create CPU profile for the whole command")
	memProfile := flags.String("memprofile", "", "exclusive-create allocation profile for the whole command")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments")
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
	if *mode != "inspect" && *mode != "block" && *mode != "stack" {
		return fmt.Errorf("unknown mode %q", *mode)
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
	selection, err := model.SelectBackend(backendMode)
	if err != nil {
		return err
	}
	if *tokens < 1 || *tokens > 256 {
		return fmt.Errorf("tokens must be 1..256")
	}
	start := time.Now()
	weights, err := loader.OpenWeights(*path)
	if err != nil {
		return err
	}
	defer weights.Close()
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
