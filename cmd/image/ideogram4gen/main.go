package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/model/ideogram4"
)

func main() {
	modelDir := flag.String("model", "", "Ideogram4 Diffusers model directory")
	prompt := flag.String("prompt", "", "text prompt")
	out := flag.String("out", "ideogram4.png", "output PNG path")
	height := flag.Int("height", 1024, "image height")
	width := flag.Int("width", 1024, "image width")
	steps := flag.Int("steps", 28, "sampling steps")
	guidance := flag.Float64("guidance", 7.0, "CFG guidance scale (OSS custom sampler default: 7.0)")
	mu := flag.Float64("mu", 0.0, "Ideogram logit-normal scheduler known-mean/mu (OSS custom/default-20 default: 0.0)")
	std := flag.Float64("std", 1.75, "Ideogram logit-normal scheduler std (OSS default-20/custom default: 1.75)")
	seed := flag.Int64("seed", 0, "init-noise seed")
	gpu := flag.Bool("gpu", false, "enable production-safe coarse Ideogram4 NVIDIA GPU gates (CFG and VAE; token/row-level experimental gates remain opt-in via env or -gpu-fp8)")
	gpuStrict := flag.Bool("gpu-strict", false, "enable strict GPU validation for enabled Ideogram4 GPU gates (no CPU fallback on GPU errors)")
	gpuFP8 := flag.Bool("gpu-fp8", false, "enable experimental Ideogram4 FP8 projection offload to NVIDIA")
	gpuFP8Cache := flag.Bool("gpu-fp8-cache", false, "cache uploaded Ideogram4 FP8 linear weights on GPU when FP8 GPU path is enabled")
	gpuFP8SGEMM := flag.Bool("gpu-fp8-sgemm", false, "use tiled SGEMM for batched no-bias FP8 projections by GPU-dequantizing weights to temporary F32")
	gpuResidency := flag.String("gpu-residency", "", "GPU residency policy: persistent, phase, or stream (empty leaves environment/default unchanged)")
	k3 := flag.Bool("k3", false, "enable SpacemiT K3/RVV/IME Ideogram kernels where available")
	k3Threads := flag.Int("k3-threads", 0, "SpacemiT K3 worker/thread count for K3 kernels (0 leaves environment/default unchanged)")
	k3Prewarm := flag.Bool("k3-prewarm", false, "pre-decode/pre-pack K3 resident FP8 linears at pipeline load")
	timing := flag.Bool("timing", false, "print coarse Ideogram4 generation timing diagnostics")
	flag.Parse()

	applyGPUFlags(*gpu, *gpuStrict, *gpuFP8, *gpuFP8Cache, *gpuFP8SGEMM, *gpuResidency)
	applyK3Flags(*k3, *k3Threads, *k3Prewarm)
	if *timing {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_TIMING", "1")
	}

	if *modelDir == "" || *prompt == "" {
		fmt.Fprintln(os.Stderr, "usage: ideogram4gen -model PATH -prompt TEXT [-out png] [-height H -width W -steps N -guidance G -seed S]")
		os.Exit(2)
	}

	loadStart := time.Now()
	pipe, err := ideogram4.LoadNativePipeline(*modelDir)
	if err != nil {
		fatal(err)
	}
	defer pipe.ReleaseGPU()
	if *timing {
		fmt.Fprintf(os.Stderr, "timing load_pipeline=%s\n", time.Since(loadStart))
	}
	cfg := pipe.Config
	gh, gw, err := cfg.LatentGrid(*height, *width)
	if err != nil {
		fatal(err)
	}
	n := gh * gw * cfg.InChannels
	rng := rand.New(rand.NewSource(*seed))
	init := make([]float32, n)
	for i := range init {
		init[i] = float32(rng.NormFloat64())
	}

	genStart := time.Now()
	schedule, err := ideogram4.DefaultScheduleForResolution(*height, *width, 512, 512, *mu, *std)
	if err != nil {
		fatal(err)
	}
	img, err := pipe.Generate(*prompt, ideogram4.GenerateOptions{
		Height:        *height,
		Width:         *width,
		Steps:         *steps,
		GuidanceScale: float32(*guidance),
		Schedule:      schedule,
		InitLatents:   init,
	})
	if err != nil {
		fatal(err)
	}
	if *timing {
		fmt.Fprintf(os.Stderr, "timing generate=%s\n", time.Since(genStart))
	}
	writeStart := time.Now()
	if err := writePNG(*out, img); err != nil {
		fatal(err)
	}
	if *timing {
		fmt.Fprintf(os.Stderr, "timing write_png=%s total_after_load=%s\n", time.Since(writeStart), time.Since(genStart))
	}
	fmt.Printf("wrote %s (%dx%d)\n", *out, img.Width, img.Height)
}

func writePNG(path string, im ideogram4.Image) error {
	out := image.NewRGBA(image.Rect(0, 0, im.Width, im.Height))
	for y := 0; y < im.Height; y++ {
		for x := 0; x < im.Width; x++ {
			i := (y*im.Width + x) * 3
			out.Set(x, y, color.RGBA{R: im.RGB[i], G: im.RGB[i+1], B: im.RGB[i+2], A: 255})
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, out)
}

func applyGPUFlags(gpu, strict, fp8, fp8Cache, fp8SGEMM bool, residency string) {
	if gpu {
		for _, k := range []string{
			"GO_PHERENCE_IDEOGRAM4_GPU_CFG",
			"GO_PHERENCE_IDEOGRAM4_GPU_VAE",
		} {
			_ = os.Setenv(k, "1")
		}
	}
	if fp8 || fp8Cache {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8", "1")
	}
	if strict {
		for _, k := range []string{
			"GO_PHERENCE_IDEOGRAM4_GPU_NORM_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_MROPE_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_ATTN_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_CFG_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_MLP_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_VAE_STRICT",
		} {
			_ = os.Setenv(k, "1")
		}
		if fp8 || fp8Cache {
			_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8_STRICT", "1")
		}
	}
	if fp8Cache {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8_CACHE", "1")
	}
	if fp8SGEMM {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8", "1")
		_ = os.Setenv("GO_PHERENCE_NVIDIA_FP8_SGEMM", "1")
	}
	if residency != "" {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_RESIDENCY", residency)
	}
}

func applyK3Flags(k3 bool, threads int, prewarm bool) {
	if k3 {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_K3", "1")
	}
	if threads > 0 {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_K3_THREADS", fmt.Sprintf("%d", threads))
	}
	if prewarm {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_K3_PREWARM", "1")
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ideogram4gen:", err)
	os.Exit(1)
}
