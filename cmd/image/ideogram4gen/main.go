package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"

	"github.com/rcarmo/go-pherence/model/ideogram4"
)

func main() {
	modelDir := flag.String("model", "", "Ideogram4 Diffusers model directory")
	prompt := flag.String("prompt", "", "text prompt")
	out := flag.String("out", "ideogram4.png", "output PNG path")
	height := flag.Int("height", 1024, "image height")
	width := flag.Int("width", 1024, "image width")
	steps := flag.Int("steps", 28, "sampling steps")
	guidance := flag.Float64("guidance", 5.0, "CFG guidance scale")
	seed := flag.Int64("seed", 0, "init-noise seed")
	gpu := flag.Bool("gpu", false, "enable all Ideogram4 NVIDIA GPU gates (FP8, norm, RoPE/MRoPE, attention, CFG, MLP, VAE)")
	gpuStrict := flag.Bool("gpu-strict", false, "enable strict GPU validation for all enabled Ideogram4 GPU gates (no CPU fallback on GPU errors)")
	gpuFP8Cache := flag.Bool("gpu-fp8-cache", false, "cache uploaded Ideogram4 FP8 linear weights on GPU when FP8 GPU path is enabled")
	gpuResidency := flag.String("gpu-residency", "", "GPU residency policy: persistent, phase, or stream (empty leaves environment/default unchanged)")
	flag.Parse()

	applyGPUFlags(*gpu, *gpuStrict, *gpuFP8Cache, *gpuResidency)

	if *modelDir == "" || *prompt == "" {
		fmt.Fprintln(os.Stderr, "usage: ideogram4gen -model PATH -prompt TEXT [-out png] [-height H -width W -steps N -guidance G -seed S]")
		os.Exit(2)
	}

	pipe, err := ideogram4.LoadNativePipeline(*modelDir)
	if err != nil {
		fatal(err)
	}
	defer pipe.ReleaseGPU()
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

	img, err := pipe.Generate(*prompt, ideogram4.GenerateOptions{
		Height:        *height,
		Width:         *width,
		Steps:         *steps,
		GuidanceScale: float32(*guidance),
		InitLatents:   init,
	})
	if err != nil {
		fatal(err)
	}
	if err := writePNG(*out, img); err != nil {
		fatal(err)
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

func applyGPUFlags(gpu, strict, fp8Cache bool, residency string) {
	if gpu {
		for _, k := range []string{
			"GO_PHERENCE_IDEOGRAM4_GPU_FP8",
			"GO_PHERENCE_IDEOGRAM4_GPU_NORM",
			"GO_PHERENCE_IDEOGRAM4_GPU_MROPE",
			"GO_PHERENCE_IDEOGRAM4_GPU_ATTN",
			"GO_PHERENCE_IDEOGRAM4_GPU_CFG",
			"GO_PHERENCE_IDEOGRAM4_GPU_MLP",
			"GO_PHERENCE_IDEOGRAM4_GPU_VAE",
		} {
			_ = os.Setenv(k, "1")
		}
	}
	if strict {
		for _, k := range []string{
			"GO_PHERENCE_IDEOGRAM4_GPU_FP8_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_NORM_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_MROPE_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_ATTN_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_CFG_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_MLP_STRICT",
			"GO_PHERENCE_IDEOGRAM4_GPU_VAE_STRICT",
		} {
			_ = os.Setenv(k, "1")
		}
	}
	if fp8Cache {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8_CACHE", "1")
	}
	if residency != "" {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_RESIDENCY", residency)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ideogram4gen:", err)
	os.Exit(1)
}
