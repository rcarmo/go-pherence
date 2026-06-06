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
	flag.Parse()

	if *modelDir == "" || *prompt == "" {
		fmt.Fprintln(os.Stderr, "usage: ideogram4gen -model PATH -prompt TEXT [-out png] [-height H -width W -steps N -guidance G -seed S]")
		os.Exit(2)
	}

	pipe, err := ideogram4.LoadNativePipeline(*modelDir)
	if err != nil {
		fatal(err)
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

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ideogram4gen:", err)
	os.Exit(1)
}
