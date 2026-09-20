package main

import (
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/ideogram4"
)

func main() {
	path := flag.String("vae", "", "path to vae safetensors")
	grid := flag.Int("grid", 16, "latent map H=W")
	flag.Parse()
	if *path == "" {
		fmt.Fprintln(os.Stderr, "usage: ideogram4vaesmoke -vae file.safetensors")
		os.Exit(2)
	}
	if err := runSmoke(*path, *grid, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// This diagnostic intentionally has a small latent-grid admission limit; it is
// not an unrestricted generation command or a process-wide memory budget.
func runSmoke(path string, grid int, out io.Writer) error {
	if grid < 1 || grid > 64 {
		return fmt.Errorf("grid must be within 1..64")
	}
	f, err := safetensors.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec, err := ideogram4.NewVAEDecoder(f, ideogram4.VAEDecoderOptions{
		BlockOutChannels: []int{128, 256, 512, 512},
		LayersPerBlock:   2,
		LatentChannels:   32,
		NormNumGroups:    32,
		ScalingFactor:    1,
		UsePostQuantConv: true,
		MidAddAttention:  true,
	})
	if err != nil {
		return err
	}
	H := grid
	z := ideogram4.FeatureMap{C: 32, H: H, W: H, Data: make([]float32, 32*H*H)}
	rng := rand.New(rand.NewSource(1))
	for i := range z.Data {
		z.Data[i] = float32(rng.NormFloat64()) * 0.5
	}
	img, err := dec.Decode(z)
	if err != nil {
		return err
	}
	if len(img.RGB) == 0 {
		return fmt.Errorf("decoder returned empty image")
	}
	var min, max, sum float64
	min = 1e9
	for _, b := range img.RGB {
		v := float64(b)
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
	}
	_, err = fmt.Fprintf(out, "decoded image %dx%d rgb_bytes=%d min=%.0f max=%.0f mean=%.1f\n",
		img.Width, img.Height, len(img.RGB), min, max, sum/float64(len(img.RGB)))
	return err
}
