package main

import (
	"flag"
	"fmt"
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
	f, err := safetensors.Open(*path)
	if err != nil {
		panic(err)
	}
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
		panic(err)
	}
	H := *grid
	z := ideogram4.FeatureMap{C: 32, H: H, W: H, Data: make([]float32, 32*H*H)}
	rng := rand.New(rand.NewSource(1))
	for i := range z.Data {
		z.Data[i] = float32(rng.NormFloat64()) * 0.5
	}
	img, err := dec.Decode(z)
	if err != nil {
		panic(err)
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
	fmt.Printf("decoded image %dx%d rgb_bytes=%d min=%.0f max=%.0f mean=%.1f\n",
		img.Width, img.Height, len(img.RGB), min, max, sum/float64(len(img.RGB)))
}
