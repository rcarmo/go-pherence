package main

import (
	"flag"
	"fmt"
	"github.com/rcarmo/go-pherence/cmd/image/internal/k3flags"
	"math/rand"
	"os"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/model/ideogram4"
)

func main() {
	modelDir := flag.String("model", "", "Ideogram4 Diffusers model directory")
	height := flag.Int("height", 512, "image height")
	width := flag.Int("width", 512, "image width")
	seed := flag.Int64("seed", 42, "random latent seed")
	gpu := flag.Bool("gpu", false, "enable Ideogram4 VAE GPU kernels")
	k3 := flag.Bool("k3", false, "enable SpacemiT K3/RVV/IME Ideogram kernels where available")
	k3Threads := flag.Int("k3-threads", 0, "SpacemiT K3 worker/thread count for K3 kernels (0 leaves environment/default unchanged)")
	k3Prewarm := flag.Bool("k3-prewarm", false, "pre-decode/pre-pack K3 resident FP8 linears at pipeline load")
	stats := flag.Bool("gpu-stats", false, "print NVIDIA GPU stats for VAE decode")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: ideogram4vaeprobe -model PATH [-height H -width W] [-gpu] [-gpu-stats]")
		os.Exit(2)
	}
	if *gpu {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_VAE", "1")
	}
	k3flags.Apply(*k3, *k3Threads, *k3Prewarm)
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
	z := make([]float32, gh*gw*cfg.InChannels)
	rng := rand.New(rand.NewSource(*seed))
	for i := range z {
		z[i] = float32(rng.NormFloat64())
	}
	if err := ideogram4.DenormalizeLatents(z, cfg.InChannels); err != nil {
		fatal(err)
	}
	fmap, err := ideogram4.UnpatchifyLatents(z, gh, gw, cfg.InChannels, cfg.VAELatentChannels, cfg.PatchSize, cfg.PatchSize)
	if err != nil {
		fatal(err)
	}
	prevStats := false
	var before nvidia.Stats
	if *stats {
		prevStats = nvidia.SetStatsEnabled(true)
		defer nvidia.SetStatsEnabled(prevStats)
		before = nvidia.StatsSnapshot()
	}
	start := time.Now()
	img, err := pipe.VAE.Decode(fmap)
	if err != nil {
		fatal(err)
	}
	elapsed := time.Since(start)
	fmt.Printf("vae_probe width=%d height=%d latent_grid=%dx%d elapsed=%s image=%dx%d\n", *width, *height, gh, gw, elapsed, img.Width, img.Height)
	if *stats {
		after := nvidia.StatsSnapshot()
		fmt.Printf("gpu_stats vae_probe kernels=%d h2d=%d h2d_bytes=%d d2h=%d d2h_bytes=%d d2d=%d d2d_bytes=%d mallocs=%d malloc_bytes=%d frees=%d free_bytes=%d syncs=%d\n",
			after.KernelLaunches-before.KernelLaunches,
			after.HostToDevice-before.HostToDevice,
			after.HostToDeviceBytes-before.HostToDeviceBytes,
			after.DeviceToHost-before.DeviceToHost,
			after.DeviceToHostBytes-before.DeviceToHostBytes,
			after.DeviceToDevice-before.DeviceToDevice,
			after.DeviceToDeviceBytes-before.DeviceToDeviceBytes,
			after.Mallocs-before.Mallocs,
			after.MallocBytes-before.MallocBytes,
			after.Frees-before.Frees,
			after.FreeBytes-before.FreeBytes,
			after.Syncs-before.Syncs)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ideogram4vaeprobe:", err)
	os.Exit(1)
}
