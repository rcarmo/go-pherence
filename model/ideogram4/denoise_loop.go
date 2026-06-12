package ideogram4

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// DenoiseLoop runs the full FlowMatch sampling loop with asymmetric CFG over a
// conditional/unconditional DiT pair.
//
//   - cond:        conditional transformer (text+image joint sequence)
//   - uncond:      unconditional transformer (image-only sequence)
//   - latents:     initial noise latents [imageTokens, in_channels]
//   - gridH/gridW: latent grid layout
//   - textFeatures:[textTokens, llm_features_dim] Qwen3-VL conditioning
//   - plan:        sampling plan (steps + guidance schedule) for this resolution
//
// Returns the denoised latents [imageTokens, in_channels].
func DenoiseLoop(cond, uncond *DiTModel, sched *FlowMatchScheduler, plan SamplingPlan, latents []float32, gridH, gridW int, textFeatures []float32) ([]float32, error) {
	if cond == nil || uncond == nil {
		return nil, fmt.Errorf("ideogram4 denoise: nil transformer")
	}
	if sched == nil {
		return nil, fmt.Errorf("ideogram4 denoise: nil scheduler")
	}
	cfg := cond.Config
	imgTokens := gridH * gridW
	if imgTokens <= 0 || len(latents) != imgTokens*cfg.InChannels {
		return nil, fmt.Errorf("ideogram4 denoise: latents=%d want %d*%d", len(latents), imgTokens, cfg.InChannels)
	}
	if plan.ImageTokens != 0 && plan.ImageTokens != imgTokens {
		return nil, fmt.Errorf("ideogram4 denoise: plan image tokens=%d grid=%d", plan.ImageTokens, imgTokens)
	}
	if len(plan.Steps) == 0 {
		return nil, fmt.Errorf("ideogram4 denoise: empty sampling plan")
	}

	traceTiming := os.Getenv("GO_PHERENCE_IDEOGRAM4_TIMING") == "1"
	traceGPUStats := os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_STATS") == "1"
	printStats := func(name string, before nvidia.Stats, since time.Time) {
		if !traceGPUStats {
			return
		}
		now := nvidia.StatsSnapshot()
		fmt.Fprintf(os.Stderr, "gpu_stats %s elapsed=%s kernels=%d h2d=%d h2d_bytes=%d d2h=%d d2h_bytes=%d d2d=%d d2d_bytes=%d mallocs=%d malloc_bytes=%d frees=%d free_bytes=%d syncs=%d\n",
			name, time.Since(since),
			now.KernelLaunches-before.KernelLaunches,
			now.HostToDevice-before.HostToDevice,
			now.HostToDeviceBytes-before.HostToDeviceBytes,
			now.DeviceToHost-before.DeviceToHost,
			now.DeviceToHostBytes-before.DeviceToHostBytes,
			now.DeviceToDevice-before.DeviceToDevice,
			now.DeviceToDeviceBytes-before.DeviceToDeviceBytes,
			now.Mallocs-before.Mallocs,
			now.MallocBytes-before.MallocBytes,
			now.Frees-before.Frees,
			now.FreeBytes-before.FreeBytes,
			now.Syncs-before.Syncs)
	}

	x := Latents{Batch: 1, Tokens: imgTokens, Channels: cfg.InChannels, Data: append([]float32(nil), latents...)}
	for si, step := range plan.Steps {
		branchStart := time.Now()
		branchStats := nvidia.StatsSnapshot()
		condVel, err := withBranchLayerCache("COND", func() ([]float32, error) {
			return cond.Velocity(x.Data, gridH, gridW, textFeatures, step.T)
		})
		if err != nil {
			return nil, fmt.Errorf("ideogram4 denoise step %d cond: %w", si, err)
		}
		if traceTiming {
			fmt.Fprintf(os.Stderr, "timing denoise_step=%d branch=cond elapsed=%s\n", si, time.Since(branchStart))
		}
		printStats(fmt.Sprintf("denoise_step_%d_cond", si), branchStats, branchStart)
		branchStart = time.Now()
		branchStats = nvidia.StatsSnapshot()
		uncondVel, err := withBranchLayerCache("UNCOND", func() ([]float32, error) {
			return uncond.Velocity(x.Data, gridH, gridW, nil, step.T)
		})
		if err != nil {
			return nil, fmt.Errorf("ideogram4 denoise step %d uncond: %w", si, err)
		}
		if traceTiming {
			fmt.Fprintf(os.Stderr, "timing denoise_step=%d branch=uncond elapsed=%s\n", si, time.Since(branchStart))
		}
		printStats(fmt.Sprintf("denoise_step_%d_uncond", si), branchStats, branchStart)
		condL := Latents{Batch: 1, Tokens: imgTokens, Channels: cfg.InChannels, Data: condVel}
		uncondL := Latents{Batch: 1, Tokens: imgTokens, Channels: cfg.InChannels, Data: uncondVel}
		branchStart = time.Now()
		branchStats = nvidia.StatsSnapshot()
		x, err = gpuCFGStepOrFallback(sched, plan, x, condL, uncondL, step, step.Index)
		if err != nil {
			return nil, fmt.Errorf("ideogram4 denoise step %d cfg/update: %w", si, err)
		}
		if traceTiming {
			fmt.Fprintf(os.Stderr, "timing denoise_step=%d branch=cfg elapsed=%s\n", si, time.Since(branchStart))
		}
		printStats(fmt.Sprintf("denoise_step_%d_cfg", si), branchStats, branchStart)
		runtime.GC()
		debug.FreeOSMemory()
	}
	return x.Data, nil
}
