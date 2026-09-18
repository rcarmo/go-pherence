package main

import (
	"encoding/json"
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	model "github.com/rcarmo/go-pherence/models/omnivoice"
	"math"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

// Profiles use exclusive-create files: never overwrite model/user data.
func startProfiles(cpuPath, memPath string) (func(), error) {
	var cpu, mem *os.File
	var err error
	if cpuPath != "" {
		cpu, err = os.OpenFile(cpuPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			return nil, err
		}
	}
	if memPath != "" {
		mem, err = os.OpenFile(memPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			if cpu != nil {
				cpu.Close()
			}
			return nil, err
		}
	}
	oldRate := runtime.MemProfileRate
	if mem != nil {
		runtime.MemProfileRate = 1
	}
	if cpu != nil {
		if err = pprof.StartCPUProfile(cpu); err != nil {
			cpu.Close()
			if mem != nil {
				mem.Close()
			}
			runtime.MemProfileRate = oldRate
			return nil, err
		}
	}
	return func() {
		if cpu != nil {
			pprof.StopCPUProfile()
			cpu.Close()
		}
		if mem != nil {
			runtime.GC()
			if err := pprof.Lookup("allocs").WriteTo(mem, 0); err != nil {
				fmt.Fprintln(os.Stderr, "allocation profile:", err)
			}
			mem.Close()
			runtime.MemProfileRate = oldRate
		}
	}, nil
}

type layerMeasurement struct {
	Layer          int     `json:"layer"`
	LoadSeconds    float64 `json:"load_seconds"`
	ForwardSeconds float64 `json:"forward_seconds"`
	AllocatedBytes uint64  `json:"forward_allocated_bytes"`
	Allocations    uint64  `json:"forward_allocations"`
}

// profileStack chains all decoder layers and final RMSNorm, starting from
// synthetic hidden states. It is not text-to-audio or a denoising sampler.
// Layer streaming intentionally bounds weight memory at the cost of reloads.
func profileStack(weights *loader.Weights, tokens int, backend model.BackendSelection) error {
	c := weights.Config.LLMConfig
	x := make([]float32, tokens*c.HiddenSize)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*.01) * .1)
	}
	measurements := make([]layerMeasurement, 0, c.NumHiddenLayers)
	start := time.Now()
	var initial, final runtime.MemStats
	runtime.ReadMemStats(&initial)
	arena, err := weights.NewLayerBuffer()
	if err != nil {
		return err
	}
	block, err := model.NewBlock(c, arena.Tensors)
	if err != nil {
		return err
	}
	scratch, err := block.NewWorkspace(tokens)
	if err != nil {
		return err
	}
	for i := 0; i < c.NumHiddenLayers; i++ {
		record, err := profileLayer(weights, arena, block, i, x, tokens, scratch)
		if err != nil {
			return err
		}
		measurements = append(measurements, record)
	}
	norm, err := weights.Float32("llm.norm.weight")
	if err != nil {
		return err
	}
	for i := 0; i < tokens; i++ {
		simd.RMSNorm(x[i*c.HiddenSize:(i+1)*c.HiddenSize], norm, float32(c.RMSNormEps))
	}
	runtime.ReadMemStats(&final)
	sum := float64(0)
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("nonfinite stack output")
		}
		sum += float64(v)
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"mode": "stack", "speech_generation": false, "inputs": "synthetic hidden states", "weight_policy": "stream into reusable float32 arena", "weight_arena_bytes": arena.Bytes(), "backend": backend.Backend, "layers": measurements, "tokens": tokens, "scratch_bytes": scratch.ScratchBytes(), "total_seconds": time.Since(start).Seconds(), "setup_and_stack_allocated_bytes": final.TotalAlloc - initial.TotalAlloc, "setup_and_stack_allocations": final.Mallocs - initial.Mallocs, "allocation_scope": "arena/block/workspace setup, layer passes and final norm; excludes checkpoint open, input setup and JSON output", "sum": sum})
}
func profileLayer(weights *loader.Weights, arena *loader.LayerBuffer, block *model.Block, index int, x []float32, tokens int, scratch *model.Workspace) (layerMeasurement, error) {
	m := layerMeasurement{Layer: index}
	start := time.Now()
	err := arena.Load(weights, index)
	if err != nil {
		return m, err
	}
	m.LoadSeconds = time.Since(start).Seconds()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start = time.Now()
	err = block.ForwardInto(x, x, tokens, nil, nil, scratch)
	m.ForwardSeconds = time.Since(start).Seconds()
	runtime.ReadMemStats(&after)
	m.Allocations = after.Mallocs - before.Mallocs
	m.AllocatedBytes = after.TotalAlloc - before.TotalAlloc
	return m, err
}
