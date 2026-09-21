package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	q "github.com/rcarmo/go-pherence/model/qwenimage21"
	"os"
	"time"
)

func main() {
	backend := flag.String("backend", "", "stable-diffusion.cpp sd-cli executable")
	dit := flag.String("diffusion-model", "", "Qwen Image 2.1 diffusion GGUF/safetensors")
	vae := flag.String("vae", "", "Qwen Image 2.1 VAE safetensors")
	llm := flag.String("llm", "", "Qwen3-VL-8B text encoder GGUF/safetensors")
	prompt := flag.String("prompt", "", "prompt")
	out := flag.String("out", "qwen-image-2.1.png", "output PNG")
	width := flag.Int("width", 1024, "width (multiple of 32)")
	height := flag.Int("height", 1024, "height (multiple of 32)")
	steps := flag.Int("steps", 20, "Euler steps")
	cfg := flag.Float64("cfg-scale", 6, "CFG scale")
	seed := flag.Int64("seed", 42, "seed")
	threads := flag.Int("threads", 1, "CPU threads")
	timeout := flag.Duration("timeout", 30*time.Minute, "generation timeout")
	inspect := flag.Bool("inspect", false, "inspect diffusion weight readiness without generating")
	flag.Parse()
	if *inspect {
		var inv q.Inventory
		var e error
		if len(*dit) > 5 && (*dit)[len(*dit)-5:] == ".gguf" {
			inv, e = q.InspectGGUF(*dit)
		} else {
			inv, e = q.InspectSafetensors(*dit)
		}
		if e != nil {
			fatal(e)
		}
		json.NewEncoder(os.Stdout).Encode(inv)
		return
	}
	r := q.Runner{Executable: *backend, Assets: q.Assets{DiffusionModel: *dit, VAE: *vae, LLM: *llm}}
	result, e := r.Generate(context.Background(), q.GenerateOptions{Prompt: *prompt, Output: *out, Width: *width, Height: *height, Steps: *steps, Guidance: float32(*cfg), Seed: *seed, Threads: *threads, Timeout: *timeout})
	if e != nil {
		fatal(e)
	}
	b, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(b))
}
func fatal(e error) { fmt.Fprintln(os.Stderr, "qwenimage21:", e); os.Exit(1) }
