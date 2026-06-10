package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/model/diffusiongemma"
)

type report struct {
	ModelPath          string                             `json:"model_path"`
	Shape              diffusiongemma.Shape               `json:"shape"`
	GenerationDefaults *diffusiongemma.GenerationDefaults `json:"generation_defaults,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "DiffusionGemma Hugging Face model directory")
	asJSON := flag.Bool("json", false, "emit JSON report")
	requireRuntime := flag.Bool("require-runtime-ready", false, "fail unless native DiffusionGemma runtime is implemented")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: diffusiongemmainspect -model PATH [-json] [-require-runtime-ready]")
		os.Exit(2)
	}
	cfg, err := loaderconfig.ReadDiffusionGemmaConfig(*modelDir)
	if err != nil {
		fatal(err)
	}
	shape, err := diffusiongemma.ShapeFromConfig(cfg)
	if err != nil {
		fatal(err)
	}
	if *requireRuntime {
		if err := diffusiongemma.RequireRuntimeReady(shape); err != nil {
			fatal(err)
		}
	}
	out := report{ModelPath: *modelDir, Shape: shape}
	if gen, ok, err := loaderconfig.ReadDiffusionGemmaGenerationConfig(*modelDir); err != nil {
		fatal(err)
	} else if ok {
		defaults := diffusiongemma.GenerationDefaultsFromConfig(gen)
		out.GenerationDefaults = &defaults
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}
		return
	}
	printText(out)
}

func printText(r report) {
	s := r.Shape
	fmt.Printf("DiffusionGemma model: %s\n", r.ModelPath)
	fmt.Printf("  arch:      %s model_type=%s dtype=%s runtime_ready=%v\n", s.Architecture, s.ModelType, s.DType, s.RuntimeReady)
	fmt.Printf("  text:      hidden=%d layers=%d heads=%d kv_heads=%d global_kv_heads=%d head_dim=%d vocab=%d sliding_window=%d\n", s.TextHiddenSize, s.TextLayers, s.TextHeads, s.TextKVHeads, s.TextGlobalKVHeads, s.TextHeadDim, s.VocabSize, s.SlidingWindow)
	fmt.Printf("  moe:       experts=%d active=%d moe_intermediate=%d\n", s.NumExperts, s.TopKExperts, s.MoEIntermediateSize)
	fmt.Printf("  canvas:    length=%d boi=%d eoi=%d image=%d\n", s.CanvasLength, s.BOITokenID, s.EOITokenID, s.ImageTokenID)
	fmt.Printf("  vision:    hidden=%d layers=%d heads=%d soft_tokens=%d patch=%d\n", s.VisionHiddenSize, s.VisionLayers, s.VisionHeads, s.VisionSoftTokens, s.PatchSize)
	if r.GenerationDefaults != nil {
		g := r.GenerationDefaults
		fmt.Printf("  generate:  max_new=%d denoise_steps=%d t=[%.3f, %.3f] stability=%d confidence=%.6f eos=%v\n", g.MaxNewTokens, g.MaxDenoisingSteps, g.TMin, g.TMax, g.StabilityThreshold, g.ConfidenceThreshold, g.EOSTokenID)
	}
	if s.RuntimeNote != "" {
		fmt.Printf("  runtime:   %s\n", s.RuntimeNote)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "diffusiongemmainspect:", err)
	os.Exit(1)
}
