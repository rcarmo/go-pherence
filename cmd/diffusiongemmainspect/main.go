package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/model/diffusiongemma"
)

type report struct {
	ModelPath          string                             `json:"model_path"`
	Shape              diffusiongemma.Shape               `json:"shape"`
	GenerationDefaults *diffusiongemma.GenerationDefaults `json:"generation_defaults,omitempty"`
	Tensors            *diffusiongemma.TensorInventory    `json:"tensors,omitempty"`
	Readiness          *diffusiongemma.TensorReadiness    `json:"readiness,omitempty"`
	TextTensorPlan     *diffusiongemma.TextTensorPlan     `json:"text_tensor_plan,omitempty"`
	TextWeightsOpened  bool                               `json:"text_weights_opened,omitempty"`
	TextWeightsGlobals int                                `json:"text_weights_globals,omitempty"`
	TextWeightsLayers  int                                `json:"text_weights_layers,omitempty"`
	TextForwardPlan    *diffusiongemma.TextForwardPlan    `json:"text_forward_plan,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "DiffusionGemma Hugging Face model directory")
	asJSON := flag.Bool("json", false, "emit JSON report")
	requireRuntime := flag.Bool("require-runtime-ready", false, "fail unless native DiffusionGemma runtime is implemented")
	openWeights := flag.Bool("open-weights", false, "open local safetensor shards and bind text tensor metadata")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: diffusiongemmainspect -model PATH [-json] [-require-runtime-ready]")
		os.Exit(2)
	}
	m, err := diffusiongemma.LoadMetadata(*modelDir)
	if err != nil {
		fatal(err)
	}
	shape := m.Shape
	if *requireRuntime {
		if err := m.RequireRuntimeReady(); err != nil {
			fatal(err)
		}
	}
	out := report{ModelPath: *modelDir, Shape: shape, GenerationDefaults: m.GenerationDefaults, Tensors: m.Tensors, Readiness: m.Readiness, TextTensorPlan: m.TextTensorPlan}
	if *openWeights {
		weights, err := diffusiongemma.OpenTextWeights(*modelDir, shape)
		if err != nil {
			fatal(err)
		}
		defer weights.Close()
		out.TextWeightsOpened = true
		out.TextWeightsGlobals = len(weights.Globals)
		out.TextWeightsLayers = len(weights.Layers)
		forwardPlan := weights.ForwardPlan()
		out.TextForwardPlan = &forwardPlan
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
		fmt.Printf("  generate:  max_new=%d denoise_steps=%d t=[%.3f, %.3f] entropy_bound=%.3f stability=%d confidence=%.6f eos=%v\n", g.MaxNewTokens, g.MaxDenoisingSteps, g.TMin, g.TMax, g.EntropyBound, g.StabilityThreshold, g.ConfidenceThreshold, g.EOSTokenID)
	}
	if r.Tensors != nil {
		fmt.Printf("  tensors:   total=%d shards=%d groups=%v\n", r.Tensors.Total, r.Tensors.Shards, r.Tensors.Groups)
	}
	if r.Readiness != nil {
		fmt.Printf("  readiness: text_ready=%v vision_inventory=%v runtime_ready=%v observed_layers=%d/%d layer_tensors=%d/%d missing_layer_tensors=%d\n", r.Readiness.TextReady, r.Readiness.VisionInventoryReady, r.Readiness.RuntimeReady, r.Readiness.ObservedTextLayers, r.Readiness.ExpectedTextLayers, r.Readiness.ObservedLayerTensors, r.Readiness.ExpectedLayerTensors, r.Readiness.MissingLayerTensors)
		if len(r.Readiness.MissingRequired) > 0 {
			fmt.Printf("  missing:   %v\n", r.Readiness.MissingRequired)
		}
	}
	if r.TextTensorPlan != nil {
		fmt.Printf("  text_plan: ready=%v globals=%d layers=%d missing=%d\n", r.TextTensorPlan.Ready, len(r.TextTensorPlan.Globals), len(r.TextTensorPlan.Layers), len(r.TextTensorPlan.Missing))
	}
	if r.TextWeightsOpened {
		fmt.Printf("  weights:   text_shards_opened=true globals=%d layers=%d\n", r.TextWeightsGlobals, r.TextWeightsLayers)
	}
	if r.TextForwardPlan != nil {
		fmt.Printf("  forward:   text_ready=%v layers=%d missing=%d\n", r.TextForwardPlan.Ready, len(r.TextForwardPlan.Layers), len(r.TextForwardPlan.Missing))
	}
	if s.RuntimeNote != "" {
		fmt.Printf("  runtime:   %s\n", s.RuntimeNote)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "diffusiongemmainspect:", err)
	os.Exit(1)
}
