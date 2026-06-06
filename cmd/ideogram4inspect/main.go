package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/model/ideogram4"
)

func main() {
	modelDir := flag.String("model", "", "Ideogram 4 Diffusers model directory")
	asJSON := flag.Bool("json", false, "emit JSON report")
	transformerIndex := flag.String("transformer-index", "", "optional transformer safetensors index JSON")
	uncondIndex := flag.String("unconditional-transformer-index", "", "optional unconditional transformer safetensors index JSON")
	requireRuntime := flag.Bool("require-runtime-ready", false, "fail unless image generation runtime is implemented")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: ideogram4inspect -model PATH [-json] [-require-runtime-ready]")
		os.Exit(2)
	}
	cfg, err := loaderconfig.ReadIdeogram4Config(*modelDir)
	if err != nil {
		fatal(err)
	}
	shape, err := ideogram4.FromLoaderConfig(cfg)
	if err != nil {
		fatal(err)
	}
	s := loaderconfig.SummarizeIdeogram4Config(cfg)
	if *requireRuntime && !s.RuntimeReady {
		fatal(fmt.Errorf("%s", s.RuntimeNote))
	}
	report := map[string]any{"summary": s}
	if *transformerIndex != "" {
		inv, names, err := inventoryFromIndex(*transformerIndex, shape.NumLayers)
		if err != nil {
			fatal(err)
		}
		coverage, err := ideogram4.ValidateLinearCoverage(shape, names)
		if err != nil {
			fatal(err)
		}
		report["transformer_inventory"] = inv
		report["transformer_linear_coverage"] = coverage
	}
	if *uncondIndex != "" {
		inv, names, err := inventoryFromIndex(*uncondIndex, shape.NumLayers)
		if err != nil {
			fatal(err)
		}
		coverage, err := ideogram4.ValidateLinearCoverage(shape, names)
		if err != nil {
			fatal(err)
		}
		report["unconditional_transformer_inventory"] = inv
		report["unconditional_transformer_linear_coverage"] = coverage
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fatal(err)
		}
		return
	}
	fmt.Printf("Ideogram4 model: %s\n", *modelDir)
	fmt.Printf("  pipeline:    %s\n", s.Pipeline)
	fmt.Printf("  transformer: %s emb=%d layers=%d heads=%d head_dim=%d mlp=%d adaln=%d in_channels=%d llm_features=%d mrope=%v\n", s.Transformer, shape.EmbDim, shape.NumLayers, shape.NumHeads, shape.HeadDim, shape.IntermediateSize, shape.AdaLNDim, shape.InChannels, shape.LLMFeaturesDim, shape.MRoPESection)
	fmt.Printf("  uncond:      %s\n", s.UnconditionalTransformer)
	fmt.Printf("  text:        %s tokenizer=%s hidden=%d layers=%d vocab=%d activations=%v\n", s.TextEncoder, s.Tokenizer, s.TextHidden, s.TextLayers, s.VocabSize, s.ActivationLayers)
	fmt.Printf("  vae/sched:   %s / %s\n", s.VAE, s.Scheduler)
	if inv, ok := report["transformer_inventory"].(ideogram4.TensorInventory); ok {
		fmt.Printf("  tensors:     transformer total=%d layers=%d fp8_weights=%d fp8_scales=%d missing_globals=%d missing_layers=%d\n", inv.Total, inv.LayerCount, inv.FP8Weights, inv.FP8Scales, len(inv.MissingGlobals), len(inv.MissingLayerTensors))
	}
	if cov, ok := report["transformer_linear_coverage"].(ideogram4.LinearCoverage); ok {
		fmt.Printf("  fp8 linear:  transformer required=%d present=%d scaled=%d missing=%d missing_scales=%d\n", cov.Required, cov.Present, cov.Scaled, len(cov.Missing), len(cov.MissingScales))
	}
	if inv, ok := report["unconditional_transformer_inventory"].(ideogram4.TensorInventory); ok {
		fmt.Printf("  tensors:     uncond total=%d layers=%d fp8_weights=%d fp8_scales=%d missing_globals=%d missing_layers=%d\n", inv.Total, inv.LayerCount, inv.FP8Weights, inv.FP8Scales, len(inv.MissingGlobals), len(inv.MissingLayerTensors))
	}
	if cov, ok := report["unconditional_transformer_linear_coverage"].(ideogram4.LinearCoverage); ok {
		fmt.Printf("  fp8 linear:  uncond required=%d present=%d scaled=%d missing=%d missing_scales=%d\n", cov.Required, cov.Present, cov.Scaled, len(cov.Missing), len(cov.MissingScales))
	}
	fmt.Printf("  runtime:     ready=%v note=%s\n", s.RuntimeReady, s.RuntimeNote)
}

func inventoryFromIndex(path string, layers int) (ideogram4.TensorInventory, []string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ideogram4.TensorInventory{}, nil, err
	}
	var raw struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return ideogram4.TensorInventory{}, nil, err
	}
	names := make([]string, 0, len(raw.WeightMap))
	for name := range raw.WeightMap {
		names = append(names, name)
	}
	sort.Strings(names)
	return ideogram4.SummarizeTensorNames(names, layers), names, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ideogram4inspect:", err)
	os.Exit(1)
}
