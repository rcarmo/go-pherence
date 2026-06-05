package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/model/ideogram4"
)

func main() {
	modelDir := flag.String("model", "", "Ideogram 4 Diffusers model directory")
	asJSON := flag.Bool("json", false, "emit JSON report")
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
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(s); err != nil {
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
	fmt.Printf("  runtime:     ready=%v note=%s\n", s.RuntimeReady, s.RuntimeNote)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ideogram4inspect:", err)
	os.Exit(1)
}
