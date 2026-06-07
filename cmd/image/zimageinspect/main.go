package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func main() {
	modelDir := flag.String("model", "", "Z-Image Diffusers model directory")
	asJSON := flag.Bool("json", false, "emit JSON report")
	requireRuntime := flag.Bool("require-runtime-ready", false, "fail unless image generation runtime is implemented")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: zimageinspect -model PATH [-json] [-require-runtime-ready]")
		os.Exit(2)
	}
	cfg, err := loaderconfig.ReadZImageConfig(*modelDir)
	if err != nil {
		fatal(err)
	}
	s := loaderconfig.SummarizeZImageConfig(cfg)
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
	fmt.Printf("Z-Image model: %s\n", *modelDir)
	fmt.Printf("  pipeline:    %s\n", s.Pipeline)
	fmt.Printf("  transformer: %s dim=%d layers=%d refiner=%d heads=%d kv_heads=%d in_channels=%d cap_feat=%d axes_dims=%v axes_lens=%v\n", s.Transformer, s.Dim, s.Layers, s.RefinerLayers, s.Heads, s.KVHeads, s.InChannels, s.CapFeatDim, s.AxesDims, s.AxesLens)
	fmt.Printf("  text:        %s tokenizer=%s hidden=%d layers=%d vocab=%d\n", s.TextEncoder, s.Tokenizer, s.TextHidden, s.TextLayers, s.VocabSize)
	fmt.Printf("  vae/sched:   %s / %s\n", s.VAE, s.Scheduler)
	fmt.Printf("  runtime:     ready=%v note=%s\n", s.RuntimeReady, s.RuntimeNote)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "zimageinspect:", err)
	os.Exit(1)
}
