package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/lfm2"
)

type report struct {
	ModelDir        string               `json:"model_dir"`
	Config          lfm2.Config          `json:"config"`
	ConvLayers      int                  `json:"conv_layers"`
	AttentionLayers int                  `json:"attention_layers"`
	TensorCoverage  *lfm2.TensorCoverage `json:"tensor_coverage,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "LFM2 model directory containing config.json")
	safetensorPath := flag.String("safetensors", "", "optional safetensors path; defaults to model.safetensors or sharded index in -model")
	jsonOut := flag.Bool("json", false, "emit JSON report")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: lfm2inspect -model <dir> [-safetensors path] [-json]")
		os.Exit(2)
	}
	cfg, err := lfm2.ReadModelDir(*modelDir)
	if err != nil {
		fatal(err)
	}
	out := report{ModelDir: *modelDir, Config: cfg, ConvLayers: cfg.ConvLayerCount(), AttentionLayers: cfg.FullAttentionLayerCount()}
	if names, err := safetensorNames(*modelDir, *safetensorPath); err == nil {
		cov := lfm2.InspectTensorNames(names)
		out.TensorCoverage = &cov
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}
		return
	}
	printText(out)
}

func safetensorNames(modelDir, explicit string) ([]string, error) {
	if explicit != "" {
		f, err := safetensors.Open(explicit)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return f.Names(), nil
	}
	if sf, err := safetensors.OpenSharded(filepath.Join(modelDir, "model.safetensors.index.json")); err == nil {
		defer sf.Close()
		return sf.Names(), nil
	}
	f, err := safetensors.Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.Names(), nil
}

func printText(r report) {
	c := r.Config
	fmt.Printf("LFM2: %s (%s)\n", c.ModelType, r.ModelDir)
	fmt.Printf("  arch=%v dtype=%s hidden=%d layers=%d conv=%d full_attention=%d heads=%d kv_heads=%d head_dim=%d max_pos=%d\n", c.Architectures, c.DType, c.HiddenSize, c.NumHiddenLayers, r.ConvLayers, r.AttentionLayers, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, c.MaxPositionEmbeddings)
	fmt.Printf("  moe: experts=%d active=%d intermediate=%d dense_layers=%d norm_topk=%v expert_bias=%v routed_scale=%g\n", c.NumExperts, c.NumExpertsPerTok, c.MoEIntermediateSize, c.NumDenseLayers, c.NormTopKProb, c.UseExpertBias, c.RoutedScalingFactor)
	fmt.Printf("  conv: L_cache=%d bias=%v rope_theta=%g\n", c.ConvLCache, c.ConvBias, c.RoPE.Theta)
	if r.TensorCoverage != nil {
		t := r.TensorCoverage
		fmt.Printf("  tensors: total=%d embeddings=%d layers=%d router=%d experts=%d lm_head=%d other=%d\n", t.Total, t.Embedding, t.Layers, t.Router, t.Experts, t.LMHead, t.Other)
	}
}

func fatal(err error) { fmt.Fprintln(os.Stderr, "lfm2inspect:", err); os.Exit(1) }
