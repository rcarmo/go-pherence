package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/hunyuan3d"
)

type tensorSummary struct {
	Path        string                                         `json:"path"`
	Total       int                                            `json:"total"`
	Model       int                                            `json:"model"`
	VAE         int                                            `json:"vae"`
	Conditioner int                                            `json:"conditioner"`
	Other       int                                            `json:"other"`
	Examples    map[loaderconfig.Hunyuan3DTensorGroup][]string `json:"examples,omitempty"`
}

type report struct {
	ConfigPath string                `json:"config_path"`
	Shape      hunyuan3d.ShapeConfig `json:"shape"`
	Tensors    []tensorSummary       `json:"tensors,omitempty"`
}

func main() {
	configPath := flag.String("config", "", "Hunyuan3D config.yaml path")
	tensorPath := flag.String("safetensors", "", "optional safetensors checkpoint path; repeat by comma-separated list")
	asJSON := flag.Bool("json", false, "emit JSON report")
	flag.Parse()
	if *configPath == "" {
		fmt.Fprintln(os.Stderr, "usage: hy3dinspect -config config.yaml [-safetensors model.safetensors[,extra.safetensors]] [-json]")
		os.Exit(2)
	}
	cfg, _, err := loaderconfig.ReadHunyuan3DConfig(*configPath)
	if err != nil {
		fatal(err)
	}
	shape, err := hunyuan3d.FromLoaderConfig(cfg)
	if err != nil {
		fatal(err)
	}
	out := report{ConfigPath: *configPath, Shape: shape}
	for _, path := range splitCSV(*tensorPath) {
		ts, err := summarizeTensorFile(path)
		if err != nil {
			fatal(err)
		}
		out.Tensors = append(out.Tensors, ts)
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

func splitCSV(s string) []string {
	if s == "" {
		return nil
	}
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ',' {
			if start < i {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	return out
}

func summarizeTensorFile(path string) (tensorSummary, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return tensorSummary{}, err
	}
	defer f.Close()
	names := make([]string, 0, len(f.Tensors))
	for name := range f.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	inv := loaderconfig.SummarizeHunyuan3DTensors(names)
	return tensorSummary{Path: path, Total: inv.Total, Model: inv.Model, VAE: inv.VAE, Conditioner: inv.Conditioner, Other: inv.Other, Examples: inv.Examples}, nil
}

func printText(r report) {
	s := r.Shape
	fmt.Printf("Hunyuan3D config: %s\n", r.ConfigPath)
	fmt.Printf("  denoiser: %s hidden=%d heads=%d head_dim=%d depth=%d single_blocks=%d in_channels=%d context=%d guidance=%v\n", s.DenoiserTarget, s.HiddenSize, s.NumHeads, s.HeadDim, s.Depth, s.DepthSingleBlocks, s.InChannels, s.ContextInDim, s.GuidanceEmbed)
	fmt.Printf("  vae:      %s latents=%d embed_dim=%d width=%d heads=%d head_dim=%d\n", s.VAETarget, s.VAELatents, s.VAEEmbedDim, s.VAEWidth, s.VAEHeads, s.VAEHeadDim)
	fmt.Printf("  cond:     %s type=%s\n", s.ConditionerTarget, s.ConditionerType)
	fmt.Printf("  sched:    %s train_steps=%d\n", s.SchedulerTarget, s.SchedulerSteps)
	for _, t := range r.Tensors {
		fmt.Printf("  tensors:  %s total=%d model=%d vae=%d conditioner=%d other=%d\n", t.Path, t.Total, t.Model, t.VAE, t.Conditioner, t.Other)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "hy3dinspect:", err)
	os.Exit(1)
}
