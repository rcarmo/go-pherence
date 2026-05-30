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
	ModelDir        string                      `json:"model_dir"`
	Config          lfm2.Config                 `json:"config"`
	ConvLayers      int                         `json:"conv_layers"`
	AttentionLayers int                         `json:"attention_layers"`
	TensorCoverage  *lfm2.TensorCoverage        `json:"tensor_coverage,omitempty"`
	TensorShapes    *lfm2.TensorShapeSummary    `json:"tensor_shapes,omitempty"`
	ShapeValidation *lfm2.TensorShapeValidation `json:"shape_validation,omitempty"`
	RuntimePlan     lfm2.RuntimePlan            `json:"runtime_plan"`
}

func main() {
	modelDir := flag.String("model", "", "LFM2 model directory containing config.json")
	safetensorPath := flag.String("safetensors", "", "optional safetensors path; defaults to model.safetensors or sharded index in -model")
	jsonOut := flag.Bool("json", false, "emit JSON report")
	strict := flag.Bool("strict", false, "exit non-zero when tensor readiness or shape validation fails")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: lfm2inspect -model <dir> [-safetensors path] [-json]")
		os.Exit(2)
	}
	cfg, err := lfm2.ReadModelDir(*modelDir)
	if err != nil {
		fatal(err)
	}
	plan, err := lfm2.NewRuntimePlan(cfg)
	if err != nil {
		fatal(err)
	}
	out := report{ModelDir: *modelDir, Config: cfg, ConvLayers: cfg.ConvLayerCount(), AttentionLayers: cfg.FullAttentionLayerCount(), RuntimePlan: plan}
	if infos, err := safetensorInfos(*modelDir, *safetensorPath); err == nil {
		names := make([]string, 0, len(infos))
		for name := range infos {
			names = append(names, name)
		}
		cov := lfm2.InspectTensorNames(names)
		out.TensorCoverage = &cov
		shapes := lfm2.InspectTensorShapes(infos)
		out.TensorShapes = &shapes
		shapeValidation := lfm2.ValidateTensorShapes(cfg, infos)
		out.ShapeValidation = &shapeValidation
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}
	} else {
		printText(out)
	}
	if *strict && !reportValid(out) {
		os.Exit(1)
	}
}

func safetensorInfos(modelDir, explicit string) (map[string]safetensors.TensorInfo, error) {
	if explicit != "" {
		f, err := safetensors.Open(explicit)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return f.TensorInfos(), nil
	}
	if sf, err := safetensors.OpenSharded(filepath.Join(modelDir, "model.safetensors.index.json")); err == nil {
		defer sf.Close()
		return sf.TensorInfos(), nil
	}
	f, err := safetensors.Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.TensorInfos(), nil
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
	fmt.Printf("  arch=%v dtype=%s hidden=%d layers=%d conv=%d full_attention=%d heads=%d kv_heads=%d head_dim=%d max_pos=%d vocab=%d tied_embeddings=%v\n", c.Architectures, c.DType, c.HiddenSize, c.NumHiddenLayers, r.ConvLayers, r.AttentionLayers, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, c.MaxPositionEmbeddings, r.RuntimePlan.ContextLayout.VocabSize, r.RuntimePlan.EmbeddingLayout.OutputSharesInput)
	fmt.Printf("  moe: experts=%d active=%d intermediate=%d dense_layers=%d routed_layers=%d expert_params=%d norm_topk=%v expert_bias=%v routed_scale=%g\n", c.NumExperts, c.NumExpertsPerTok, c.MoEIntermediateSize, c.NumDenseLayers, r.RuntimePlan.Routing.MoELayers, r.RuntimePlan.FFNLayout.ExpertParamsPerExpert, c.NormTopKProb, c.UseExpertBias, c.RoutedScalingFactor)
	fmt.Printf("  conv: L_cache=%d bias=%v rope_theta=%g rope_layers=%d norm_eps=%g state_floats/layer=%d\n", c.ConvLCache, c.ConvBias, c.RoPE.Theta, r.RuntimePlan.RoPELayout.FullAttentionLayers, r.RuntimePlan.NormLayout.Epsilon, r.RuntimePlan.ConvStateLayout.FloatsPerLayer)
	fmt.Printf("  runtime plan: conv_state_floats=%d kv_floats/token=%d attention_layers=%v attention_kv_floats/token=%d dense_layers=%v moe_layers=%d embedding_floats=%d lm_head_floats=%d\n", r.RuntimePlan.ConvStateFloats, r.RuntimePlan.KVFloatsPerToken, r.RuntimePlan.Schedule.FullAttentionIndices, r.RuntimePlan.AttentionKVLayout.FloatsPerToken, r.RuntimePlan.Execution.DenseIndices, len(r.RuntimePlan.Execution.MoEIndices), r.RuntimePlan.EmbeddingLayout.EmbeddingFloats, r.RuntimePlan.EmbeddingLayout.LMHeadFloats)
	if r.TensorCoverage != nil {
		t := r.TensorCoverage
		shapeValid := true
		if r.ShapeValidation != nil {
			shapeValid = r.ShapeValidation.Valid
		}
		fmt.Printf("  tensors: total=%d embeddings=%d layers=%d router=%d experts=%d lm_head=%d other=%d ready=%v missing=%v shapes_valid=%v\n", t.Total, t.Embedding, t.Layers, t.Router, t.Experts, t.LMHead, t.Other, t.Readiness.Ready, t.Readiness.MissingRequired, shapeValid)
	}
}

func reportValid(r report) bool {
	if r.TensorCoverage != nil && !r.TensorCoverage.Readiness.Ready {
		return false
	}
	if r.ShapeValidation != nil && !r.ShapeValidation.Valid {
		return false
	}
	return true
}

func fatal(err error) { fmt.Fprintln(os.Stderr, "lfm2inspect:", err); os.Exit(1) }
