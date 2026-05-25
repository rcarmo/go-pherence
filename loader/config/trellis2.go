package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Trellis2Config struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type Trellis2Family string

const (
	Trellis2FamilyPipeline                    Trellis2Family = "pipeline"
	Trellis2FamilyTexturingPipeline           Trellis2Family = "texturing_pipeline"
	Trellis2FamilyShapeEncoder                Trellis2Family = "shape_encoder"
	Trellis2FamilyShapeDecoder                Trellis2Family = "shape_decoder"
	Trellis2FamilyTextureEncoder              Trellis2Family = "texture_encoder"
	Trellis2FamilyTextureDecoder              Trellis2Family = "texture_decoder"
	Trellis2FamilySparseStructureFlow         Trellis2Family = "sparse_structure_flow"
	Trellis2FamilyStructuredLatentShapeFlow   Trellis2Family = "structured_latent_shape_flow"
	Trellis2FamilyStructuredLatentTextureFlow Trellis2Family = "structured_latent_texture_flow"
	Trellis2FamilyUnknown                     Trellis2Family = "unknown"
)

type Trellis2Summary struct {
	Family              Trellis2Family `json:"family"`
	Name                string         `json:"name"`
	ModelKeys           []string       `json:"model_keys,omitempty"`
	DefaultPipelineType string         `json:"default_pipeline_type,omitempty"`
	Resolution          int            `json:"resolution,omitempty"`
	InChannels          int            `json:"in_channels,omitempty"`
	OutChannels         int            `json:"out_channels,omitempty"`
	HiddenSize          int            `json:"hidden_size,omitempty"`
	NumHeads            int            `json:"num_heads,omitempty"`
	Depth               int            `json:"depth,omitempty"`
	LatentChannels      int            `json:"latent_channels,omitempty"`
	ModelChannels       []int          `json:"model_channels,omitempty"`
	NumBlocks           []int          `json:"num_blocks,omitempty"`
	DType               string         `json:"dtype,omitempty"`
}

var Trellis2ShapePipelineModelKeys = []string{
	"sparse_structure_decoder",
	"sparse_structure_flow_model",
	"shape_slat_decoder",
	"shape_slat_flow_model_512",
	"shape_slat_flow_model_1024",
}

var Trellis2TexturePipelineModelKeys = []string{
	"tex_slat_decoder",
	"tex_slat_flow_model_512",
	"tex_slat_flow_model_1024",
}

var Trellis2TexturingPipelineModelKeys = []string{
	"shape_slat_encoder",
	"tex_slat_decoder",
	"tex_slat_flow_model_512",
	"tex_slat_flow_model_1024",
}

func ReadTrellis2Config(path string) (*Trellis2Config, Trellis2Family, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, Trellis2FamilyUnknown, err
	}
	var cfg Trellis2Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, Trellis2FamilyUnknown, err
	}
	family := Trellis2FamilyForPath(path)
	if family == Trellis2FamilyUnknown {
		family = Trellis2FamilyForName(cfg.Name)
	}
	if err := ValidateTrellis2Config(&cfg, family); err != nil {
		return nil, family, err
	}
	return &cfg, family, nil
}

func ReadTrellis2ConfigWithFamily(path string, family Trellis2Family) (*Trellis2Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Trellis2Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if err := ValidateTrellis2Config(&cfg, family); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func ValidateTrellis2Config(cfg *Trellis2Config, family Trellis2Family) error {
	if cfg == nil {
		return fmt.Errorf("trellis2 config: nil")
	}
	if cfg.Name == "" {
		return fmt.Errorf("trellis2 config: missing name")
	}
	if cfg.Args == nil {
		return fmt.Errorf("trellis2 config %q: missing args", cfg.Name)
	}
	if family == Trellis2FamilyPipeline || family == Trellis2FamilyTexturingPipeline {
		models, ok := cfg.Args["models"].(map[string]any)
		if !ok || len(models) == 0 {
			return fmt.Errorf("trellis2 %s %q: missing models map", family, cfg.Name)
		}
		return nil
	}
	if family == Trellis2FamilyUnknown {
		return nil
	}
	return nil
}

func ValidateTrellis2ShapePipeline(cfg *Trellis2Config, includeTexture bool) error {
	if err := ValidateTrellis2Config(cfg, Trellis2FamilyPipeline); err != nil {
		return err
	}
	required := append([]string(nil), Trellis2ShapePipelineModelKeys...)
	if includeTexture {
		required = append(required, Trellis2TexturePipelineModelKeys...)
	}
	return validateTrellis2ModelKeys(cfg, required)
}

func ValidateTrellis2TexturingPipeline(cfg *Trellis2Config) error {
	if err := ValidateTrellis2Config(cfg, Trellis2FamilyTexturingPipeline); err != nil {
		return err
	}
	return validateTrellis2ModelKeys(cfg, Trellis2TexturingPipelineModelKeys)
}

func SummarizeTrellis2Config(cfg *Trellis2Config, family Trellis2Family) Trellis2Summary {
	if cfg == nil {
		return Trellis2Summary{Family: family}
	}
	out := Trellis2Summary{Family: family, Name: cfg.Name}
	out.DefaultPipelineType = stringArg(cfg.Args, "default_pipeline_type")
	out.Resolution = intArg(cfg.Args, "resolution")
	out.InChannels = intArg(cfg.Args, "in_channels")
	out.OutChannels = intArg(cfg.Args, "out_channels")
	out.HiddenSize = intArg(cfg.Args, "hidden_size")
	out.NumHeads = intArg(cfg.Args, "num_heads")
	out.Depth = intArg(cfg.Args, "depth")
	out.LatentChannels = intArg(cfg.Args, "latent_channels")
	out.ModelChannels = intSliceArg(cfg.Args, "model_channels")
	out.NumBlocks = intSliceArg(cfg.Args, "num_blocks")
	out.DType = stringArg(cfg.Args, "dtype")
	if models, ok := cfg.Args["models"].(map[string]any); ok {
		out.ModelKeys = make([]string, 0, len(models))
		for k := range models {
			out.ModelKeys = append(out.ModelKeys, k)
		}
		sort.Strings(out.ModelKeys)
	}
	return out
}

func Trellis2FamilyForPath(path string) Trellis2Family {
	base := filepath.Base(path)
	switch {
	case base == "pipeline.json":
		return Trellis2FamilyPipeline
	case base == "texturing_pipeline.json":
		return Trellis2FamilyTexturingPipeline
	case strings.HasPrefix(base, "shape_enc_"):
		return Trellis2FamilyShapeEncoder
	case strings.HasPrefix(base, "shape_dec_"):
		return Trellis2FamilyShapeDecoder
	case strings.HasPrefix(base, "tex_enc_"):
		return Trellis2FamilyTextureEncoder
	case strings.HasPrefix(base, "tex_dec_"):
		return Trellis2FamilyTextureDecoder
	case strings.HasPrefix(base, "ss_flow_"):
		return Trellis2FamilySparseStructureFlow
	case strings.HasPrefix(base, "slat_flow_img2shape_"):
		return Trellis2FamilyStructuredLatentShapeFlow
	case strings.HasPrefix(base, "slat_flow_imgshape2tex_"):
		return Trellis2FamilyStructuredLatentTextureFlow
	default:
		return Trellis2FamilyUnknown
	}
}

func Trellis2FamilyForName(name string) Trellis2Family {
	switch name {
	case "Trellis2ImageTo3DPipeline":
		return Trellis2FamilyPipeline
	case "Trellis2TexturingPipeline":
		return Trellis2FamilyTexturingPipeline
	case "FlexiDualGridVaeEncoder":
		return Trellis2FamilyShapeEncoder
	case "FlexiDualGridVaeDecoder":
		return Trellis2FamilyShapeDecoder
	case "SparseUnetVaeEncoder":
		return Trellis2FamilyTextureEncoder
	case "SparseUnetVaeDecoder":
		return Trellis2FamilyTextureDecoder
	case "SparseStructureFlowModel":
		return Trellis2FamilySparseStructureFlow
	case "SLatFlowModel":
		return Trellis2FamilyStructuredLatentShapeFlow
	default:
		return Trellis2FamilyUnknown
	}
}

func validateTrellis2ModelKeys(cfg *Trellis2Config, required []string) error {
	models, ok := cfg.Args["models"].(map[string]any)
	if !ok {
		return fmt.Errorf("trellis2 pipeline %q: missing models map", cfg.Name)
	}
	var missing []string
	for _, key := range required {
		if _, ok := models[key]; !ok {
			missing = append(missing, key)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("trellis2 pipeline %q: missing model keys %s", cfg.Name, strings.Join(missing, ", "))
	}
	return nil
}

func stringArg(args map[string]any, key string) string {
	if v, ok := args[key].(string); ok {
		return v
	}
	return ""
}

func intArg(args map[string]any, key string) int {
	v, ok := args[key]
	if !ok {
		return 0
	}
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	default:
		return 0
	}
}

func intSliceArg(args map[string]any, key string) []int {
	raw, ok := args[key].([]any)
	if !ok {
		return nil
	}
	out := make([]int, 0, len(raw))
	for _, v := range raw {
		if f, ok := v.(float64); ok {
			out = append(out, int(f))
		}
	}
	return out
}
