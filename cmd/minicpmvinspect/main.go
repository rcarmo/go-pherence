package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/minicpmv"
)

type report struct {
	ModelPath       string                              `json:"model_path"`
	Summary         config.MiniCPMVSummary              `json:"summary"`
	Processor       *config.MiniCPMVProcessorConfig     `json:"processor,omitempty"`
	Tokenizer       *config.MiniCPMVTokenizerMetadata   `json:"tokenizer,omitempty"`
	SpecialTokenIDs *minicpmv.SpecialTokenIDs           `json:"special_token_ids,omitempty"`
	Tensors         *minicpmv.TensorInventory           `json:"tensors,omitempty"`
	Readiness       *minicpmv.TensorReadiness           `json:"tensor_readiness,omitempty"`
	ShapeValidation minicpmv.TensorShapeValidation      `json:"shape_validation,omitempty"`
	PromptText      *minicpmv.PromptText                `json:"prompt_text,omitempty"`
	ImagePreprocess *minicpmv.ImageFilePreprocessResult `json:"image_preprocess,omitempty"`
	RuntimePlan     minicpmv.RuntimePlan                `json:"runtime_plan"`
	Config          config.MiniCPMVConfig               `json:"config,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "MiniCPM-V/O Hugging Face model directory")
	asJSON := flag.Bool("json", false, "emit JSON report")
	showConfig := flag.Bool("show-config", false, "include decoded config in JSON output")
	imagePath := flag.String("image", "", "optional image path to decode/preprocess using MiniCPM-V/O metadata")
	requireConfig := flag.Bool("require-config-ready", false, "fail unless MiniCPM-V config and prompt-planning metadata are valid")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: minicpmvinspect -model PATH [-json] [-require-config-ready]")
		os.Exit(2)
	}
	cfg, err := config.ReadMiniCPMVConfig(*modelDir)
	if err != nil {
		fatal(err)
	}
	summary := cfg.MiniCPMVSummary()
	if *requireConfig && summary.NumQuery <= 0 {
		fatal(fmt.Errorf("MiniCPM-V config metadata is not ready"))
	}
	out := report{ModelPath: *modelDir, Summary: summary}
	if proc, ok, err := config.ReadMiniCPMVProcessorConfig(*modelDir); err != nil {
		fatal(err)
	} else if ok {
		out.Processor = &proc
	}
	if tok, ok, err := config.ReadMiniCPMVTokenizerMetadata(*modelDir); err != nil {
		fatal(err)
	} else if ok {
		out.Tokenizer = &tok
		if ids, err := minicpmv.ResolveSpecialTokenIDs(summary, &tok); err == nil {
			out.SpecialTokenIDs = &ids
		}
	} else if ids, err := minicpmv.ResolveSpecialTokenIDs(summary, nil); err == nil {
		out.SpecialTokenIDs = &ids
	}
	if inv, ok, err := minicpmv.TensorInventoryFromModelDir(*modelDir); err != nil {
		fatal(err)
	} else if ok {
		out.Tensors = &inv
		readiness := minicpmv.TensorReadinessFromInventory(inv)
		out.Readiness = &readiness
		if infos, err := safetensors.TensorInfosFrom(*modelDir, ""); err == nil {
			out.ShapeValidation = minicpmv.ValidateTensorShapes(summary, infos)
		}
	}
	if prompt, err := minicpmv.BuildPromptText("Describe the image.", 1, summary, out.Tokenizer, minicpmv.PromptTextOptions{}); err == nil {
		out.PromptText = &prompt
	}
	if *imagePath != "" {
		cfg := minicpmv.DefaultImagePreprocessConfig(summary.ImageSize, summary.PatchSize)
		if out.Processor != nil {
			cfg = minicpmv.ImagePreprocessConfigFromProcessor(*out.Processor, summary.ImageSize, summary.PatchSize)
		}
		res, err := minicpmv.PreprocessImageFile(*imagePath, cfg)
		if err != nil {
			fatal(err)
		}
		out.ImagePreprocess = &res
	}
	out.RuntimePlan = minicpmv.BuildRuntimePlan(summary, out.Processor, out.Tokenizer, out.Tensors)
	if *showConfig {
		out.Config = cfg
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
	s := r.Summary
	fmt.Printf("MiniCPM-V/O model: %s\n", r.ModelPath)
	fmt.Printf("  arch:      %s model_type=%s runtime_ready=%v\n", s.Architecture, s.ModelType, s.RuntimeReady)
	fmt.Printf("  text:      type=%s hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d intermediate=%d vocab=%d\n", s.TextModelType, s.HiddenSize, s.Layers, s.Heads, s.KVHeads, s.HeadDim, s.IntermediateSize, s.VocabSize)
	fmt.Printf("  vision:    type=%s hidden=%d layers=%d heads=%d image=%d patch=%d slice_mode=%v\n", s.VisionModelType, s.VisionHiddenSize, s.VisionLayers, s.VisionHeads, s.ImageSize, s.PatchSize, s.SliceMode)
	fmt.Printf("  resampler: num_query=%d grid=%d heads=%d\n", s.NumQuery, s.ResamplerGrid, s.ResamplerHeads)
	fmt.Printf("  tokens:    image=%d start=%d end=%d use_start_end=%v\n", s.ImageTokenID, s.ImageStartTokenID, s.ImageEndTokenID, s.UseImageStartEnd)
	if r.Processor != nil {
		p := r.Processor
		fmt.Printf("  processor: class=%s image=%s size=%d resize=%v rescale=%v normalize=%v patch=%d image_seq=%d\n", p.ProcessorClass, p.ImageProcessorType, p.NormalizedSize, p.DoResize, p.DoRescale, p.DoNormalize, p.PatchSize, p.ImageSeqLength)
	}
	if r.Tokenizer != nil {
		t := r.Tokenizer
		fmt.Printf("  tokenizer: class=%s bos=%q eos=%q pad=%q image=%q ids=%v template_bytes=%d\n", t.TokenizerClass, t.BOS, t.EOS, t.Pad, t.Image, t.TokenIDs, t.ChatTemplateBytes)
	}
	if r.SpecialTokenIDs != nil {
		ids := r.SpecialTokenIDs
		fmt.Printf("  specials:  image=%d patch=%d start=%d end=%d use_start_end=%v\n", ids.Image, ids.ImagePatch, ids.ImageStart, ids.ImageEnd, ids.UseStartEnd)
	}
	if r.Tensors != nil {
		fmt.Printf("  tensors:   total=%d groups=%v\n", r.Tensors.Total, r.Tensors.Groups)
	}
	if r.Readiness != nil {
		fmt.Printf("  readiness: text=%v vision=%v resampler=%v metadata=%v runtime=%v\n", r.Readiness.HasTextEmbedding && r.Readiness.HasTextLayers, r.Readiness.HasVisionTower, r.Readiness.HasResampler, r.Readiness.MetadataReady, r.Readiness.RuntimeReady)
	}
	if len(r.ShapeValidation.Issues) > 0 || r.ShapeValidation.Valid {
		fmt.Printf("  shapes:    valid=%v issues=%d\n", r.ShapeValidation.Valid, len(r.ShapeValidation.Issues))
	}
	if r.PromptText != nil {
		fmt.Printf("  prompt:    images=%d patch_tokens=%d placeholder_bytes=%d\n", r.PromptText.Images, r.PromptText.PatchTokens, len(r.PromptText.ImagePlaceholder))
	}
	if r.ImagePreprocess != nil {
		img := r.ImagePreprocess
		fmt.Printf("  image:     path=%s format=%s shape=%v patch_grid=%v patch_count=%d\n", img.Path, img.Format, img.Result.Shape, img.Result.PatchGrid, img.Result.PatchCount)
	}
	fmt.Printf("  plan:      config=%v processor=%v tokenizer=%v specials=%v tensors=%v preprocess=%v prompt=%v runtime=%v\n", r.RuntimePlan.ConfigReady, r.RuntimePlan.ProcessorReady, r.RuntimePlan.TokenizerReady, r.RuntimePlan.SpecialTokensReady, r.RuntimePlan.TensorMetadataReady, r.RuntimePlan.ImagePreprocessReady, r.RuntimePlan.PromptPlanningReady, r.RuntimePlan.RuntimeReady)
	fmt.Printf("  status:    %s\n", s.RuntimeNote)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
