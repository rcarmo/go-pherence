package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/model/minicpmv"
)

type report struct {
	ModelPath            string                              `json:"model_path"`
	Summary              config.MiniCPMVSummary              `json:"summary"`
	Processor            *config.MiniCPMVProcessorConfig     `json:"processor,omitempty"`
	Tokenizer            *config.MiniCPMVTokenizerMetadata   `json:"tokenizer,omitempty"`
	Generation           *config.MiniCPMVGenerationConfig    `json:"generation,omitempty"`
	SpecialTokenIDs      *minicpmv.SpecialTokenIDs           `json:"special_token_ids,omitempty"`
	AudioSpecialTokenIDs *minicpmv.AudioSpecialTokenIDs      `json:"audio_special_token_ids,omitempty"`
	Tensors              *minicpmv.TensorInventory           `json:"tensors,omitempty"`
	TensorInfoSummary    *minicpmv.TensorInfoSummary         `json:"tensor_info_summary,omitempty"`
	Readiness            *minicpmv.TensorReadiness           `json:"tensor_readiness,omitempty"`
	ShapeValidation      minicpmv.TensorShapeValidation      `json:"shape_validation,omitempty"`
	Capabilities         minicpmv.Capabilities               `json:"capabilities"`
	PromptText           *minicpmv.PromptText                `json:"prompt_text,omitempty"`
	AudioPromptText      *minicpmv.AudioPromptText           `json:"audio_prompt_text,omitempty"`
	MultiModalPrompt     *minicpmv.MultiModalPromptPlan      `json:"multimodal_prompt,omitempty"`
	ImagePreprocess      *minicpmv.ImageFilePreprocessResult `json:"image_preprocess,omitempty"`
	TextPlan             minicpmv.TextExecutionPlan          `json:"text_plan"`
	VisionPlan           minicpmv.VisionExecutionPlan        `json:"vision_plan"`
	AudioPlan            minicpmv.AudioExecutionPlan         `json:"audio_plan"`
	AudioFeaturePlan     minicpmv.AudioFeaturePlan           `json:"audio_feature_plan"`
	SlicePlan            minicpmv.SlicePlan                  `json:"slice_plan"`
	ResamplerPlan        *minicpmv.ResamplerTensorPlan       `json:"resampler_plan,omitempty"`
	RuntimePlan          minicpmv.RuntimePlan                `json:"runtime_plan"`
	ReadinessReport      minicpmv.ReadinessReport            `json:"readiness_report"`
	Config               config.MiniCPMVConfig               `json:"config,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "MiniCPM-V/O Hugging Face model directory")
	asJSON := flag.Bool("json", false, "emit JSON report")
	showConfig := flag.Bool("show-config", false, "include decoded config in JSON output")
	capabilitiesOnly := flag.Bool("capabilities", false, "print MiniCPM-V/O implementation capability summary without requiring -model")
	showVersion := flag.Bool("version", false, "print MiniCPM-V/O support version/status without requiring -model")
	pendingSteps := flag.Bool("pending-runtime-steps", false, "print pending MiniCPM-V/O runtime implementation steps without requiring -model")
	fixturePath := flag.Bool("fixture-path", false, "print committed MiniCPM-O metadata fixture path without requiring -model")
	fixtureSummary := flag.Bool("fixture-summary", false, "print committed MiniCPM-O expected summary without requiring -model")
	requireFixture := flag.Bool("require-fixture-ready", false, "validate committed MiniCPM-O fixture metadata against expected summary without requiring -model")
	requireCapabilities := flag.Bool("require-capabilities-ready", false, "with -capabilities, fail unless scaffold capabilities are present and runtime capabilities remain pending")
	safetensorsPath := flag.String("safetensors", "", "optional safetensors file path; defaults to model.safetensors or sharded index in -model")
	imagePath := flag.String("image", "", "optional image path to decode/preprocess using MiniCPM-V/O metadata")
	promptText := flag.String("prompt", "Describe the image.", "prompt text used for the MiniCPM-V/O image-placeholder preview")
	promptImages := flag.Int("images", 1, "number of image placeholders to include in the prompt preview")
	audioDurationMS := flag.Int("audio-duration-ms", 0, "optional audio duration in milliseconds for MiniCPM-O feature-frame estimate")
	strict := flag.Bool("strict", false, "fail unless metadata, tensor inventory, and safetensor shapes are ready; does not require full runtime execution")
	requireConfig := flag.Bool("require-config-ready", false, "fail unless MiniCPM-V config and prompt-planning metadata are valid")
	requireMetadata := flag.Bool("require-metadata-ready", false, "fail unless config, processor, tokenizer, special-token, image-preprocess, and prompt-planning metadata are ready")
	requireTensors := flag.Bool("require-tensors-ready", false, "fail unless local safetensor inventory has text, vision, and resampler metadata")
	requireShapes := flag.Bool("require-shapes-ready", false, "fail unless local safetensor header shapes match normalized MiniCPM-V/O config dimensions")
	requireRuntime := flag.Bool("require-runtime-ready", false, "fail unless full MiniCPM-V/O runtime tensor execution is ready")
	flag.Parse()
	if *showVersion {
		if *asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(map[string]string{"support_version": minicpmv.SupportVersion, "runtime_status": minicpmv.RuntimeStatusPending}); err != nil {
				fatal(err)
			}
			return
		}
		fmt.Printf("%s %s\n", minicpmv.SupportVersion, minicpmv.RuntimeStatusPending)
		return
	}
	if *requireFixture {
		meta, err := minicpmv.LoadMiniCPMOFixtureMetadata()
		if err != nil {
			fatal(err)
		}
		expected, err := minicpmv.LoadMiniCPMOFixtureExpectedSummary()
		if err != nil {
			fatal(err)
		}
		if meta.Summary.ModelType != expected.ModelType || meta.Summary.AudioModelType != expected.AudioModelType || meta.SpecialTokenIDs == nil || meta.SpecialTokenIDs.ImagePatch != expected.ImagePatchTokenID || meta.AudioSpecialTokenIDs == nil || meta.AudioSpecialTokenIDs.AudioPatch != expected.AudioPatchTokenID || meta.ReadinessReport.MetadataReady != expected.MetadataReady || meta.ReadinessReport.RuntimeReady != expected.RuntimeReady {
			fatal(fmt.Errorf("MiniCPM-O fixture is not ready: summary=%+v expected=%+v readiness=%+v", meta.Summary, expected, meta.ReadinessReport))
		}
		if *asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(map[string]any{"fixture_ready": true, "expected": expected}); err != nil {
				fatal(err)
			}
			return
		}
		fmt.Println("MiniCPM-O fixture ready")
		return
	}
	if *pendingSteps {
		steps := minicpmv.PendingRuntimeSteps()
		if *asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(map[string][]string{"pending_runtime_steps": steps}); err != nil {
				fatal(err)
			}
			return
		}
		for _, step := range steps {
			fmt.Println(step)
		}
		return
	}
	if *fixtureSummary {
		summary, err := minicpmv.LoadMiniCPMOFixtureExpectedSummary()
		if err != nil {
			fatal(err)
		}
		if *asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(summary); err != nil {
				fatal(err)
			}
			return
		}
		fmt.Printf("MiniCPM-O fixture: model=%s arch=%s text=%s vision=%s audio=%s num_query=%d runtime_ready=%v\n", summary.ModelType, summary.Architecture, summary.TextModelType, summary.VisionModelType, summary.AudioModelType, summary.NumQuery, summary.RuntimeReady)
		return
	}
	if *fixturePath {
		if *asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(map[string]string{"minicpmo_fixture": minicpmv.MiniCPMOFixturePath}); err != nil {
				fatal(err)
			}
			return
		}
		fmt.Println(minicpmv.MiniCPMOFixturePath)
		return
	}
	if *capabilitiesOnly {
		caps := minicpmv.CurrentCapabilities()
		if *requireCapabilities && (!caps.ConfigParsing || !caps.ProcessorMetadata || !caps.TokenizerMetadata || !caps.MultimodalPromptPlanning || !caps.TensorShapeValidation || !caps.ValidationGate || caps.EndToEndGeneration || caps.TextRuntimeGeneration || caps.VisionTowerRuntime || caps.ResamplerRuntime || caps.AudioEncoderRuntime) {
			fatal(fmt.Errorf("MiniCPM-V/O capabilities are inconsistent: %+v", caps))
		}
		if *asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(caps); err != nil {
				fatal(err)
			}
			return
		}
		printCapabilities(caps)
		return
	}
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: minicpmvinspect -model PATH [-json] [-require-config-ready] or minicpmvinspect -capabilities [-json]")
		os.Exit(2)
	}
	meta, err := minicpmv.LoadMetadataWithOptions(*modelDir, minicpmv.MetadataOptions{SafetensorsPath: *safetensorsPath})
	if err != nil {
		fatal(err)
	}
	if *strict {
		*requireMetadata = true
		*requireTensors = true
		*requireShapes = true
	}
	if *requireConfig && meta.Summary.NumQuery <= 0 {
		fatal(fmt.Errorf("MiniCPM-V config metadata is not ready"))
	}
	if *requireMetadata && (!meta.RuntimePlan.ConfigReady || !meta.RuntimePlan.ProcessorReady || !meta.RuntimePlan.TokenizerReady || !meta.RuntimePlan.SpecialTokensReady || !meta.RuntimePlan.ImagePreprocessReady || !meta.RuntimePlan.PromptPlanningReady) {
		fatal(fmt.Errorf("MiniCPM-V/O metadata is not ready: config=%v processor=%v tokenizer=%v specials=%v preprocess=%v prompt=%v", meta.RuntimePlan.ConfigReady, meta.RuntimePlan.ProcessorReady, meta.RuntimePlan.TokenizerReady, meta.RuntimePlan.SpecialTokensReady, meta.RuntimePlan.ImagePreprocessReady, meta.RuntimePlan.PromptPlanningReady))
	}
	if *requireTensors && (meta.TensorReadiness == nil || !meta.TensorReadiness.MetadataReady) {
		fatal(fmt.Errorf("MiniCPM-V/O tensor metadata is not ready"))
	}
	if *requireShapes && (!meta.ShapeValidation.Valid || len(meta.ShapeValidation.Issues) > 0) {
		fatal(fmt.Errorf("MiniCPM-V/O safetensor shape validation failed: valid=%v issues=%d", meta.ShapeValidation.Valid, len(meta.ShapeValidation.Issues)))
	}
	if *requireRuntime {
		if err := meta.RequireRuntimeReady(); err != nil {
			fatal(err)
		}
	}
	if *audioDurationMS > 0 {
		if plan, err := minicpmv.BuildAudioFeaturePlan(meta.Summary, *audioDurationMS); err == nil {
			meta.AudioFeaturePlan = plan
		}
	}
	out := report{
		ModelPath:            *modelDir,
		Summary:              meta.Summary,
		Processor:            meta.Processor,
		Tokenizer:            meta.Tokenizer,
		Generation:           meta.Generation,
		SpecialTokenIDs:      meta.SpecialTokenIDs,
		AudioSpecialTokenIDs: meta.AudioSpecialTokenIDs,
		Tensors:              meta.Tensors,
		TensorInfoSummary:    meta.TensorInfoSummary,
		Readiness:            meta.TensorReadiness,
		ShapeValidation:      meta.ShapeValidation,
		Capabilities:         meta.Capabilities,
		TextPlan:             meta.TextPlan,
		VisionPlan:           meta.VisionPlan,
		AudioPlan:            meta.AudioPlan,
		AudioFeaturePlan:     meta.AudioFeaturePlan,
		SlicePlan:            meta.SlicePlan,
		ResamplerPlan:        meta.ResamplerPlan,
		RuntimePlan:          meta.RuntimePlan,
		ReadinessReport:      meta.ReadinessReport,
	}
	if prompt, err := minicpmv.BuildPromptText(*promptText, *promptImages, meta.Summary, meta.Tokenizer, minicpmv.PromptTextOptions{}); err == nil {
		out.PromptText = &prompt
	}
	if meta.Summary.AudioModelType != "" || meta.AudioSpecialTokenIDs != nil {
		if prompt, err := minicpmv.BuildAudioPromptText(*promptText, 1, meta.Summary.NumQuery, meta.Tokenizer, minicpmv.PromptTextOptions{}); err == nil {
			out.AudioPromptText = &prompt
		}
		if prompt, err := minicpmv.BuildMultiModalPromptPlan(meta.Summary, meta.Tokenizer, minicpmv.MultiModalPromptOptions{Question: *promptText, Images: *promptImages, Audios: 1}); err == nil {
			out.MultiModalPrompt = &prompt
		}
	}
	if *imagePath != "" {
		cfg := minicpmv.DefaultImagePreprocessConfig(meta.Summary.ImageSize, meta.Summary.PatchSize)
		if meta.Processor != nil {
			cfg = minicpmv.ImagePreprocessConfigFromProcessor(*meta.Processor, meta.Summary.ImageSize, meta.Summary.PatchSize)
		}
		res, err := minicpmv.PreprocessImageFile(*imagePath, cfg)
		if err != nil {
			fatal(err)
		}
		out.ImagePreprocess = &res
		out.SlicePlan = minicpmv.BuildSlicePlan(meta.Summary, res.Result.Width, res.Result.Height)
	}
	if *showConfig {
		out.Config = meta.Config
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
	if s.AudioModelType != "" || s.AudioHiddenSize > 0 || s.AudioMelBins > 0 {
		fmt.Printf("  audio:     type=%s hidden=%d layers=%d heads=%d feature=%d mel_bins=%d sampling_rate=%d\n", s.AudioModelType, s.AudioHiddenSize, s.AudioLayers, s.AudioHeads, s.AudioFeatureSize, s.AudioMelBins, s.AudioSamplingRate)
	}
	fmt.Printf("  resampler: num_query=%d grid=%d heads=%d\n", s.NumQuery, s.ResamplerGrid, s.ResamplerHeads)
	fmt.Printf("  tokens:    image=%d start=%d end=%d use_start_end=%v\n", s.ImageTokenID, s.ImageStartTokenID, s.ImageEndTokenID, s.UseImageStartEnd)
	if r.Processor != nil {
		p := r.Processor
		fmt.Printf("  processor: class=%s image=%s size=%d resize=%v rescale=%v normalize=%v patch=%d image_seq=%d\n", p.ProcessorClass, p.ImageProcessorType, p.NormalizedSize, p.DoResize, p.DoRescale, p.DoNormalize, p.PatchSize, p.ImageSeqLength)
	}
	if r.Tokenizer != nil {
		t := r.Tokenizer
		fmt.Printf("  tokenizer: class=%s bos=%q eos=%q pad=%q image=%q audio=%q ids=%v template_bytes=%d\n", t.TokenizerClass, t.BOS, t.EOS, t.Pad, t.Image, t.Audio, t.TokenIDs, t.ChatTemplateBytes)
		if t.ChatTemplate != nil {
			ct := t.ChatTemplate
			fmt.Printf("  template:  system=%v user=%v assistant=%v image=%v tools=%v markers=%v\n", ct.HasSystemRole, ct.HasUserRole, ct.HasAssistantRole, ct.HasImageMarker, ct.HasToolSupport, ct.Markers)
		}
	}
	if r.SpecialTokenIDs != nil {
		ids := r.SpecialTokenIDs
		fmt.Printf("  specials:  image=%d patch=%d start=%d end=%d use_start_end=%v\n", ids.Image, ids.ImagePatch, ids.ImageStart, ids.ImageEnd, ids.UseStartEnd)
	}
	if r.AudioSpecialTokenIDs != nil {
		ids := r.AudioSpecialTokenIDs
		fmt.Printf("  audiospec: audio=%d patch=%d start=%d end=%d\n", ids.Audio, ids.AudioPatch, ids.AudioStart, ids.AudioEnd)
	}
	if r.Generation != nil {
		g := r.Generation
		fmt.Printf("  generate:  max_new=%d max_len=%d sample=%v temp=%.3g top_p=%.3g top_k=%d repeat=%.3g stops=%d\n", g.MaxNewTokens, g.MaxLength, g.DoSample, g.Temperature, g.TopP, g.TopK, g.RepetitionPenalty, len(g.StopStrings))
	}
	if r.Tensors != nil {
		fmt.Printf("  tensors:   total=%d groups=%v\n", r.Tensors.Total, r.Tensors.Groups)
	}
	if r.TensorInfoSummary != nil {
		sum := r.TensorInfoSummary
		fmt.Printf("  tensorinfo: dtypes=%v ranks=%v elements=%d bytes=%d\n", sum.DTypes, sum.Ranks, sum.TotalElements, sum.TotalBytes)
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
	if r.AudioPromptText != nil {
		fmt.Printf("  audioprompt: audios=%d patch_tokens=%d placeholder_bytes=%d\n", r.AudioPromptText.Audios, r.AudioPromptText.PatchTokens, len(r.AudioPromptText.AudioPlaceholder))
	}
	if r.MultiModalPrompt != nil {
		fmt.Printf("  multimodal: images=%d audios=%d text_bytes=%d\n", r.MultiModalPrompt.Images, r.MultiModalPrompt.Audios, len(r.MultiModalPrompt.Text))
	}
	if r.ImagePreprocess != nil {
		img := r.ImagePreprocess
		fmt.Printf("  image:     path=%s format=%s shape=%v patch_grid=%v patch_count=%d\n", img.Path, img.Format, img.Result.Shape, img.Result.PatchGrid, img.Result.PatchCount)
	}
	fmt.Printf("  textplan:  model=%s layers=%d embedding=%v layers_ready=%v lm_head=%v generation=%v ready=%v\n", r.TextPlan.TextModelType, r.TextPlan.Layers, r.TextPlan.HasEmbedding, r.TextPlan.HasLayers, r.TextPlan.HasLMHead, r.TextPlan.Generation, r.TextPlan.Ready)
	fmt.Printf("  visionplan: model=%s patch_grid=%d vision_tokens=%d resampler_query=%d needs_kv_proj=%v ready=%v\n", r.VisionPlan.VisionModelType, r.VisionPlan.PatchGrid, r.VisionPlan.VisionTokens, r.VisionPlan.ResamplerQuery, r.VisionPlan.NeedsKVProj, r.VisionPlan.Ready)
	if r.AudioPlan.MetadataReady || len(r.AudioPlan.Bindings) > 0 {
		fmt.Printf("  audioplan: model=%s mel_bins=%d sampling_rate=%d bindings=%d tensor_ready=%v ready=%v\n", r.AudioPlan.AudioModelType, r.AudioPlan.MelBins, r.AudioPlan.SamplingRate, len(r.AudioPlan.Bindings), r.AudioPlan.TensorReady, r.AudioPlan.Ready)
	}
	if r.AudioFeaturePlan.Ready {
		fmt.Printf("  audiofeat: sample_rate=%d mel_bins=%d feature=%d duration_ms=%d frames=%d\n", r.AudioFeaturePlan.SamplingRate, r.AudioFeaturePlan.MelBins, r.AudioFeaturePlan.FeatureSize, r.AudioFeaturePlan.DurationMS, r.AudioFeaturePlan.EstimatedFrames)
	}
	fmt.Printf("  sliceplan: enabled=%v scale=%d max=%d estimated=%d ready=%v\n", r.SlicePlan.Enabled, r.SlicePlan.ScaleResolution, r.SlicePlan.MaxSliceNums, r.SlicePlan.EstimatedSlices, r.SlicePlan.Ready)
	if r.ResamplerPlan != nil {
		fmt.Printf("  resampler: bindings=%d ready=%v missing=%v counts=%v\n", len(r.ResamplerPlan.Bindings), r.ResamplerPlan.Ready, r.ResamplerPlan.MissingRequired, r.ResamplerPlan.Counts)
	}
	fmt.Printf("  plan:      config=%v processor=%v tokenizer=%v specials=%v tensors=%v preprocess=%v prompt=%v runtime=%v\n", r.RuntimePlan.ConfigReady, r.RuntimePlan.ProcessorReady, r.RuntimePlan.TokenizerReady, r.RuntimePlan.SpecialTokensReady, r.RuntimePlan.TensorMetadataReady, r.RuntimePlan.ImagePreprocessReady, r.RuntimePlan.PromptPlanningReady, r.RuntimePlan.RuntimeReady)
	fmt.Printf("  caps:      metadata=%v prompts=%v tensors=%v runtime=%v end_to_end=%v\n", r.Capabilities.ConfigParsing && r.Capabilities.ProcessorMetadata && r.Capabilities.TokenizerMetadata, r.Capabilities.MultimodalPromptPlanning, r.Capabilities.TensorInventory && r.Capabilities.TensorShapeValidation, r.Capabilities.TextRuntimeGeneration || r.Capabilities.VisionTowerRuntime || r.Capabilities.ResamplerRuntime || r.Capabilities.AudioEncoderRuntime, r.Capabilities.EndToEndGeneration)
	fmt.Printf("  readiness: metadata=%v tensors=%v shapes=%v runtime=%v blockers=%d\n", r.ReadinessReport.MetadataReady, r.ReadinessReport.TensorReady, r.ReadinessReport.ShapesReady, r.ReadinessReport.RuntimeReady, len(r.ReadinessReport.Blockers))
	for i, blocker := range r.ReadinessReport.Blockers {
		if i >= 5 {
			fmt.Printf("  blocker:   ... %d more\n", len(r.ReadinessReport.Blockers)-i)
			break
		}
		fmt.Printf("  blocker:   %s\n", blocker)
	}
	fmt.Printf("  status:    %s\n", s.RuntimeNote)
}

func printCapabilities(c minicpmv.Capabilities) {
	fmt.Printf("MiniCPM-V/O capabilities:\n")
	fmt.Printf("  metadata:  config=%v processor=%v tokenizer=%v generation=%v template=%v\n", c.ConfigParsing, c.ProcessorMetadata, c.TokenizerMetadata, c.GenerationMetadata, c.ChatTemplateSummary)
	fmt.Printf("  prompts:   image=%v audio=%v multimodal=%v special_tokens=%v/%v\n", c.ImagePromptPlanning, c.AudioPromptPlanning, c.MultimodalPromptPlanning, c.ImageSpecialTokens, c.AudioSpecialTokens)
	fmt.Printf("  tensors:   inventory=%v shapes=%v explicit_safetensors=%v\n", c.TensorInventory, c.TensorShapeValidation, c.ExplicitSafetensorsPath)
	fmt.Printf("  plans:     text=%v vision=%v resampler=%v audio=%v readiness=%v\n", c.TextExecutionPlan, c.VisionExecutionPlan, c.ResamplerTensorPlan, c.AudioExecutionPlan, c.ReadinessReport)
	fmt.Printf("  status:    %s\n", c.RuntimeStatus)
	fmt.Printf("  runtime:   text=%v vision=%v resampler=%v audio=%v end_to_end=%v\n", c.TextRuntimeGeneration, c.VisionTowerRuntime, c.ResamplerRuntime, c.AudioEncoderRuntime, c.EndToEndGeneration)
	for i, step := range c.PendingRuntimeSteps {
		if i >= 5 {
			fmt.Printf("  pending:   ... %d more\n", len(c.PendingRuntimeSteps)-i)
			break
		}
		fmt.Printf("  pending:   %s\n", step)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
