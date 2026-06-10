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
	ForwardBufferPlan  diffusiongemma.ForwardBufferPlan   `json:"forward_buffer_plan"`
	ForwardOpPlan      diffusiongemma.ForwardOpPlan       `json:"forward_op_plan"`
	Capabilities       diffusiongemma.RuntimeCapabilities `json:"capabilities"`
	OperationStatus    []diffusiongemma.OpStatus          `json:"operation_status,omitempty"`
	Processor          *diffusiongemma.ProcessorMetadata  `json:"processor,omitempty"`
	Tokenizer          *diffusiongemma.TokenizerMetadata  `json:"tokenizer,omitempty"`
	SpecialTokenIDs    *diffusiongemma.SpecialTokenIDs    `json:"special_token_ids,omitempty"`
	Shards             *diffusiongemma.ShardAvailability  `json:"shards,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "DiffusionGemma Hugging Face model directory")
	asJSON := flag.Bool("json", false, "emit JSON report")
	requireRuntime := flag.Bool("require-runtime-ready", false, "fail unless native DiffusionGemma runtime is reference-complete")
	requireTextScaffold := flag.Bool("require-text-scaffold-ready", false, "fail unless the current text-only scaffold/inventory is ready")
	requireShards := flag.Bool("require-shards-ready", false, "fail unless all safetensor shards from the index are present locally")
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
	if *requireTextScaffold {
		caps := diffusiongemma.Capabilities()
		if !caps.TextOnlyScaffoldReady || m.Readiness == nil || !m.Readiness.TextReady || m.TextTensorPlan == nil || !m.TextTensorPlan.Ready {
			fatal(fmt.Errorf("DiffusionGemma text scaffold is not ready"))
		}
	}
	if *requireShards {
		if m.Shards == nil || !m.Shards.Ready {
			present, expected := 0, 0
			var missing []string
			if m.Shards != nil {
				present = m.Shards.PresentShards
				expected = m.Shards.ExpectedShards
				missing = m.Shards.MissingShards
				if len(missing) > 5 {
					missing = missing[:5]
				}
			}
			fatal(fmt.Errorf("DiffusionGemma shards are not ready: present=%d/%d missing=%v", present, expected, missing))
		}
	}
	if *requireRuntime {
		if err := m.RequireRuntimeReady(); err != nil {
			fatal(err)
		}
	}
	out := report{ModelPath: *modelDir, Shape: shape, GenerationDefaults: m.GenerationDefaults, Tensors: m.Tensors, Readiness: m.Readiness, TextTensorPlan: m.TextTensorPlan, ForwardBufferPlan: diffusiongemma.BuildForwardBufferPlan(shape), ForwardOpPlan: diffusiongemma.BuildForwardOpPlan(shape, nil), Capabilities: diffusiongemma.Capabilities(), OperationStatus: diffusiongemma.OperationStatuses(), Processor: m.Processor, Tokenizer: m.Tokenizer, Shards: m.Shards}
	if m.Tokenizer != nil {
		specials := m.Tokenizer.SpecialTokenIDs(m.Processor)
		out.SpecialTokenIDs = &specials
	}
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
	if r.Processor != nil {
		fmt.Printf("  processor: tokenizer=%s processor=%s mask=%q image=%q think=%q chat_template_bytes=%d\n", r.Processor.TokenizerClass, r.Processor.ProcessorClass, r.Processor.Mask, r.Processor.Image, r.Processor.Think, r.Processor.ChatTemplateBytes)
		if r.Processor.ChatTemplate != nil {
			ct := r.Processor.ChatTemplate
			fmt.Printf("  template:  system=%v user=%v assistant=%v tools=%v thinking=%v markers=%v\n", ct.HasSystemRole, ct.HasUserRole, ct.HasAssistantRole, ct.HasToolSupport, ct.HasThinkingToken, ct.Markers)
		}
	}
	if r.Tokenizer != nil {
		fmt.Printf("  tokenizer: vocab=%d added=%d ids=%v\n", r.Tokenizer.VocabSize, r.Tokenizer.AddedTokens, r.Tokenizer.TokenIDs)
	}
	if r.SpecialTokenIDs != nil {
		s := r.SpecialTokenIDs
		fmt.Printf("  specials:  bos=%d eos=%d pad=%d mask=%d think=%d boi=%d eoi=%d image=%d bot=%d eot=%d\n", s.BOS, s.EOS, s.PAD, s.MASK, s.THINK, s.BOI, s.EOI, s.IMAGE, s.BOT, s.EOT)
	}
	if r.Tensors != nil {
		fmt.Printf("  tensors:   total=%d shards=%d parameters=%d size_bytes=%d size_gib=%.2f groups=%v\n", r.Tensors.Total, r.Tensors.Shards, r.Tensors.TotalParameters, r.Tensors.TotalSizeBytes, float64(r.Tensors.TotalSizeBytes)/(1024*1024*1024), r.Tensors.Groups)
	}
	if r.Shards != nil {
		fmt.Printf("  shards:    ready=%v present=%d/%d (%.1f%%) bytes=%d/%d (%.1f%%) missing=%d\n", r.Shards.Ready, r.Shards.PresentShards, r.Shards.ExpectedShards, r.Shards.PresentPercent, r.Shards.PresentBytes, r.Shards.ExpectedBytes, r.Shards.PresentBytePercent, len(r.Shards.MissingShards))
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
	fb := r.ForwardBufferPlan
	fmt.Printf("  buffers:   hidden=%d residual=%d logits=%d router=%d experts=%d top_k=%d\n", fb.Hidden, fb.Residual, fb.Logits, fb.Router, fb.Experts, fb.TopKExperts)
	fmt.Printf("  ops:       ready=%v prefix_ops=%d layer_ops=%d tail_ops=%d reason=%s\n", r.ForwardOpPlan.Ready, len(r.ForwardOpPlan.Prefix), len(r.ForwardOpPlan.Layers), len(r.ForwardOpPlan.Tail), r.ForwardOpPlan.Reason)
	if r.TextWeightsOpened {
		fmt.Printf("  weights:   text_shards_opened=true globals=%d layers=%d\n", r.TextWeightsGlobals, r.TextWeightsLayers)
	}
	if r.TextForwardPlan != nil {
		fmt.Printf("  forward:   text_ready=%v layers=%d missing=%d\n", r.TextForwardPlan.Ready, len(r.TextForwardPlan.Layers), len(r.TextForwardPlan.Missing))
	}
	fmt.Printf("  caps:      sampler=%v ops=%d/%d text_scaffold=%v attention_scaffold=%v rope=%v sliding_mask=%v encoder_kv=%v reference_complete=%v\n", r.Capabilities.Sampler, r.Capabilities.ImplementedOps, r.Capabilities.TotalOps, r.Capabilities.TextOnlyScaffoldReady, r.Capabilities.SelfAttentionScaffold, r.Capabilities.RoPE, r.Capabilities.SlidingWindowMask, r.Capabilities.EncoderKVConcat, r.Capabilities.ReferenceComplete)
	if len(r.Capabilities.MissingForReference) > 0 {
		fmt.Printf("  missing_reference: %v\n", r.Capabilities.MissingForReference)
	}
	if len(r.OperationStatus) > 0 {
		implemented, referenceComplete := 0, 0
		for _, op := range r.OperationStatus {
			if op.Implemented {
				implemented++
			}
			if op.ReferenceComplete {
				referenceComplete++
			}
		}
		fmt.Printf("  op_status: implemented=%d/%d reference_complete=%d/%d\n", implemented, len(r.OperationStatus), referenceComplete, len(r.OperationStatus))
	}
	if s.RuntimeNote != "" {
		fmt.Printf("  runtime:   %s\n", s.RuntimeNote)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "diffusiongemmainspect:", err)
	os.Exit(1)
}
