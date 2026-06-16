package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/model"
)

func main() {
	dir := flag.String("model", "", "model directory")
	prompt := flag.String("prompt", "The meaning of life is", "input prompt")
	tokens := flag.Int("tokens", 50, "tokens to generate")
	useGPU := flag.Bool("gpu", false, "use GPU-resident forward pass")
	gpuLayers := flag.Int("gpu-layers", 0, "number of layers on GPU (0=all)")
	gpuKVMaxSeq := flag.Int("gpu-kv-max-seq", 0, "GPU KV cache sequence horizon (0=default 2048; lower values fit more layers for prompt smokes)")
	turboQuant := flag.Bool("turbo-quant", false, "enable TurboQuant KV cache compression on CPU backend")
	speculative := flag.Bool("speculative", false, "enable opt-in stock-weight speculative decoding path (CPU backend)")
	specBlock := flag.Int("speculative-block", 8, "speculative proposal block size")
	specNGram := flag.Int("speculative-ngram", 4, "speculative prompt-lookup n-gram size")
	specMinProposal := flag.Int("speculative-min-proposal", 2, "minimum proposal length before verifier attempt")
	specProposer := flag.String("speculative-proposer", "prompt", "speculative proposer: prompt, repeat-last, none")
	specBackend := flag.String("speculative-backend", "replay", "speculative verifier backend: replay")
	specDebug := flag.Bool("speculative-debug", false, "print speculative proposal/acceptance stats")
	eagerLoad := flag.Bool("eager-load", false, "pre-fault mmap'd model weights at startup")
	mtpDrafter := flag.String("mtp-drafter", "", "Gemma4 MTP assistant/drafter directory (experimental)")
	mtpSmoke := flag.Bool("mtp-smoke", false, "load -mtp-drafter and run one packed 4-bit MTP drafter step instead of generation")
	mtpGenerate := flag.Bool("mtp-generate", false, "experimental Gemma4 MTP graph generation using -mtp-drafter (CPU verifier path)")
	mtpSeq := flag.Int("mtp-seq", 1, "external KV sequence length for -mtp-smoke when -mtp-real-prompt is false")
	mtpRealPrompt := flag.Bool("mtp-real-prompt", false, "for -mtp-smoke, prefill the prompt with the main model and feed real activation/KV to the drafter")
	mtpKVReuse := flag.Bool("mtp-kv-reuse", false, "for -mtp-smoke/-mtp-generate, build/reuse an in-process prompt KV context cache")
	mtpKVRepeat := flag.Int("mtp-kv-repeat", 1, "repeat MTP real-prompt context build N times to validate -mtp-kv-reuse hits")
	mtpDraftMin := flag.Int("mtp-draft-min", 1, "minimum draft count for -mtp-generate")
	mtpDraftInitial := flag.Int("mtp-draft-initial", 2, "initial draft count for -mtp-generate")
	mtpDraftMax := flag.Int("mtp-draft-max", 4, "maximum draft count for -mtp-generate")
	flag.Parse()

	if *eagerLoad {
		os.Setenv("GO_PHERENCE_EAGER_LOAD", "1")
	}
	if *gpuKVMaxSeq > 0 {
		os.Setenv("GO_PHERENCE_GPU_KV_MAX_SEQ", fmt.Sprint(*gpuKVMaxSeq))
	}
	if *useGPU && *mtpSmoke && *mtpRealPrompt && *gpuKVMaxSeq == 0 && os.Getenv("GO_PHERENCE_GPU_KV_MAX_SEQ") == "" {
		os.Setenv("GO_PHERENCE_GPU_KV_MAX_SEQ", "256")
	}
	if *useGPU || *mtpSmoke || *mtpGenerate {
		model.ForceOnTheFly = true
	}
	if *useGPU {
		if *turboQuant {
			fmt.Fprintln(os.Stderr, "warning: --turbo-quant currently applies to the CPU backend only")
		}
		if *speculative {
			fmt.Fprintln(os.Stderr, "warning: --speculative currently applies to the CPU backend only")
		}
	}
	if *speculative {
		os.Setenv("GO_PHERENCE_SPECULATIVE", "1")
		os.Setenv("GO_PHERENCE_SPECULATIVE_BLOCK", fmt.Sprint(*specBlock))
		os.Setenv("GO_PHERENCE_SPECULATIVE_NGRAM", fmt.Sprint(*specNGram))
		os.Setenv("GO_PHERENCE_SPECULATIVE_MIN_PROPOSAL", fmt.Sprint(*specMinProposal))
		os.Setenv("GO_PHERENCE_SPECULATIVE_PROPOSER", *specProposer)
		os.Setenv("GO_PHERENCE_SPECULATIVE_BACKEND", *specBackend)
		if *specDebug {
			os.Setenv("GO_PHERENCE_SPECULATIVE_DEBUG", "1")
		}
	}

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: llmgen -model <dir> [-prompt text] [-tokens N]")
		os.Exit(1)
	}
	if err := validateMTPCLIFlags(*tokens, *mtpSmoke, *mtpGenerate, *mtpDrafter, *mtpSeq); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Printf("Loading model from %s...\n", *dir)
	t0 := time.Now()
	m, err := model.LoadLlama(*dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	m.EnableTurboQuant = *turboQuant
	fmt.Printf("Loaded in %.2fs (%d layers, h=%d)\n", time.Since(t0).Seconds(),
		m.Config.NumLayers, m.Config.HiddenSize)

	tok, err := tokenizer.Load(*dir + "/tokenizer.json")
	if err == nil {
		m.Tok = tok
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(1)
	}

	ids := tok.Encode(*prompt)
	var gpuMod *model.GPUModel
	if *useGPU {
		var err error
		gpuMod, err = model.LoadGPUModelWithLayers(m, *gpuLayers)
		if err != nil {
			fmt.Printf("GPU model failed: %v (falling back to CPU)\n", err)
		}
	}

	if *mtpSmoke {
		if err := runGemma4MTPSmoke(m, gpuMod, *mtpDrafter, ids, *mtpSeq, *mtpRealPrompt, *mtpKVReuse, *mtpKVRepeat); err != nil {
			fmt.Fprintf(os.Stderr, "mtp smoke: %v\n", err)
			os.Exit(1)
		}
		return
	}

	fmt.Printf("Prompt: '%s' (%d tokens)\n", *prompt, len(ids))
	fmt.Printf("Generating %d tokens...\n\n", *tokens)

	start := time.Now()
	var output []int
	var mtpStats model.MTPSpeculationStats
	var mtpMissing []string
	var mtpGraphOutput int
	var mtpGreedyTail int
	var mtpCompressedKV bool
	var mtpFinalStateOutputLen int
	var mtpStepSummaries []model.MTPGraphGenerationStepSummary
	if *mtpGenerate {
		result, err := runGemma4MTPGenerate(m, gpuMod, *mtpDrafter, ids, *tokens, *mtpKVReuse, *turboQuant, model.MTPAdaptiveDraftPolicy{MinDrafts: *mtpDraftMin, InitialDrafts: *mtpDraftInitial, MaxDrafts: *mtpDraftMax})
		if err != nil {
			fmt.Fprintf(os.Stderr, "mtp generate: %v\n", err)
			os.Exit(1)
		}
		output = result.Output
		mtpStats = result.Stats
		mtpMissing = result.MissingForPublicGeneration
		mtpGraphOutput = result.GraphOutputTokens
		mtpGreedyTail = result.GreedyTailTokens
		mtpCompressedKV = result.UsedCompressedKV
		mtpFinalStateOutputLen = result.FinalStateOutputLen
		mtpStepSummaries = result.StepSummaries
	} else if gpuMod != nil {
		output = append(ids, gpuMod.Generate(ids, *tokens)...)
	} else if *speculative {
		output = append(ids, generatedSuffixFromFullOutput(len(ids), *tokens, m.GenerateSpeculative(ids, *tokens, model.SpeculativeConfigFromEnv()))...)
	} else {
		output = append(ids, generatedSuffixFromFullOutput(len(ids), *tokens, m.Generate(ids, *tokens))...)
	}
	elapsed := time.Since(start)

	promptTokenCount := mtpEffectivePromptTokenCount(len(ids), len(output), *mtpGenerate, mtpGraphOutput, mtpGreedyTail)
	generated := output
	if len(output) >= promptTokenCount {
		generated = output[promptTokenCount:]
	}
	text := tok.Decode(output)
	genText := tok.Decode(generated)

	fmt.Printf("--- Output ---\n%s\n--- End ---\n\n", text)
	fmt.Printf("Prompt tokens:    %d\n", promptTokenCount)
	fmt.Printf("Generated tokens: %d\n", len(generated))
	fmt.Printf("Total time:       %.2fs\n", elapsed.Seconds())

	if len(generated) > 0 && len(output) > 0 {
		promptTime := elapsed.Seconds() * float64(len(ids)) / float64(len(output))
		genTime := elapsed.Seconds() - promptTime
		tokPerSec := 0.0
		msPerTok := 0.0
		if genTime > 0 {
			tokPerSec = float64(len(generated)) / genTime
			msPerTok = genTime / float64(len(generated)) * 1000
		}
		fmt.Printf("Generation time:  %.2fs\n", genTime)
		fmt.Printf("Tokens/sec:       %.1f\n", tokPerSec)
		fmt.Printf("ms/token:         %.1f\n", msPerTok)
	}
	if *mtpGenerate {
		fmt.Printf("MTP graph steps:   %d\n", mtpStats.Steps)
		fmt.Printf("MTP drafted:       %d\n", mtpStats.DraftedTokens)
		fmt.Printf("MTP verified:      %d\n", mtpStats.VerifiedTokens)
		fmt.Printf("MTP acceptance:    %.2f\n", mtpStats.AcceptanceRate())
		fmt.Printf("MTP bonus tokens:  %d\n", mtpStats.BonusTokens)
		fmt.Printf("MTP graph output:  %d\n", mtpGraphOutput)
		fmt.Printf("MTP greedy tail:   %d\n", mtpGreedyTail)
		fmt.Printf("MTP compressed KV: %v\n", mtpCompressedKV)
		fmt.Printf("MTP state covers:  %s\n", formatMTPFinalStateCoverage(mtpFinalStateOutputLen, len(output)))
		for i, s := range mtpStepSummaries {
			fmt.Printf("MTP cycle %-3d: input=%d drafted=%v verifier_in=%v verifier_out=%v accepted=%d bonus=%d output=%v commit_pos=%v verifier_pos=%v all=%v\n", i, s.InputToken, s.DraftedTokens, s.VerifierTokens, s.VerifierOutputTokens, s.AcceptedPrefixLen, s.BonusToken, s.OutputTokens, s.Positions, s.VerifierPositions, s.AllDraftsAccepted)
		}
		if len(mtpMissing) > 0 {
			fmt.Printf("MTP public blockers: %v\n", mtpMissing)
		}
	}
	_ = genText
}

func generatedSuffixFromFullOutput(inputIDs, maxTokens int, output []int) []int {
	if maxTokens <= 0 || len(output) == 0 {
		return nil
	}
	if len(output) >= inputIDs {
		generated := len(output) - inputIDs
		if generated >= 0 && generated <= maxTokens {
			return append([]int(nil), output[inputIDs:]...)
		}
	}
	if len(output) <= maxTokens {
		return append([]int(nil), output...)
	}
	return append([]int(nil), output[len(output)-maxTokens:]...)
}

func mtpEffectivePromptTokenCount(inputIDs, outputLen int, mtpGenerate bool, graphOutput, greedyTail int) int {
	if !mtpGenerate {
		return inputIDs
	}
	prompt := outputLen - graphOutput - greedyTail
	if prompt < 0 || prompt > outputLen {
		return inputIDs
	}
	return prompt
}

func formatMTPFinalStateCoverage(finalStateOutputLen, outputLen int) string {
	if finalStateOutputLen == outputLen {
		return fmt.Sprintf("%d/%d tokens", finalStateOutputLen, outputLen)
	}
	return fmt.Sprintf("%d/%d tokens (greedy tail not covered)", finalStateOutputLen, outputLen)
}

func validateMTPCLIFlags(tokens int, mtpSmoke, mtpGenerate bool, mtpDrafter string, mtpSeq int) error {
	if tokens < 0 {
		return fmt.Errorf("tokens must be non-negative")
	}
	if (mtpSmoke || mtpGenerate) && mtpDrafter == "" {
		return fmt.Errorf("-mtp-smoke/-mtp-generate requires -mtp-drafter <dir>")
	}
	if mtpSeq <= 0 {
		return fmt.Errorf("-mtp-seq must be positive")
	}
	return nil
}

var mtpPromptContextCache = map[string]model.MTPPromptContext{}

func mtpPromptCacheKey(m *model.LlamaModel, ids []int) string {
	return fmt.Sprintf("%s:%d:%d:%v", m.Config.ModelType, m.Config.HiddenSize, m.Config.NumLayers, ids)
}

func buildMTPPromptContextCached(m *model.LlamaModel, gpuMod *model.GPUModel, ids []int, enabled bool) (model.MTPPromptContext, bool, error) {
	key := mtpPromptCacheKey(m, ids)
	if enabled {
		if ctx, ok := mtpPromptContextCache[key]; ok {
			return ctx, true, nil
		}
	}
	var ctx model.MTPPromptContext
	var err error
	if gpuMod != nil {
		ctx, err = gpuMod.BuildMTPPromptContext(ids)
	} else {
		ctx, err = m.BuildMTPPromptContext(ids)
	}
	if err != nil {
		return model.MTPPromptContext{}, false, err
	}
	if enabled {
		mtpPromptContextCache[key] = ctx
	}
	return ctx, false, nil
}

func runGemma4MTPGenerate(m *model.LlamaModel, gpuMod *model.GPUModel, drafterDir string, ids []int, maxTokens int, kvReuse bool, useCompressedKV bool, policy model.MTPAdaptiveDraftPolicy) (model.MTPGraphGenerationResult, error) {
	if maxTokens < 0 {
		return model.MTPGraphGenerationResult{}, fmt.Errorf("maxTokens=%d must be non-negative", maxTokens)
	}
	d, err := model.LoadGemma4MTPDrafter(drafterDir)
	if err != nil {
		return model.MTPGraphGenerationResult{}, fmt.Errorf("load drafter: %w", err)
	}
	if m.Config.HiddenSize != d.BackboneHiddenSize || m.Config.VocabSize != d.Config.VocabSize {
		return model.MTPGraphGenerationResult{}, fmt.Errorf("model/drafter mismatch model h/vocab=%d/%d drafter backbone/vocab=%d/%d", m.Config.HiddenSize, m.Config.VocabSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	ctx, _, err := buildMTPPromptContextCached(m, gpuMod, ids, kvReuse)
	if err != nil {
		return model.MTPGraphGenerationResult{}, fmt.Errorf("prompt context: %w", err)
	}
	externalKV, err := model.NewMTPDrafterExternalKVFromPromptContext(m, d, ctx)
	if err != nil {
		return model.MTPGraphGenerationResult{}, fmt.Errorf("external KV: %w", err)
	}
	return m.GenerateMTPGraphFromPromptContext(d, ctx, externalKV, model.MTPGraphGenerationOptions{MaxTokens: maxTokens, Policy: policy, UseCompressedKV: useCompressedKV})
}

type gemma4MTPSmokeResult struct {
	DrafterDir                    string                     `json:"drafter_dir"`
	ModelHidden                   int                        `json:"model_hidden"`
	ModelLayers                   int                        `json:"model_layers"`
	DrafterHidden                 int                        `json:"drafter_hidden"`
	DrafterBackbone               int                        `json:"drafter_backbone"`
	DrafterLayers                 int                        `json:"drafter_layers"`
	PackedEmbedding               bool                       `json:"packed_embedding"`
	PackedProjection              bool                       `json:"packed_projection"`
	PackedLayerWeights            bool                       `json:"packed_layer_weights"`
	PreviousToken                 int                        `json:"previous_token"`
	Token                         int                        `json:"token"`
	LogitsLen                     int                        `json:"logits_len"`
	NextActivationLen             int                        `json:"next_activation_len"`
	LoadSeconds                   float64                    `json:"load_seconds"`
	PromptTokens                  int                        `json:"prompt_tokens,omitempty"`
	PromptSeconds                 float64                    `json:"prompt_seconds,omitempty"`
	RealPrompt                    bool                       `json:"real_prompt"`
	KVReuse                       bool                       `json:"kv_reuse,omitempty"`
	KVCacheHit                    bool                       `json:"kv_cache_hit,omitempty"`
	KVRepeat                      int                        `json:"kv_repeat,omitempty"`
	StepSeconds                   float64                    `json:"step_seconds"`
	MTPGraphCapabilities          model.MTPGraphCapabilities `json:"mtp_graph_capabilities"`
	MTPMissingForPublicGeneration []string                   `json:"mtp_missing_for_public_generation,omitempty"`
}

func runGemma4MTPSmoke(m *model.LlamaModel, gpuMod *model.GPUModel, drafterDir string, ids []int, seqLen int, realPrompt bool, kvReuse bool, kvRepeat int) error {
	start := time.Now()
	d, err := model.LoadGemma4MTPDrafter(drafterDir)
	if err != nil {
		return fmt.Errorf("load drafter: %w", err)
	}
	loadElapsed := time.Since(start)
	if m.Config.HiddenSize != d.BackboneHiddenSize || m.Config.VocabSize != d.Config.VocabSize {
		return fmt.Errorf("model/drafter mismatch model h/vocab=%d/%d drafter backbone/vocab=%d/%d", m.Config.HiddenSize, m.Config.VocabSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	var promptCtx model.MTPPromptContext
	var promptSeconds float64
	var externalKV *model.MTPDrafterExternalKV
	var state model.MTPDrafterState
	previousToken := 0
	cacheHit := false
	if realPrompt {
		if kvRepeat < 1 {
			kvRepeat = 1
		}
		prefillStart := time.Now()
		for i := 0; i < kvRepeat; i++ {
			var hit bool
			var err error
			promptCtx, hit, err = buildMTPPromptContextCached(m, gpuMod, ids, kvReuse)
			if err != nil {
				return fmt.Errorf("prompt context: %w", err)
			}
			cacheHit = cacheHit || hit
		}
		promptSeconds = time.Since(prefillStart).Seconds()
		externalKV, err = model.NewMTPDrafterExternalKVFromPromptContext(m, d, promptCtx)
		if err != nil {
			return fmt.Errorf("external KV: %w", err)
		}
		previousToken = promptCtx.PreviousToken
		state, err = model.NewMTPDrafterState(previousToken, promptCtx.Activation, d.BackboneHiddenSize)
		if err != nil {
			return fmt.Errorf("state: %w", err)
		}
	} else {
		k := make([][]float32, d.Config.NumLayers)
		v := make([][]float32, d.Config.NumLayers)
		for i := range d.Layers {
			kvDim, err := d.LayerKVDim(i)
			if err != nil {
				return fmt.Errorf("drafter layer %d KV dim: %w", i, err)
			}
			k[i] = make([]float32, seqLen*kvDim)
			v[i] = make([]float32, seqLen*kvDim)
		}
		var err error
		externalKV, err = model.NewMTPDrafterExternalKV(d, k, v, seqLen)
		if err != nil {
			return fmt.Errorf("external KV: %w", err)
		}
		if len(ids) > 0 {
			previousToken = ids[len(ids)-1]
		}
		state, err = model.NewMTPDrafterState(previousToken, make([]float32, d.BackboneHiddenSize), d.BackboneHiddenSize)
		if err != nil {
			return fmt.Errorf("state: %w", err)
		}
	}
	stepStart := time.Now()
	step, err := m.RunMTPDrafterStepWithExternalKV(d, state, externalKV)
	if err != nil {
		return fmt.Errorf("drafter step: %w", err)
	}
	stepElapsed := time.Since(stepStart)
	packedLayer := len(d.Layers) > 0 && d.Layers[0].QWm != nil && d.Layers[0].OWm != nil && d.Layers[0].GateWm != nil && d.Layers[0].UpWm != nil && d.Layers[0].DownWm != nil
	caps := model.Gemma4MTPGraphCapabilities()
	res := gemma4MTPSmokeResult{
		DrafterDir:                    drafterDir,
		ModelHidden:                   m.Config.HiddenSize,
		ModelLayers:                   len(m.Layers),
		DrafterHidden:                 d.Config.HiddenSize,
		DrafterBackbone:               d.BackboneHiddenSize,
		DrafterLayers:                 len(d.Layers),
		PackedEmbedding:               d.EmbedTokensMLX != nil,
		PackedProjection:              d.PreProjectionMLX != nil && d.PostProjectionMLX != nil,
		PackedLayerWeights:            packedLayer,
		PreviousToken:                 previousToken,
		Token:                         step.Token,
		LogitsLen:                     len(step.Logits),
		NextActivationLen:             len(step.NextActivation),
		LoadSeconds:                   loadElapsed.Seconds(),
		PromptTokens:                  promptCtx.SeqLen,
		PromptSeconds:                 promptSeconds,
		RealPrompt:                    realPrompt,
		KVReuse:                       kvReuse,
		KVCacheHit:                    cacheHit,
		KVRepeat:                      kvRepeat,
		StepSeconds:                   stepElapsed.Seconds(),
		MTPGraphCapabilities:          caps,
		MTPMissingForPublicGeneration: caps.MissingForPublicGeneration(),
	}
	if realPrompt {
		fmt.Printf("MTP prompt prefill: %.2fs (%d tokens, next=%d)\n", promptSeconds, promptCtx.SeqLen, promptCtx.FinalToken)
	}
	out, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return err
	}
	fmt.Printf("--- MTP Smoke ---\n%s\n--- End MTP Smoke ---\n", out)
	return nil
}
