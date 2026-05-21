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
	mtpSeq := flag.Int("mtp-seq", 1, "external KV sequence length for -mtp-smoke when -mtp-real-prompt is false")
	mtpRealPrompt := flag.Bool("mtp-real-prompt", false, "for -mtp-smoke, prefill the prompt with the main model and feed real activation/KV to the drafter")
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
	if *useGPU || *mtpSmoke {
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
	if *tokens < 0 {
		fmt.Fprintln(os.Stderr, "tokens must be non-negative")
		os.Exit(1)
	}
	if *mtpSmoke && *mtpDrafter == "" {
		fmt.Fprintln(os.Stderr, "-mtp-smoke requires -mtp-drafter <dir>")
		os.Exit(1)
	}
	if *mtpSeq <= 0 {
		fmt.Fprintln(os.Stderr, "-mtp-seq must be positive")
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
		if err := runGemma4MTPSmoke(m, gpuMod, *mtpDrafter, ids, *mtpSeq, *mtpRealPrompt); err != nil {
			fmt.Fprintf(os.Stderr, "mtp smoke: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if *mtpDrafter != "" {
		fmt.Fprintln(os.Stderr, "warning: -mtp-drafter is currently only wired for -mtp-smoke; generation uses the regular path")
	}

	fmt.Printf("Prompt: '%s' (%d tokens)\n", *prompt, len(ids))
	fmt.Printf("Generating %d tokens...\n\n", *tokens)

	start := time.Now()
	var output []int
	if gpuMod != nil {
		output = append(ids, gpuMod.Generate(ids, *tokens)...)
	} else if *speculative {
		output = m.GenerateSpeculative(ids, *tokens, model.SpeculativeConfigFromEnv())
	} else {
		output = m.Generate(ids, *tokens)
	}
	elapsed := time.Since(start)

	generated := output
	if len(output) >= len(ids) {
		generated = output[len(ids):]
	}
	text := tok.Decode(output)
	genText := tok.Decode(generated)

	fmt.Printf("--- Output ---\n%s\n--- End ---\n\n", text)
	fmt.Printf("Prompt tokens:    %d\n", len(ids))
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
	_ = genText
}

type gemma4MTPSmokeResult struct {
	DrafterDir         string  `json:"drafter_dir"`
	ModelHidden        int     `json:"model_hidden"`
	ModelLayers        int     `json:"model_layers"`
	DrafterHidden      int     `json:"drafter_hidden"`
	DrafterBackbone    int     `json:"drafter_backbone"`
	DrafterLayers      int     `json:"drafter_layers"`
	PackedEmbedding    bool    `json:"packed_embedding"`
	PackedProjection   bool    `json:"packed_projection"`
	PackedLayerWeights bool    `json:"packed_layer_weights"`
	PreviousToken      int     `json:"previous_token"`
	Token              int     `json:"token"`
	LogitsLen          int     `json:"logits_len"`
	NextActivationLen  int     `json:"next_activation_len"`
	LoadSeconds        float64 `json:"load_seconds"`
	PromptTokens       int     `json:"prompt_tokens,omitempty"`
	PromptSeconds      float64 `json:"prompt_seconds,omitempty"`
	RealPrompt         bool    `json:"real_prompt"`
	StepSeconds        float64 `json:"step_seconds"`
}

func mapDrafterSourcesByWidth(m *model.LlamaModel, d *model.Gemma4MTPDrafter, seqLen int) ([]int, error) {
	if m == nil || d == nil || seqLen <= 0 {
		return nil, fmt.Errorf("invalid source mapping inputs")
	}
	sources := make([]int, d.Config.NumLayers)
	used := make(map[int]bool)
	for i := 0; i < d.Config.NumLayers; i++ {
		headDim := d.Config.HeadDim
		if i < len(d.Layers) && d.Layers[i].HeadDimLocal > 0 {
			headDim = d.Layers[i].HeadDimLocal
		}
		kvHeads := d.Config.NumKVHeads
		if i < len(d.Config.LayerTypes) && d.Config.LayerTypes[i] == "full_attention" && d.Config.NumGlobalKVHeads > 0 {
			kvHeads = d.Config.NumGlobalKVHeads
		}
		wantDim := kvHeads * headDim
		wantLen := seqLen * wantDim
		best := -1
		for l := 0; l < len(m.Layers); l++ {
			if used[l] {
				continue
			}
			dim, err := m.LayerKVDim(l)
			if err != nil {
				return nil, err
			}
			if dim == wantDim {
				best = l
				break
			}
		}
		if best < 0 {
			return nil, fmt.Errorf("no main KV source for drafter layer %d width=%d len=%d", i, wantDim, wantLen)
		}
		sources[i] = best
		used[best] = true
	}
	return sources, nil
}

func runGemma4MTPSmoke(m *model.LlamaModel, gpuMod *model.GPUModel, drafterDir string, ids []int, seqLen int, realPrompt bool) error {
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
	if realPrompt {
		prefillStart := time.Now()
		var err error
		if gpuMod != nil {
			promptCtx, err = gpuMod.BuildMTPPromptContext(ids)
		} else {
			promptCtx, err = m.BuildMTPPromptContext(ids)
		}
		if err != nil {
			return fmt.Errorf("prompt context: %w", err)
		}
		promptSeconds = time.Since(prefillStart).Seconds()
		sources, err := mapDrafterSourcesByWidth(m, d, promptCtx.SeqLen)
		if err != nil {
			return fmt.Errorf("map external KV: %w", err)
		}
		externalKV = &model.MTPDrafterExternalKV{K: promptCtx.KVCacheK, V: promptCtx.KVCacheV, SourceLayers: sources, SeqLen: promptCtx.SeqLen}
		previousToken = promptCtx.PreviousToken
		state, err = model.NewMTPDrafterState(previousToken, promptCtx.Activation, d.BackboneHiddenSize)
		if err != nil {
			return fmt.Errorf("state: %w", err)
		}
	} else {
		k := make([][]float32, d.Config.NumLayers)
		v := make([][]float32, d.Config.NumLayers)
		for i := range d.Layers {
			headDim := d.Config.HeadDim
			if d.Layers[i].HeadDimLocal > 0 {
				headDim = d.Layers[i].HeadDimLocal
			}
			kvDim := d.Config.NumKVHeads * headDim
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
	res := gemma4MTPSmokeResult{
		DrafterDir:         drafterDir,
		ModelHidden:        m.Config.HiddenSize,
		ModelLayers:        len(m.Layers),
		DrafterHidden:      d.Config.HiddenSize,
		DrafterBackbone:    d.BackboneHiddenSize,
		DrafterLayers:      len(d.Layers),
		PackedEmbedding:    d.EmbedTokensMLX != nil,
		PackedProjection:   d.PreProjectionMLX != nil && d.PostProjectionMLX != nil,
		PackedLayerWeights: packedLayer,
		PreviousToken:      previousToken,
		Token:              step.Token,
		LogitsLen:          len(step.Logits),
		NextActivationLen:  len(step.NextActivation),
		LoadSeconds:        loadElapsed.Seconds(),
		PromptTokens:       promptCtx.SeqLen,
		PromptSeconds:      promptSeconds,
		RealPrompt:         realPrompt,
		StepSeconds:        stepElapsed.Seconds(),
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
