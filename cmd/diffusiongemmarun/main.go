package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/rcarmo/go-pherence/cmd/internal/dgflags"
	"log"
	"os"
	"runtime/pprof"
	"strconv"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/diffusiongemma"
)

type stringList []string

var fatalCleanups []func()

func registerFatalCleanup(fn func()) {
	if fn != nil {
		fatalCleanups = append(fatalCleanups, fn)
	}
}

func runFatalCleanups() {
	for i := len(fatalCleanups) - 1; i >= 0; i-- {
		func(fn func()) {
			defer func() { _ = recover() }()
			fn()
		}(fatalCleanups[i])
	}
	fatalCleanups = nil
}

func (s *stringList) String() string { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error {
	*s = append(*s, v)
	return nil
}

type report struct {
	ModelPath       string                             `json:"model_path"`
	PromptIDs       []int                              `json:"prompt_ids"`
	Options         diffusiongemma.InferenceOptions    `json:"options"`
	PromptTokens    []string                           `json:"prompt_tokens,omitempty"`
	GeneratedTokens []string                           `json:"generated_tokens,omitempty"`
	Capabilities    diffusiongemma.RuntimeCapabilities `json:"capabilities"`
	OperationStatus []diffusiongemma.OpStatus          `json:"operation_status,omitempty"`
	Shards          *diffusiongemma.ShardAvailability  `json:"shards,omitempty"`
	Summary         diffusiongemma.ReadinessSummary    `json:"summary"`
	Result          *diffusiongemma.InferenceResult    `json:"result,omitempty"`
	Error           string                             `json:"error,omitempty"`
}

func flagWasSet(name string) bool {
	found := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
}

func applyK3LMHeadPreset(k3, k3A100Q8 *bool, lmHeadTopK *int, lmHead, lmHeadPrefetch *bool) {
	if k3 == nil || k3A100Q8 == nil || !*k3 || !*k3A100Q8 {
		return
	}
	if lmHeadTopK != nil && *lmHeadTopK == 0 && !flagWasSet("lm-head-top-k") {
		*lmHeadTopK = 64
	}
	if lmHead != nil && !*lmHead && !flagWasSet("k3-a100-lmhead") {
		*lmHead = true
	}
	if lmHeadPrefetch != nil && !*lmHeadPrefetch && !flagWasSet("k3-a100-lmhead-prefetch") {
		*lmHeadPrefetch = true
	}
}

func main() {
	modelDir := flag.String("model", "", "DiffusionGemma model directory")
	promptCSV := flag.String("prompt-ids", "", "comma-separated already-tokenized prompt IDs")
	promptText := flag.String("prompt", "", "text prompt to tokenize with tokenizer.json")
	var messages stringList
	flag.Var(&messages, "message", "chat message as role:text; may be repeated; simplified scaffold, not full Jinja")
	messagesJSON := flag.String("messages-json", "", "JSON array of {role,content} messages")
	messagesFile := flag.String("messages-file", "", "path to JSON array of {role,content} messages")
	exactTokensCSV := flag.String("tokens", "", "comma-separated exact tokenizer vocabulary entries (no BPE tokenization)")
	maxNew := flag.Int("max-new", 0, "maximum generated tokens")
	canvas := flag.Int("canvas", 0, "override canvas length")
	diffusionSteps := flag.Int("diffusion-steps", 0, "record requested public diffusion steps separately from effective denoising max")
	denoiseSteps := flag.Int("denoise-steps", 0, "override effective maximum denoising steps")
	tMin := flag.Float64("t-min", -1, "override final denoising temperature")
	tMax := flag.Float64("t-max", -1, "override initial denoising temperature")
	entropyBound := flag.Float64("entropy-bound", -1, "override entropy-bound sampler threshold")
	samplerMode := flag.String("sampler-mode", "", "denoising sampler mode: argmax or entropy_bound")
	stabilityThreshold := flag.Int("stability", -1, "override stable-canvas stopping threshold")
	confidenceThreshold := flag.Float64("confidence", -1, "override mean entropy confidence threshold")
	seed := flag.Int64("seed", 0, "deterministic canvas RNG seed")
	addBOS := flag.Bool("add-bos", false, "prepend BOS token from tokenizer metadata")
	enableThinking := flag.Bool("think", false, "prepend thinking control token from tokenizer metadata")
	addGenerationPrompt := flag.Bool("generation-prompt", false, "append generation prompt token when available")
	useChatTemplate := flag.Bool("chat-template", false, "render repeated -message values through simplified native Gemma chat template")
	decode := flag.Bool("decode", false, "decode prompt/generated IDs through exact tokenizer vocabulary entries")
	mockToken := flag.Int("mock-token", -1, "use deterministic mock denoiser that always favors this token ID")
	mockTokensCSV := flag.String("mock-tokens", "", "comma-separated deterministic mock denoiser token ID pattern")
	useCPUDispatcher := flag.Bool("cpu-dispatcher", false, "use CPU/SIMD DiffusionGemma dispatcher as a parity/reference path")
	useGPUDispatcher := flag.Bool("gpu-dispatcher", false, "open local text weights and use GPU/CUDA dispatcher")
	fp8Model := flag.String("fp8-model", "", "path to FP8-dynamic DiffusionGemma checkpoint directory for GPU inference")
	residentExpertLayers := flag.Int("resident-expert-layers", -1, "deprecated alias for -fp8-expert-prewarm-layers")
	fp8ExpertPrewarmLayers := flag.Int("fp8-expert-prewarm-layers", 9, "pre-upload and pin all FP8 experts for the first N layers to GPU (0 disables)")
	allowSlowCPU := flag.Bool("allow-slow-cpu", false, "allow explicit CPU/SIMD reference generation when -cpu-dispatcher is selected")
	eagerMmap := flag.Bool("eager-mmap", false, "prefault all mapped safetensor shards before CPU dispatcher run")
	preloadGlobals := flag.Bool("preload-globals", false, "predecode/cache global text tensors before CPU dispatcher run")
	residentLayers := flag.Int("resident-layers", 0, "predecode/cache first N text layers before CPU dispatcher run")
	residencyBudgetGiB := flag.Float64("residency-budget-gib", 0, "choose resident layer prefix from decoded float32 cache budget in GiB")
	k3Native := flag.Bool("k3", false, "enable native K3/RVV dispatch for DiffusionGemma")
	k3Threads := flag.Int("k3-threads", 0, "number of K3/X100 worker threads for DiffusionGemma native paths")
	k3A100Q8 := flag.Bool("k3-a100-q8", false, "enable K3 A100 row-scale Q80x32 projection paths")
	k3A100Workers := flag.Int("k3-a100-workers", 0, "number of K3 A100 worker threads for Q80x32 GEMMs")
	q80PrewarmLayers := flag.Int("k3-q80-prewarm-layers", 0, "prepack first N text layers into the K3 A100 Q80 cache")
	q80PrewarmExperts := flag.Bool("k3-q80-prewarm-experts", false, "include all per-expert tensors when using -k3-q80-prewarm-layers; memory-heavy")
	q80ResidencyBudgetGiB := flag.Float64("k3-q80-residency-budget-gib", 0, "choose K3 Q80 prewarm layer count from a packed-weight cache budget in GiB")
	q80RetainSelectedExpertLayers := flag.Int("k3-q80-retain-selected-expert-layers", 0, "retain on-demand selected expert Q80 caches for first N layers across denoising steps")
	maxDispatchLayers := flag.Int("max-dispatch-layers", 0, "debug: execute at most N text layers")
	tailAfterMaxLayers := flag.Bool("tail-after-max-layers", false, "debug: run tail ops after -max-dispatch-layers")
	lmHeadTopK := flag.Int("lm-head-top-k", 0, "debug: keep only top-K LM head logits per position, storing -Inf elsewhere")
	k3A100LMHead := flag.Bool("k3-a100-lmhead", false, "enable K3 A100 Q80 LM-head shortlist with exact rerank")
	k3A100LMHeadPrefetch := flag.Bool("k3-a100-lmhead-prefetch", false, "prepack K3 A100 LM-head Q80 weights while decoder layers run")
	k3A100LMHeadCandidates := flag.Int("k3-a100-lmhead-candidates", 0, "candidate count for K3 A100 LM-head exact rerank")
	k3Q80Prefetch := flag.Bool("k3-q80-prefetch", false, "prefetch the next layer's K3 Q80 cache while the current layer runs")
	k3Q80PrefetchExperts := flag.Bool("k3-q80-prefetch-experts", false, "include experts in next-layer K3 Q80 prefetch")
	k3Q80SelectedPrefetch := flag.Bool("k3-q80-selected-prefetch", false, "prefetch router-selected expert K3 Q80 weights while dense MLP runs")
	dispatchProgress := flag.Bool("dispatch-progress", false, "print GPU/backend dispatcher layer/tail progress to stderr")
	skipEviction := flag.Bool("skip-eviction", false, "debug: retain decoded/Q80 layer caches across denoising steps")
	ggufModel := flag.String("gguf-model", "", "GGUF model file (Q4_K_M, same as llama.cpp) — replaces safetensor weights")
	preloadOnly := flag.Bool("preload-only", false, "open weights, apply residency/preload options, report cache entries, and exit without generation")
	asJSON := flag.Bool("json", false, "emit JSON")
	cpuProfile := flag.String("cpuprofile", "", "write CPU profile to file")
	flag.Parse()
	applyK3LMHeadPreset(k3Native, k3A100Q8, lmHeadTopK, k3A100LMHead, k3A100LMHeadPrefetch)
	if *k3Native {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3", "1")
	}
	if *k3Threads > 0 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_THREADS", strconv.Itoa(*k3Threads))
	}
	if *k3A100Q8 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_Q8", "1")
	}
	if *k3A100Workers > 0 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_WORKERS", strconv.Itoa(*k3A100Workers))
	}
	if *k3A100LMHead {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_LMHEAD", "1")
	}
	if *k3A100LMHeadPrefetch {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_LMHEAD_PREFETCH", "1")
	}
	if *k3A100LMHeadCandidates > 0 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_LMHEAD_CANDIDATES", strconv.Itoa(*k3A100LMHeadCandidates))
	}
	if *k3Q80Prefetch {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_Q80_PREFETCH", "1")
	}
	if *k3Q80PrefetchExperts {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_Q80_PREFETCH_EXPERTS", "1")
	}
	if *k3Q80SelectedPrefetch {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_Q80_SELECTED_PREFETCH", "1")
	}
	if *allowSlowCPU && !*useCPUDispatcher {
		fmt.Fprintf(os.Stderr, "diffusiongemmarun: -allow-slow-cpu is accepted for compatibility; use -cpu-dispatcher to select the CPU/SIMD reference dispatcher\n")
	}
	if *residentExpertLayers >= 0 {
		*fp8ExpertPrewarmLayers = *residentExpertLayers
	}
	if *cpuProfile != "" {
		f, _ := os.Create(*cpuProfile)
		if f != nil {
			pprof.StartCPUProfile(f)
			defer pprof.StopCPUProfile()
		}
	}
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: diffusiongemmarun -model PATH [-prompt-ids 1,2] [-max-new N] [-json]")
		os.Exit(2)
	}
	promptIDs, err := parseIDs(*promptCSV)
	if err != nil {
		fatal(err)
	}
	jsonMessages, err := loadJSONMessages(*messagesJSON, *messagesFile)
	if err != nil {
		fatal(err)
	}
	var vocab *diffusiongemma.Vocab
	var tok *tokenizer.Tokenizer
	if strings.TrimSpace(*promptText) != "" || len(messages) > 0 || len(jsonMessages) > 0 {
		tok, err = tokenizer.Load(*modelDir + "/tokenizer.json")
		if err != nil {
			fatal(err)
		}
	}
	if strings.TrimSpace(*promptText) != "" {
		promptIDs = append(promptIDs, tok.Encode(*promptText)...)
	}
	if strings.TrimSpace(*exactTokensCSV) != "" || *decode {
		vocab, err = diffusiongemma.LoadVocab(*modelDir)
		if err != nil {
			fatal(err)
		}
	}
	if strings.TrimSpace(*exactTokensCSV) != "" {
		ids, err := vocab.EncodeExact(splitCSV(*exactTokensCSV))
		if err != nil {
			fatal(err)
		}
		promptIDs = append(promptIDs, ids...)
	}
	m, err := diffusiongemma.LoadMetadata(*modelDir)
	if err != nil {
		fatal(err)
	}
	if *enableThinking && len(messages) == 0 && len(jsonMessages) == 0 {
		messages = append(messages, "system:")
	}
	if len(messages) > 0 || len(jsonMessages) > 0 {
		if m.Tokenizer == nil {
			fatal(fmt.Errorf("DiffusionGemma tokenizer metadata unavailable"))
		}
		if *useChatTemplate {
			specials := m.Tokenizer.SpecialTokenIDs(m.Processor)
			parsed := append([]diffusiongemma.TextChatMessage(nil), jsonMessages...)
			flagMessages, err := parseFlagMessages(messages)
			if err != nil {
				fatal(err)
			}
			parsed = append(parsed, flagMessages...)
			framed, err := diffusiongemma.BuildTemplateChatPromptIDs(parsed, specials, tok.Encode, diffusiongemma.ChatRenderOptions{AddBOS: *addBOS, EnableThinking: *enableThinking, AddGenerationPrompt: *addGenerationPrompt})
			if err != nil {
				fatal(err)
			}
			promptIDs = append(promptIDs, framed.InputIDs...)
		} else {
			specials := m.Tokenizer.SpecialTokenIDs(m.Processor)
			parsed := append([]diffusiongemma.TextChatMessage(nil), jsonMessages...)
			flagMessages, err := parseFlagMessages(messages)
			if err != nil {
				fatal(err)
			}
			parsed = append(parsed, flagMessages...)
			chatMessages := make([]diffusiongemma.ChatMessage, 0, len(parsed))
			for _, msg := range parsed {
				role := strings.TrimSpace(msg.Role)
				text := role + "\n" + msg.Content
				if *enableThinking && (role == "system" || role == "developer") {
					text = role + "\n" + m.Processor.Think + "\n" + msg.Content
				}
				chatMessages = append(chatMessages, diffusiongemma.ChatMessage{Role: role, Content: tok.Encode(text)})
			}
			framed, err := diffusiongemma.BuildSimpleChatPromptIDs(chatMessages, specials, diffusiongemma.ChatPromptOptions{AddBOS: *addBOS})
			if err != nil {
				fatal(err)
			}
			promptIDs = append(promptIDs, framed.InputIDs...)
			if *addGenerationPrompt {
				if specials.BOT < 0 {
					fatal(fmt.Errorf("DiffusionGemma begin-turn token ID unavailable"))
				}
				promptIDs = append(promptIDs, specials.BOT)
				promptIDs = append(promptIDs, tok.Encode("model")...)
				promptIDs = append(promptIDs, 107) // newline token, matching BuildTemplateChatPromptIDs
				if !*enableThinking {
					if specials.BOC < 0 || specials.EOC < 0 {
						fatal(fmt.Errorf("DiffusionGemma channel token IDs unavailable"))
					}
					promptIDs = append(promptIDs, specials.BOC)
					promptIDs = append(promptIDs, tok.Encode("thought")...)
					promptIDs = append(promptIDs, 107) // newline token, matching BuildTemplateChatPromptIDs
					promptIDs = append(promptIDs, specials.EOC)
				}
			}
		}
	} else if *addBOS || *enableThinking || *addGenerationPrompt {
		if m.Tokenizer == nil {
			fatal(fmt.Errorf("DiffusionGemma tokenizer metadata unavailable"))
		}
		specials := m.Tokenizer.SpecialTokenIDs(m.Processor)
		// If we tokenized text in this process, render the generation prompt as a
		// full Gemma turn header (`<|turn>model\n`) rather than a bare begin-turn
		// token. Token-ID-only callers keep the older token-level helper behavior
		// because there is no encoder available to synthesize `model\n`.
		textGenerationPrompt := *addGenerationPrompt && tok != nil
		framed, err := diffusiongemma.BuildPromptIDs(promptIDs, specials, diffusiongemma.PromptOptions{AddBOS: *addBOS, EnableThinking: *enableThinking, AddGenerationPrompt: *addGenerationPrompt && !textGenerationPrompt})
		if err != nil {
			fatal(err)
		}
		promptIDs = framed.InputIDs
		if textGenerationPrompt {
			if specials.BOT < 0 {
				fatal(fmt.Errorf("DiffusionGemma begin-turn token ID unavailable"))
			}
			promptIDs = append(promptIDs, specials.BOT)
			promptIDs = append(promptIDs, tok.Encode("model")...)
			promptIDs = append(promptIDs, 107) // newline token, matching BuildTemplateChatPromptIDs
			if !*enableThinking {
				if specials.BOC < 0 || specials.EOC < 0 {
					fatal(fmt.Errorf("DiffusionGemma channel token IDs unavailable"))
				}
				promptIDs = append(promptIDs, specials.BOC)
				promptIDs = append(promptIDs, tok.Encode("thought")...)
				promptIDs = append(promptIDs, 107) // newline token, matching BuildTemplateChatPromptIDs
				promptIDs = append(promptIDs, specials.EOC)
			}
		}
	}
	if m.Tokenizer != nil {
		specials := m.Tokenizer.SpecialTokenIDs(m.Processor)
		promptIDs, err = diffusiongemma.ExpandImagePlaceholderTokens(promptIDs, specials, m.Shape.VisionSoftTokens)
		if err != nil {
			fatal(err)
		}
	}
	var denoiser diffusiongemma.Denoiser
	var weights *diffusiongemma.TextWeights
	q80Prewarmed := 0
	if strings.TrimSpace(*mockTokensCSV) != "" {
		mockIDs, err := parseIDs(*mockTokensCSV)
		if err != nil {
			fatal(err)
		}
		denoiser = diffusiongemma.MockDenoiser{VocabSize: m.Shape.VocabSize, TokenIDs: mockIDs}
	} else if *mockToken >= 0 {
		denoiser = diffusiongemma.MockDenoiser{VocabSize: m.Shape.VocabSize, TokenID: *mockToken}
	}
	var ggufFile *gguf.GGUF // keep alive for mmap
	if *ggufModel != "" {
		// GGUF-only path: use exactly the same Q4_K_M weights as llama.cpp
		log.Printf("loading ALL weights from GGUF: %s", *ggufModel)
		ggufFile, err = gguf.Open(*ggufModel)
		if err != nil {
			fatal(err)
		}
		weights, err = diffusiongemma.OpenTextWeightsFromGGUF(ggufFile, m.Shape)
		if err != nil {
			fatal(err)
		}
		defer weights.Close()
		ggufIdx, err := diffusiongemma.BuildGGUFExpertIndex(ggufFile, m.Shape.TextLayers, m.Shape.NumExperts)
		if err != nil {
			fatal(fmt.Errorf("GGUF expert index build failed: %w", err))
		}
		log.Printf("GGUF expert index: %d layers × %d experts, intermediate=%d",
			ggufIdx.NumLayers, ggufIdx.NumExperts, ggufIdx.Intermediate)
		if *useGPUDispatcher {
			strictGGUFGraph := diffusionGemmaBackendGraphRequired()
			ggufRuntimeCleaned := false
			cleanupGGUFRuntime := func() {
				if !ggufRuntimeCleaned {
					ggufRuntimeCleaned = true
					diffusiongemma.FreeGGUFGPURuntimeCaches()
				}
			}
			defer cleanupGGUFRuntime()
			registerFatalCleanup(cleanupGGUFRuntime)
			prewarmStart := time.Now()
			densePrewarmLayers := *residentLayers
			layers, bytes, err := diffusiongemma.PrewarmGGUFGPUWeightCacheLayers(weights, densePrewarmLayers)
			if err != nil {
				fatal(err)
			}
			fmt.Fprintf(os.Stderr, "diffusiongemmarun: prewarmed resident GGUF GPU dense weights layers=%d bytes=%.2f GiB elapsed=%.1fs\n", layers, float64(bytes)/(1024*1024*1024), time.Since(prewarmStart).Seconds())
			if denseTransposeBudget := diffusionGemmaGGUFDenseTransposeCacheMB(); denseTransposeBudget > 0 {
				transposeStart := time.Now()
				matrices, transposeBytes, err := diffusiongemma.PrewarmGGUFDenseTransposeCache(weights, *residentLayers)
				if err != nil {
					fatal(err)
				}
				cacheEntries, cacheBytes := diffusiongemma.GGUFDenseTransposeCacheStats()
				fmt.Fprintf(os.Stderr, "diffusiongemmarun: prewarmed GGUF dense transpose cache matrices=%d bytes=%.2f GiB cache_entries=%d cache_bytes=%.2f GiB budget=%dMiB elapsed=%.1fs\n", matrices, float64(transposeBytes)/(1024*1024*1024), cacheEntries, float64(cacheBytes)/(1024*1024*1024), denseTransposeBudget, time.Since(transposeStart).Seconds())
			}
			if expertLayers := diffusionGemmaGGUFGPUExpertPrewarmLayers(); expertLayers != 0 {
				expertStart := time.Now()
				layers, experts, expertBytes, err := diffusiongemma.PrewarmGGUFGPUPointerExpertCache(ggufIdx, expertLayers)
				if err != nil {
					fatal(err)
				}
				fmt.Fprintf(os.Stderr, "diffusiongemmarun: prewarmed GGUF pointer expert cache layers=%d experts=%d bytes=%.2f GiB elapsed=%.1fs\n", layers, experts, float64(expertBytes)/(1024*1024*1024), time.Since(expertStart).Seconds())
			}
			gpuDisp := diffusiongemma.GPUDispatcher{
				ResidentLayerPrefix:   *residentLayers,
				MaxLayers:             *maxDispatchLayers,
				TailAfterMaxLayers:    *tailAfterMaxLayers,
				GGUFExpertIndex:       ggufIdx,
				LMHeadTopK:            *lmHeadTopK,
				Progress:              *dispatchProgress,
				SkipEviction:          true, // GGUF TextWeights are fully pre-cached and cannot reload evicted tensors.
				FinalLogitSoftcapping: float32(m.Config.TextConfig.FinalLogitSoftcapping),
			}
			if strictGGUFGraph {
				lmStart := time.Now()
				lmBuf, lmVocab, lmHidden, err := diffusiongemma.UploadGGUFF32LMHeadBuffer(weights)
				if err != nil {
					fatal(err)
				}
				gpuDisp.F32LMHead = lmBuf
				gpuDisp.F32LMHeadVocab = lmVocab
				gpuDisp.F32LMHeadHidden = lmHidden
				lmCleaned := false
				cleanupLM := func() {
					if !lmCleaned {
						lmCleaned = true
						lmBuf.Free()
					}
				}
				defer cleanupLM()
				registerFatalCleanup(cleanupLM)
				fmt.Fprintf(os.Stderr, "diffusiongemmarun: GGUF dense F32 LM-head [%d,%d] resident for backend graph (%.2f GiB) elapsed=%.1fs\n", lmVocab, lmHidden, float64(lmVocab)*float64(lmHidden)*4/(1024*1024*1024), time.Since(lmStart).Seconds())
			}
			if strictGGUFGraph || dgflags.GPUSelfCondEnabled() {
				if strictGGUFGraph {
					fmt.Fprintf(os.Stderr, "diffusiongemmarun: GGUF GPU self-conditioning resident for backend graph\n")
				} else {
					fmt.Fprintf(os.Stderr, "diffusiongemmarun: WARNING: GGUF GPU self-conditioning is experimental; enabling resident F32 embed_tokens for SC\n")
				}
				scBuf, scVocab, scHidden, err := diffusiongemma.UploadSelfConditioningEmbeddingBuffer(weights)
				if err != nil {
					fatal(err)
				}
				gpuDisp.SCEmbed = scBuf
				gpuDisp.SCEmbedVocab = scVocab
				gpuDisp.SCEmbedHidden = scHidden
				scCleaned := false
				cleanupSC := func() {
					if !scCleaned {
						scCleaned = true
						scBuf.Free()
					}
				}
				defer cleanupSC()
				registerFatalCleanup(cleanupSC)
				fmt.Fprintf(os.Stderr, "diffusiongemmarun: GGUF SC embedding [%d,%d] resident (%.2f GiB)\n", scVocab, scHidden, float64(scVocab)*float64(scHidden)*4/(1024*1024*1024))
			}
			denoiser, err = diffusiongemma.NewTextDenoiserWithDispatcher(m.Shape, weights, gpuDisp)
			if err != nil {
				if gpuDisp.SCEmbed != nil {
					gpuDisp.SCEmbed.Free()
					gpuDisp.SCEmbed = nil
				}
				if gpuDisp.F32LMHead != nil {
					gpuDisp.F32LMHead.Free()
					gpuDisp.F32LMHead = nil
				}
			}
		} else {
			cpuDisp := diffusiongemma.CPUDispatcher{
				ResidentLayerPrefix:   *residentLayers,
				MaxLayers:             *maxDispatchLayers,
				TailAfterMaxLayers:    *tailAfterMaxLayers,
				GGUFExpertIndex:       ggufIdx,
				LMHeadTopK:            *lmHeadTopK,
				Progress:              *dispatchProgress,
				SkipEviction:          true, // GGUF TextWeights are fully pre-cached and cannot reload evicted tensors.
				FinalLogitSoftcapping: float32(m.Config.TextConfig.FinalLogitSoftcapping),
			}
			denoiser, err = diffusiongemma.NewTextDenoiserWithDispatcher(m.Shape, weights, cpuDisp)
		}
		if err != nil {
			fatal(err)
		}
		_ = ggufFile // keep alive
	} else if *useCPUDispatcher || *useGPUDispatcher {
		if m.Shards != nil && !m.Shards.Ready {
			missing := m.Shards.MissingShards
			if len(missing) > 5 {
				missing = missing[:5]
			}
			fatal(fmt.Errorf("DiffusionGemma weight shards not ready: present=%d/%d missing=%v", m.Shards.PresentShards, m.Shards.ExpectedShards, missing))
		}
		weights, err = diffusiongemma.OpenTextWeights(*modelDir, m.Shape)
		if err != nil {
			fatal(err)
		}
		defer weights.Close()
		if *eagerMmap {
			if n, err := weights.EagerLoad(); err != nil {
				fatal(err)
			} else {
				fmt.Fprintf(os.Stderr, "diffusiongemmarun: eager mmap touched %d bytes\n", n)
			}
		}
		if *preloadGlobals {
			if err := weights.PreloadGlobals(); err != nil {
				fatal(err)
			}
		}
		if *residencyBudgetGiB > 0 {
			budgetBytes := int64(*residencyBudgetGiB * 1024 * 1024 * 1024)
			budget := diffusiongemma.EstimateResidencyBudgetFromWeights(weights, true, budgetBytes)
			*residentLayers = budget.ResidentLayers
			fmt.Fprintf(os.Stderr, "diffusiongemmarun: residency budget %.2f GiB selects resident_layers=%d/%d resident_bytes=%d\n", *residencyBudgetGiB, budget.ResidentLayers, budget.TotalLayers, budget.ResidentBytes)
		}
		if *residentLayers > 0 {
			if *useGPUDispatcher {
				if err := weights.PreloadLayerRangeLightweight(0, *residentLayers); err != nil {
					fatal(err)
				}
			} else {
				if err := weights.PreloadLayerRange(0, *residentLayers); err != nil {
					fatal(err)
				}
			}
		}
		if *q80RetainSelectedExpertLayers > 0 {
			weights.RetainQ80ExpertLayerPrefix(*q80RetainSelectedExpertLayers)
			fmt.Fprintf(os.Stderr, "diffusiongemmarun: K3 Q80 retaining selected expert caches for layers=%d\n", *q80RetainSelectedExpertLayers)
		}
		if *q80ResidencyBudgetGiB > 0 {
			budgetBytes := int64(*q80ResidencyBudgetGiB * 1024 * 1024 * 1024)
			budget := diffusiongemma.EstimateQ80ResidencyBudgetFromWeights(weights, *q80PrewarmExperts, budgetBytes)
			*q80PrewarmLayers = budget.ResidentLayers
			fmt.Fprintf(os.Stderr, "diffusiongemmarun: K3 Q80 residency budget %.2f GiB selects q80_prewarm_layers=%d/%d q80_resident_bytes=%d include_experts=%v\n", *q80ResidencyBudgetGiB, budget.ResidentLayers, budget.TotalLayers, budget.ResidentBytes, *q80PrewarmExperts)
		}
		if *q80PrewarmLayers > 0 {
			q80Prewarmed, err = weights.PreloadLayerRangeQ80(0, *q80PrewarmLayers, *q80PrewarmExperts)
			if err != nil {
				fatal(err)
			}
			if n, err := weights.PreloadSelfConditioningQ80(); err != nil {
				fatal(err)
			} else {
				q80Prewarmed += n
			}
			fmt.Fprintf(os.Stderr, "diffusiongemmarun: K3 Q80 prewarmed layers=%d tensors=%d include_experts=%v\n", *q80PrewarmLayers, q80Prewarmed, *q80PrewarmExperts)
		}
		if *preloadOnly {
			present, expected := 0, 0
			ready := false
			if m.Shards != nil {
				present, expected, ready = m.Shards.PresentShards, m.Shards.ExpectedShards, m.Shards.Ready
			}
			fmt.Printf("DiffusionGemma preload scaffold: %s\n", *modelDir)
			fmt.Printf("  shards_ready=%v present=%d/%d\n", ready, present, expected)
			fmt.Printf("  preload_globals=%v resident_layers=%d residency_budget_gib=%.2f eager_mmap=%v k3_q80_prewarm_layers=%d k3_q80_residency_budget_gib=%.2f k3_q80_retain_selected_expert_layers=%d k3_q80_prewarm_tensors=%d k3_q80_prewarm_experts=%v float_cache_entries=%d float_cache_bytes=%d q80_cache_entries=%d q80_cache_bytes=%d\n", *preloadGlobals, *residentLayers, *residencyBudgetGiB, *eagerMmap, *q80PrewarmLayers, *q80ResidencyBudgetGiB, *q80RetainSelectedExpertLayers, q80Prewarmed, *q80PrewarmExperts, weights.FloatCacheEntries(), weights.FloatCacheBytes(), weights.Q80CacheEntries(), weights.Q80CacheBytes())
			return
		}
		finalSoftcap := float32(m.Shape.FinalLogitSoftcapping)
		if finalSoftcap == 0 {
			finalSoftcap = float32(m.Config.TextConfig.FinalLogitSoftcapping)
		}
		if *useGPUDispatcher {
			gpuDisp := diffusiongemma.GPUDispatcher{ResidentLayerPrefix: *residentLayers, MaxLayers: *maxDispatchLayers, TailAfterMaxLayers: *tailAfterMaxLayers, LMHeadTopK: *lmHeadTopK, Progress: *dispatchProgress, SkipEviction: *skipEviction, FinalLogitSoftcapping: finalSoftcap}
			if *fp8Model != "" {
				fmt.Fprintf(os.Stderr, "diffusiongemmarun: loading FP8 weights from %s\n", *fp8Model)
				fp8Weights, err := diffusiongemma.OpenFP8TextWeights(*fp8Model, m.Shape)
				if err != nil {
					fatal(err)
				}
				defer fp8Weights.Close()
				if n, err := fp8Weights.EagerLoad(); err == nil && n > 0 {
					fmt.Fprintf(os.Stderr, "diffusiongemmarun: eager-loaded FP8 file: %d bytes\n", n)
				}
				fmt.Fprintf(os.Stderr, "diffusiongemmarun: uploading %d FP8 layers to GPU\n", len(fp8Weights.Layers))
				gpuModel, err := diffusiongemma.UploadFP8Layers(fp8Weights)
				if err != nil {
					fatal(err)
				}
				gpuDisp.FP8Model = gpuModel
				gpuDisp.FP8Weights = fp8Weights
				lmHeadBuf, lmVocab, lmHidden, err := diffusiongemma.UploadFP8LMHeadBuffer(fp8Weights)
				if err != nil {
					gpuModel.Free()
					fatal(err)
				}
				gpuDisp.FP8LMHead = lmHeadBuf
				gpuDisp.FP8LMHeadVocab = lmVocab
				gpuDisp.FP8LMHeadHidden = lmHidden
				fp8Cleaned := false
				cleanupFP8GPU := func() {
					if !fp8Cleaned {
						fp8Cleaned = true
						if gpuDisp.ExpertCache != nil {
							gpuDisp.ExpertCache.ClearAll()
						}
						if gpuDisp.FP8LMHead != nil {
							gpuDisp.FP8LMHead.Free()
							gpuDisp.FP8LMHead = nil
						}
						if gpuDisp.FP8Model != nil {
							gpuDisp.FP8Model.Free()
							gpuDisp.FP8Model = nil
						}
					}
				}
				defer cleanupFP8GPU()
				registerFatalCleanup(cleanupFP8GPU)
				fmt.Fprintf(os.Stderr, "diffusiongemmarun: %d FP8 layers uploaded to GPU; LM head [%d,%d] resident\n", len(gpuModel.Layers), lmVocab, lmHidden)
				var scEmbedBytes int64
				if dgflags.GPUSelfCondEnabled() {
					fmt.Fprintf(os.Stderr, "diffusiongemmarun: WARNING: GPU self-conditioning is experimental and not reference-correct yet\n")
					scBuf, scVocab, scHidden, err := diffusiongemma.UploadSelfConditioningEmbeddingBuffer(weights)
					if err != nil {
						gpuModel.Free()
						lmHeadBuf.Free()
						fatal(err)
					}
					gpuDisp.SCEmbed = scBuf
					gpuDisp.SCEmbedVocab = scVocab
					gpuDisp.SCEmbedHidden = scHidden
					scCleaned := false
					cleanupSC := func() {
						if !scCleaned {
							scCleaned = true
							scBuf.Free()
						}
					}
					defer cleanupSC()
					registerFatalCleanup(cleanupSC)
					scEmbedBytes = int64(scVocab) * int64(scHidden) * 4
					fmt.Fprintf(os.Stderr, "diffusiongemmarun: SC embedding [%d,%d] resident (%.1f MB)\n", scVocab, scHidden, float64(scEmbedBytes)/1e6)
				}
				// FP8 expert cache: reserve remaining VRAM for experts. Default is the RTX
				// 3060-safe historical budget, override with
				// GO_PHERENCE_DIFFUSIONGEMMA_EXPERT_CACHE_MB for experiments.
				expertBudget := dgflags.ExpertCacheBudgetBytes(7200) - int64(lmVocab)*int64(lmHidden)*2 - scEmbedBytes - gpuModel.DenseTransposeBytes()
				nExperts := m.Shape.NumExperts
				if nExperts <= 0 {
					nExperts = 128
				}
				if expertBudget > 0 {
					gpuDisp.ExpertCache = diffusiongemma.NewExpertLRUCache(expertBudget)
					if *fp8ExpertPrewarmLayers > 0 {
						layers, experts, err := gpuDisp.ExpertCache.PrewarmLayerPrefix(fp8Weights, *fp8ExpertPrewarmLayers, nExperts)
						if err != nil {
							fatal(fmt.Errorf("FP8 expert prewarm failed: %w", err))
						}
						fmt.Fprintf(os.Stderr, "diffusiongemmarun: prewarmed FP8 experts layers=%d/%d experts=%d pinned_prefix=%d cache=%s\n", layers, *fp8ExpertPrewarmLayers, experts, gpuDisp.ExpertCache.PinnedLayerPrefix(nExperts), gpuDisp.ExpertCache.Stats())
					}
				} else if *fp8ExpertPrewarmLayers > 0 {
					fmt.Fprintf(os.Stderr, "diffusiongemmarun: skipping FP8 expert prewarm; LM head leaves no expert cache budget\n")
				}
			}
			denoiser, err = diffusiongemma.NewTextDenoiserWithDispatcher(m.Shape, weights, gpuDisp)
			if err != nil && gpuDisp.SCEmbed != nil {
				gpuDisp.SCEmbed.Free()
				gpuDisp.SCEmbed = nil
			}
		} else {
			cpuDisp := diffusiongemma.CPUDispatcher{ResidentLayerPrefix: *residentLayers, MaxLayers: *maxDispatchLayers, TailAfterMaxLayers: *tailAfterMaxLayers, LMHeadTopK: *lmHeadTopK, Progress: *dispatchProgress, SkipEviction: *skipEviction, FinalLogitSoftcapping: finalSoftcap}
			denoiser, err = diffusiongemma.NewTextDenoiserWithDispatcher(m.Shape, weights, cpuDisp)
		}
		if err != nil {
			fatal(err)
		}
	}
	eng, err := diffusiongemma.NewEngineWithTextWeights(m, weights, denoiser)
	if err != nil {
		fatal(err)
	}
	opts := diffusiongemma.InferenceOptions{MaxNewTokens: *maxNew, CanvasLength: *canvas, Seed: *seed, RequestedDiffusionSteps: *diffusionSteps}
	if denoising := buildDenoisingOverride(m.Denoising, *denoiseSteps, *tMin, *tMax, *entropyBound, *samplerMode, *stabilityThreshold, *confidenceThreshold); denoising != nil {
		opts.Denoising = denoising
	}
	caps := diffusiongemma.Capabilities()
	out := report{ModelPath: *modelDir, PromptIDs: promptIDs, Options: opts, Capabilities: caps, OperationStatus: diffusiongemma.OperationStatuses(), Shards: m.Shards, Summary: diffusiongemma.BuildReadinessSummary(caps, m.Shards, m.Readiness)}
	if *decode {
		if tok != nil {
			out.PromptTokens = []string{tok.Decode(promptIDs)}
		} else if vocab != nil {
			out.PromptTokens = vocab.DecodeIDs(promptIDs)
		}
	}
	res, err := eng.GenerateTokenIDs(promptIDs, opts)
	if err != nil {
		out.Error = err.Error()
	} else {
		out.Result = &res
		if *decode {
			if tok != nil {
				out.GeneratedTokens = []string{tok.Decode(res.Generated)}
			} else if vocab != nil {
				out.GeneratedTokens = vocab.DecodeIDs(res.Generated)
			}
		}
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}
		if out.Error != "" {
			os.Exit(1)
		}
		return
	}
	fmt.Printf("DiffusionGemma run scaffold: %s\n", *modelDir)
	fmt.Printf("  prompt_ids=%v max_new=%d canvas=%d seed=%d cpu_dispatcher=%v mock_token=%d mock_tokens=%q\n", promptIDs, opts.MaxNewTokens, opts.CanvasLength, opts.Seed, *useCPUDispatcher, *mockToken, *mockTokensCSV)
	if opts.Denoising != nil {
		d := opts.Denoising
		fmt.Printf("  denoising: requested_diffusion_steps=%d effective_steps=%d t=[%.3f, %.3f] sampler_mode=%s entropy_bound=%.3f stability=%d confidence=%.6f\n", opts.RequestedDiffusionSteps, d.MaxDenoisingSteps, d.TMin, d.TMax, d.Sampler.Mode, d.Sampler.EntropyBound, d.StabilityThreshold, d.ConfidenceThreshold)
	}
	if len(out.PromptTokens) > 0 {
		fmt.Printf("  prompt_tokens=%v\n", out.PromptTokens)
	}
	if out.Shards != nil {
		fmt.Printf("  shards_ready=%v present=%d/%d\n", out.Shards.Ready, out.Shards.PresentShards, out.Shards.ExpectedShards)
	}
	fmt.Printf("  caps: text_runtime=%v reference_complete=%v encoder_kv=%v sliding_mask=%v rope=%v\n", out.Capabilities.TextOnlyScaffoldReady, out.Capabilities.ReferenceComplete, out.Capabilities.EncoderKVConcat, out.Capabilities.SlidingWindowMask, out.Capabilities.RoPE)
	if len(out.Capabilities.MissingForReference) > 0 {
		fmt.Printf("  missing_reference=%v\n", out.Capabilities.MissingForReference)
	}
	fmt.Printf("  summary: %s\n", out.Summary.String())
	if len(out.OperationStatus) > 0 {
		implemented, referenceComplete := 0, 0
		for _, op := range out.OperationStatus {
			if op.Implemented {
				implemented++
			}
			if op.ReferenceComplete {
				referenceComplete++
			}
		}
		fmt.Printf("  op_status: implemented=%d/%d reference_complete=%d/%d\n", implemented, len(out.OperationStatus), referenceComplete, len(out.OperationStatus))
	}
	if weights != nil {
		fmt.Printf("  residency: resident_layers=%d residency_budget_gib=%.2f k3_q80_prewarm_layers=%d k3_q80_residency_budget_gib=%.2f k3_q80_retain_selected_expert_layers=%d max_dispatch_layers=%d tail_after_max_layers=%v lm_head_top_k=%d skip_eviction=%v float_cache_entries=%d float_cache_bytes=%d q80_cache_entries=%d q80_cache_bytes=%d\n", *residentLayers, *residencyBudgetGiB, *q80PrewarmLayers, *q80ResidencyBudgetGiB, *q80RetainSelectedExpertLayers, *maxDispatchLayers, *tailAfterMaxLayers, *lmHeadTopK, *skipEviction, weights.FloatCacheEntries(), weights.FloatCacheBytes(), weights.Q80CacheEntries(), weights.Q80CacheBytes())
	}
	if *useGPUDispatcher && *fp8Model != "" {
		gpuDisp, ok := denoiser.(*diffusiongemma.TextDenoiser)
		if ok {
			if gd, ok := gpuDisp.Dispatcher.(diffusiongemma.GPUDispatcher); ok && gd.ExpertCache != nil {
				gd.ExpertCache.ReportCacheStats()
			}
		}
	}
	if out.Error != "" {
		fmt.Printf("  error: %s\n", out.Error)
		os.Exit(1)
	}
	fmt.Printf("  generated=%v canvases=%d\n", res.Generated, len(res.Canvases))
	if len(out.GeneratedTokens) > 0 {
		fmt.Printf("  generated_tokens=%v\n", out.GeneratedTokens)
	}
}

func loadJSONMessages(raw, path string) ([]diffusiongemma.TextChatMessage, error) {
	if strings.TrimSpace(raw) != "" && strings.TrimSpace(path) != "" {
		return nil, fmt.Errorf("use only one of -messages-json or -messages-file")
	}
	if strings.TrimSpace(path) != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		raw = string(b)
	}
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	var messages []diffusiongemma.TextChatMessage
	if err := json.Unmarshal([]byte(raw), &messages); err != nil {
		return nil, err
	}
	return messages, nil
}

func parseFlagMessages(messages []string) ([]diffusiongemma.TextChatMessage, error) {
	out := make([]diffusiongemma.TextChatMessage, 0, len(messages))
	for _, raw := range messages {
		role, content, ok := strings.Cut(raw, ":")
		if !ok {
			return nil, fmt.Errorf("bad -message %q, want role:text", raw)
		}
		out = append(out, diffusiongemma.TextChatMessage{Role: strings.TrimSpace(role), Content: content})
	}
	return out, nil
}

func diffusionGemmaBackendGraphRequired() bool {
	for _, name := range []string{"GO_PHERENCE_DIFFUSIONGEMMA_BACKEND_GRAPH", "GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_GROUPED_EXPERT_GRAPH"} {
		v := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
		if v == "1" || v == "true" || v == "yes" || v == "on" {
			return true
		}
	}
	return false
}

func diffusionGemmaGGUFDenseTransposeCacheMB() int {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_DENSE_TRANSPOSE_CACHE_MB"))
	if v == "" {
		return 0
	}
	if n, err := strconv.Atoi(v); err == nil && n > 0 {
		return n
	}
	return 0
}

func diffusionGemmaGGUFGPUExpertPrewarmLayers() int {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_LAYERS"))
	if v == "" {
		return 0
	}
	if n, err := strconv.Atoi(v); err == nil {
		return n
	}
	return 0
}

func buildDenoisingOverride(base diffusiongemma.DenoisingConfig, steps int, tMin, tMax, entropyBound float64, samplerMode string, stability int, confidence float64) *diffusiongemma.DenoisingConfig {
	changed := false
	cfg := base
	if steps > 0 {
		cfg.MaxDenoisingSteps = steps
		changed = true
	}
	if tMin >= 0 {
		cfg.TMin = tMin
		changed = true
	}
	if tMax >= 0 {
		cfg.TMax = tMax
		changed = true
	}
	if entropyBound >= 0 {
		cfg.Sampler.EntropyBound = entropyBound
		changed = true
	}
	if strings.TrimSpace(samplerMode) != "" {
		cfg.Sampler.Mode = diffusiongemma.SamplerMode(strings.TrimSpace(samplerMode))
		changed = true
	}
	if stability >= 0 {
		cfg.StabilityThreshold = stability
		changed = true
	}
	if confidence >= 0 {
		cfg.ConfidenceThreshold = confidence
		changed = true
	}
	if !changed {
		return nil
	}
	return &cfg
}

func splitCSV(csv string) []string {
	if strings.TrimSpace(csv) == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseIDs(csv string) ([]int, error) {
	if strings.TrimSpace(csv) == "" {
		return nil, nil
	}
	parts := strings.Split(csv, ",")
	ids := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("bad token id %q: %w", part, err)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func fatal(err error) {
	runFatalCleanups()
	fmt.Fprintln(os.Stderr, "diffusiongemmarun:", err)
	os.Exit(1)
}
