package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/diffusiongemma"
)

type server struct {
	modelID             string
	modelDir            string
	model               *diffusiongemma.Model
	engine              *diffusiongemma.Engine
	tok                 *tokenizer.Tokenizer
	vocab               *diffusiongemma.Vocab
	mu                  sync.Mutex
	defaultMaxNew       int
	defaultCanvas       int
	defaultDenoiseStep  int
	defaultSeed         int64
	addBOS              bool
	addGenerationPrompt bool
}

type openAIMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type completionRequest struct {
	Model                string                          `json:"model"`
	Prompt               interface{}                     `json:"prompt,omitempty"`
	Messages             []openAIMessage                 `json:"messages,omitempty"`
	PromptIDs            []int                           `json:"prompt_ids,omitempty"`
	MaxTokens            int                             `json:"max_tokens,omitempty"`
	MaxNewTokens         int                             `json:"max_new_tokens,omitempty"`
	CanvasLength         int                             `json:"canvas_length,omitempty"`
	DenoiseSteps         int                             `json:"denoise_steps,omitempty"`
	Seed                 int64                           `json:"seed,omitempty"`
	Stream               bool                            `json:"stream,omitempty"`
	ReturnDiffusionSteps bool                            `json:"return_diffusion_steps,omitempty"`
	Denoising            *diffusiongemma.DenoisingConfig `json:"denoising,omitempty"`
}

type renderStep struct {
	GeneratedTokenIndex int      `json:"generated_token_index"`
	Step                int      `json:"step"`
	Temperature         float64  `json:"temperature"`
	Canvas              []int    `json:"canvas"`
	AcceptedMask        []bool   `json:"accepted_mask"`
	Text                string   `json:"text,omitempty"`
	Tokens              []string `json:"tokens,omitempty"`
	MeanEntropy         float64  `json:"mean_entropy"`
	Stopped             bool     `json:"stopped"`
}

type usage struct {
	PromptTokens     int   `json:"prompt_tokens"`
	CompletionTokens int   `json:"completion_tokens"`
	TotalTokens      int   `json:"total_tokens"`
	LatencyMS        int64 `json:"latency_ms,omitempty"`
}

type diffusionStats struct {
	RequestedMaxTokens        int     `json:"requested_max_tokens"`
	EffectiveMaxTokens        int     `json:"effective_max_tokens"`
	RequestedCanvasLength     int     `json:"requested_canvas_length,omitempty"`
	EffectiveCanvasLength     int     `json:"effective_canvas_length"`
	RequestedDenoisingSteps   int     `json:"requested_denoising_steps,omitempty"`
	EffectiveDenoisingSteps   int     `json:"effective_denoising_steps"`
	GeneratedTokens           int     `json:"generated_tokens"`
	Blocks                    int     `json:"blocks"`
	DenoisingStepsExecuted    int     `json:"denoising_steps_executed"`
	CanvasPositionEvaluations int     `json:"canvas_position_evaluations"`
	LatencyMS                 int64   `json:"latency_ms"`
	GeneratedTokensPerSecond  float64 `json:"generated_tokens_per_second"`
	CanvasPositionsPerSecond  float64 `json:"canvas_positions_per_second"`
}

type choice struct {
	Index        int               `json:"index"`
	Text         string            `json:"text,omitempty"`
	Message      *openAIChatChoice `json:"message,omitempty"`
	Delta        *openAIChatChoice `json:"delta,omitempty"`
	FinishReason string            `json:"finish_reason"`
}

type openAIChatChoice struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type completionResponse struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	Choices           []choice       `json:"choices"`
	Usage             usage          `json:"usage"`
	PromptTokenIDs    []int          `json:"prompt_token_ids,omitempty"`
	GeneratedTokenIDs []int          `json:"generated_token_ids,omitempty"`
	DiffusionSteps    []renderStep   `json:"diffusion_steps,omitempty"`
	DiffusionStats    diffusionStats `json:"diffusion_stats"`
}

func main() {
	modelDir := flag.String("model", "", "DiffusionGemma model directory")
	listen := flag.String("listen", ":8080", "listen address")
	modelID := flag.String("model-id", "diffusiongemma-k3", "OpenAI model id to advertise")
	maxNew := flag.Int("max-new", 1, "default maximum generated tokens")
	canvas := flag.Int("canvas", 0, "default canvas length")
	denoiseSteps := flag.Int("denoise-steps", 0, "default denoising steps override")
	seed := flag.Int64("seed", 1, "default deterministic canvas RNG seed")
	addBOS := flag.Bool("add-bos", false, "prepend BOS token for chat prompts when metadata is available")
	addGenerationPrompt := flag.Bool("generation-prompt", true, "append model generation prompt for chat prompts when metadata is available")
	mockToken := flag.Int("mock-token", -1, "use deterministic mock denoiser that always favors this token ID")
	mockTokensCSV := flag.String("mock-tokens", "", "comma-separated deterministic mock denoiser token ID pattern")
	useCPUDispatcher := flag.Bool("cpu-dispatcher", true, "open local text weights and attach CPU/SIMD dispatcher")
	allowSlowCPU := flag.Bool("allow-slow-cpu", false, "allow experimental full-weight CPU dispatcher")
	eagerMmap := flag.Bool("eager-mmap", false, "prefault all mapped safetensor shards")
	preloadGlobals := flag.Bool("preload-globals", false, "predecode/cache global text tensors")
	residentLayers := flag.Int("resident-layers", 0, "predecode/cache first N text layers")
	residencyBudgetGiB := flag.Float64("residency-budget-gib", 0, "choose resident layer prefix from decoded float32 cache budget in GiB")
	k3Native := flag.Bool("k3", false, "enable native K3/RVV dispatch")
	k3Threads := flag.Int("k3-threads", 0, "number of K3/X100 worker threads")
	k3A100Q8 := flag.Bool("k3-a100-q8", false, "enable K3 A100 row-scale Q80x32 projection paths")
	k3A100Workers := flag.Int("k3-a100-workers", 0, "number of K3 A100 worker threads for Q80x32 GEMMs")
	q80PrewarmLayers := flag.Int("k3-q80-prewarm-layers", 0, "prepack first N text layers into K3 A100 Q80 cache")
	q80PrewarmExperts := flag.Bool("k3-q80-prewarm-experts", false, "include all per-expert tensors when prewarming Q80")
	q80ResidencyBudgetGiB := flag.Float64("k3-q80-residency-budget-gib", 0, "choose K3 Q80 prewarm layer count from packed-weight cache budget in GiB")
	q80RetainSelectedExpertLayers := flag.Int("k3-q80-retain-selected-expert-layers", 0, "retain selected expert Q80 caches for first N layers")
	maxDispatchLayers := flag.Int("max-dispatch-layers", 0, "debug: execute at most N text layers")
	tailAfterMaxLayers := flag.Bool("tail-after-max-layers", false, "debug: run tail ops after -max-dispatch-layers")
	lmHeadTopK := flag.Int("lm-head-top-k", 0, "debug: keep only top-K LM head logits")
	k3A100LMHead := flag.Bool("k3-a100-lmhead", false, "enable K3 A100 Q80 LM-head shortlist")
	k3A100LMHeadPrefetch := flag.Bool("k3-a100-lmhead-prefetch", false, "prepack K3 A100 LM-head Q80 while decoder layers run")
	k3A100LMHeadCandidates := flag.Int("k3-a100-lmhead-candidates", 0, "candidate count for K3 A100 LM-head exact rerank")
	k3Q80Prefetch := flag.Bool("k3-q80-prefetch", false, "prefetch next layer's K3 Q80 cache while current layer runs")
	k3Q80SelectedPrefetch := flag.Bool("k3-q80-selected-prefetch", false, "prefetch router-selected expert K3 Q80 weights while dense MLP runs")
	dispatchProgress := flag.Bool("dispatch-progress", false, "print dispatcher progress to stderr")
	skipEviction := flag.Bool("skip-eviction", false, "keep decoded/Q80 layer caches across requests")
	flag.Parse()
	applyK3LMHeadPreset(k3Native, k3A100Q8, lmHeadTopK, k3A100LMHead, k3A100LMHeadPrefetch)
	if *modelDir == "" {
		log.Fatal("usage: diffusiongemmaserver -model PATH [-listen :8080] [-allow-slow-cpu]")
	}
	setK3Env(*k3Native, *k3Threads, *k3A100Q8, *k3A100Workers, *k3A100LMHead, *k3A100LMHeadPrefetch, *k3A100LMHeadCandidates, *k3Q80Prefetch, *k3Q80SelectedPrefetch)

	m, err := diffusiongemma.LoadMetadata(*modelDir)
	if err != nil {
		log.Fatal(err)
	}
	tok, err := tokenizer.Load(*modelDir + "/tokenizer.json")
	if err != nil {
		log.Printf("diffusiongemmaserver: tokenizer unavailable, prompt_ids requests still work: %v", err)
	}
	vocab, err := diffusiongemma.LoadVocab(*modelDir)
	if err != nil {
		log.Printf("diffusiongemmaserver: exact vocab unavailable: %v", err)
	}

	var weights *diffusiongemma.TextWeights
	var denoiser diffusiongemma.Denoiser
	if strings.TrimSpace(*mockTokensCSV) != "" {
		ids, err := parseIDs(*mockTokensCSV)
		if err != nil {
			log.Fatal(err)
		}
		denoiser = diffusiongemma.MockDenoiser{VocabSize: m.Shape.VocabSize, TokenIDs: ids}
	} else if *mockToken >= 0 {
		denoiser = diffusiongemma.MockDenoiser{VocabSize: m.Shape.VocabSize, TokenID: *mockToken}
	}
	if *useCPUDispatcher {
		if !*allowSlowCPU {
			log.Fatal("DiffusionGemma full-weight CPU dispatcher is experimental; re-run with -allow-slow-cpu to proceed")
		}
		if m.Shards != nil && !m.Shards.Ready {
			missing := m.Shards.MissingShards
			if len(missing) > 5 {
				missing = missing[:5]
			}
			log.Fatalf("DiffusionGemma weight shards not ready: present=%d/%d missing=%v", m.Shards.PresentShards, m.Shards.ExpectedShards, missing)
		}
		weights, err = diffusiongemma.OpenTextWeights(*modelDir, m.Shape)
		if err != nil {
			log.Fatal(err)
		}
		defer weights.Close()
		if *eagerMmap {
			if n, err := weights.EagerLoad(); err != nil {
				log.Fatal(err)
			} else {
				log.Printf("diffusiongemmaserver: eager mmap touched %d bytes", n)
			}
		}
		if *preloadGlobals {
			if err := weights.PreloadGlobals(); err != nil {
				log.Fatal(err)
			}
		}
		if *residencyBudgetGiB > 0 {
			budget := diffusiongemma.EstimateResidencyBudgetFromWeights(weights, true, int64(*residencyBudgetGiB*1024*1024*1024))
			*residentLayers = budget.ResidentLayers
			log.Printf("diffusiongemmaserver: residency budget %.2f GiB selects resident_layers=%d/%d resident_bytes=%d", *residencyBudgetGiB, budget.ResidentLayers, budget.TotalLayers, budget.ResidentBytes)
		}
		if *residentLayers > 0 {
			if err := weights.PreloadLayerRange(0, *residentLayers); err != nil {
				log.Fatal(err)
			}
		}
		if *q80RetainSelectedExpertLayers > 0 {
			weights.RetainQ80ExpertLayerPrefix(*q80RetainSelectedExpertLayers)
			log.Printf("diffusiongemmaserver: K3 Q80 retaining selected expert caches for layers=%d", *q80RetainSelectedExpertLayers)
		}
		if *q80ResidencyBudgetGiB > 0 {
			budget := diffusiongemma.EstimateQ80ResidencyBudgetFromWeights(weights, *q80PrewarmExperts, int64(*q80ResidencyBudgetGiB*1024*1024*1024))
			*q80PrewarmLayers = budget.ResidentLayers
			log.Printf("diffusiongemmaserver: K3 Q80 residency budget %.2f GiB selects q80_prewarm_layers=%d/%d q80_resident_bytes=%d include_experts=%v", *q80ResidencyBudgetGiB, budget.ResidentLayers, budget.TotalLayers, budget.ResidentBytes, *q80PrewarmExperts)
		}
		if *q80PrewarmLayers > 0 {
			n, err := weights.PreloadLayerRangeQ80(0, *q80PrewarmLayers, *q80PrewarmExperts)
			if err != nil {
				log.Fatal(err)
			}
			if sc, err := weights.PreloadSelfConditioningQ80(); err != nil {
				log.Fatal(err)
			} else {
				n += sc
			}
			log.Printf("diffusiongemmaserver: K3 Q80 prewarmed layers=%d tensors=%d include_experts=%v", *q80PrewarmLayers, n, *q80PrewarmExperts)
		}
		denoiser, err = diffusiongemma.NewTextDenoiserWithDispatcher(m.Shape, weights, diffusiongemma.CPUDispatcher{ResidentLayerPrefix: *residentLayers, MaxLayers: *maxDispatchLayers, TailAfterMaxLayers: *tailAfterMaxLayers, LMHeadTopK: *lmHeadTopK, Progress: *dispatchProgress, SkipEviction: *skipEviction})
		if err != nil {
			log.Fatal(err)
		}
	}
	eng, err := diffusiongemma.NewEngineWithTextWeights(m, weights, denoiser)
	if err != nil {
		log.Fatal(err)
	}
	s := &server{modelID: *modelID, modelDir: *modelDir, model: m, engine: eng, tok: tok, vocab: vocab, defaultMaxNew: *maxNew, defaultCanvas: *canvas, defaultDenoiseStep: *denoiseSteps, defaultSeed: *seed, addBOS: *addBOS, addGenerationPrompt: *addGenerationPrompt}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/completions", s.handleCompletions)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	log.Printf("diffusiongemmaserver: listening on %s model=%s", *listen, *modelID)
	log.Fatal(http.ListenAndServe(*listen, logRequests(mux)))
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

func setK3Env(k3 bool, k3Threads int, k3A100Q8 bool, k3A100Workers int, lmHead bool, lmHeadPrefetch bool, lmHeadCandidates int, q80Prefetch bool, selectedPrefetch bool) {
	if k3 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3", "1")
	}
	if k3Threads > 0 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_THREADS", strconv.Itoa(k3Threads))
	}
	if k3A100Q8 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_Q8", "1")
	}
	if k3A100Workers > 0 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_WORKERS", strconv.Itoa(k3A100Workers))
	}
	if lmHead {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_LMHEAD", "1")
	}
	if lmHeadPrefetch {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_LMHEAD_PREFETCH", "1")
	}
	if lmHeadCandidates > 0 {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_LMHEAD_CANDIDATES", strconv.Itoa(lmHeadCandidates))
	}
	if q80Prefetch {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_Q80_PREFETCH", "1")
	}
	if selectedPrefetch {
		_ = os.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_Q80_SELECTED_PREFETCH", "1")
	}
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "model": s.modelID})
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []map[string]any{{"id": s.modelID, "object": "model", "created": time.Now().Unix(), "owned_by": "go-pherence"}}})
}

func (s *server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleGenerate(w, r, false)
}

func (s *server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleGenerate(w, r, true)
}

func (s *server) handleGenerate(w http.ResponseWriter, r *http.Request, chat bool) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		return
	}
	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	promptIDs, err := s.promptIDs(req, chat)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if req.Stream {
		s.handleStream(w, r, req, promptIDs, chat)
		return
	}
	steps := make([]renderStep, 0)
	opts := s.options(req, func(idx int, snap diffusiongemma.DiffusionStepSnapshot) error {
		if req.ReturnDiffusionSteps {
			steps = append(steps, s.renderStep(idx, snap))
		}
		return nil
	})
	start := time.Now()
	s.mu.Lock()
	res, err := s.engine.GenerateTokenIDs(promptIDs, opts)
	s.mu.Unlock()
	latency := time.Since(start)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, s.response(req, promptIDs, res, steps, latency, chat, false))
}

func (s *server) handleStream(w http.ResponseWriter, r *http.Request, req completionRequest, promptIDs []int, chat bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "streaming unsupported by ResponseWriter"})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	ctx := r.Context()
	steps := make([]renderStep, 0)
	opts := s.options(req, func(idx int, snap diffusiongemma.DiffusionStepSnapshot) error {
		step := s.renderStep(idx, snap)
		steps = append(steps, step)
		return writeSSE(ctx, w, flusher, "diffusion_step", step)
	})
	start := time.Now()
	s.mu.Lock()
	res, err := s.engine.GenerateTokenIDs(promptIDs, opts)
	s.mu.Unlock()
	latency := time.Since(start)
	if err != nil {
		_ = writeSSE(ctx, w, flusher, "error", map[string]any{"error": err.Error()})
		return
	}
	chunk := s.response(req, promptIDs, res, nil, latency, chat, true)
	_ = writeSSEData(ctx, w, flusher, chunk)
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	_ = steps
}

func (s *server) options(req completionRequest, cb func(int, diffusiongemma.DiffusionStepSnapshot) error) diffusiongemma.InferenceOptions {
	maxNew := req.MaxNewTokens
	if maxNew <= 0 {
		maxNew = req.MaxTokens
	}
	if maxNew <= 0 {
		maxNew = s.defaultMaxNew
	}
	canvas := req.CanvasLength
	if canvas <= 0 {
		canvas = s.defaultCanvas
	}
	seed := req.Seed
	if seed == 0 {
		seed = s.defaultSeed
	}
	denoise := req.Denoising
	if denoise == nil && (req.DenoiseSteps > 0 || s.defaultDenoiseStep > 0) {
		cfg := s.model.Denoising
		if req.DenoiseSteps > 0 {
			cfg.MaxDenoisingSteps = req.DenoiseSteps
		} else {
			cfg.MaxDenoisingSteps = s.defaultDenoiseStep
		}
		denoise = &cfg
	}
	return diffusiongemma.InferenceOptions{MaxNewTokens: maxNew, CanvasLength: canvas, Seed: seed, Denoising: denoise, StepCallback: cb}
}

func (s *server) promptIDs(req completionRequest, chat bool) ([]int, error) {
	if len(req.PromptIDs) > 0 {
		return append([]int(nil), req.PromptIDs...), nil
	}
	if s.tok == nil {
		return nil, errors.New("tokenizer unavailable; send prompt_ids")
	}
	if chat || len(req.Messages) > 0 {
		msgs := make([]diffusiongemma.TextChatMessage, 0, len(req.Messages))
		for _, m := range req.Messages {
			msgs = append(msgs, diffusiongemma.TextChatMessage{Role: strings.TrimSpace(m.Role), Content: stringifyContent(m.Content)})
		}
		if len(msgs) == 0 && req.Prompt != nil {
			msgs = append(msgs, diffusiongemma.TextChatMessage{Role: "user", Content: stringifyPrompt(req.Prompt)})
		}
		if len(msgs) == 0 {
			return nil, errors.New("missing messages or prompt_ids")
		}
		if s.model.Tokenizer == nil {
			// Fall back to a plain role/content transcript if metadata is unavailable.
			var b strings.Builder
			for _, m := range msgs {
				fmt.Fprintf(&b, "%s\n%s\n", m.Role, m.Content)
			}
			return s.tok.Encode(b.String()), nil
		}
		specials := s.model.Tokenizer.SpecialTokenIDs(s.model.Processor)
		framed, err := diffusiongemma.BuildTemplateChatPromptIDs(msgs, specials, s.tok.Encode, diffusiongemma.ChatRenderOptions{AddBOS: s.addBOS, AddGenerationPrompt: s.addGenerationPrompt})
		if err != nil {
			return nil, err
		}
		return framed.InputIDs, nil
	}
	prompt := stringifyPrompt(req.Prompt)
	if strings.TrimSpace(prompt) == "" {
		return nil, errors.New("missing prompt or prompt_ids")
	}
	return s.tok.Encode(prompt), nil
}

func (s *server) renderStep(idx int, snap diffusiongemma.DiffusionStepSnapshot) renderStep {
	step := renderStep{GeneratedTokenIndex: idx, Step: snap.Step, Temperature: snap.Temperature, Canvas: append([]int(nil), snap.Canvas...), AcceptedMask: append([]bool(nil), snap.AcceptedMask...), MeanEntropy: snap.MeanEntropy, Stopped: snap.Stopped}
	step.Text = s.decode(snap.Canvas)
	if s.vocab != nil {
		step.Tokens = s.vocab.DecodeIDs(snap.Canvas)
	}
	return step
}

func (s *server) response(req completionRequest, promptIDs []int, res diffusiongemma.InferenceResult, steps []renderStep, latency time.Duration, chat bool, stream bool) completionResponse {
	generated := res.Generated
	idPrefix := "cmpl"
	object := "text_completion"
	choices := []choice{{Index: 0, Text: s.decode(generated), FinishReason: "stop"}}
	if chat {
		idPrefix = "chatcmpl"
		object = "chat.completion"
		if stream {
			object = "chat.completion.chunk"
			choices = []choice{{Index: 0, Delta: &openAIChatChoice{Role: "assistant", Content: s.decode(generated)}, FinishReason: "stop"}}
		} else {
			choices = []choice{{Index: 0, Message: &openAIChatChoice{Role: "assistant", Content: s.decode(generated)}, FinishReason: "stop"}}
		}
	} else if stream {
		object = "completion.chunk"
	}
	return completionResponse{ID: fmt.Sprintf("%s-%d", idPrefix, time.Now().UnixNano()), Object: object, Created: time.Now().Unix(), Model: s.modelID, Choices: choices, Usage: usage{PromptTokens: len(promptIDs), CompletionTokens: len(generated), TotalTokens: len(promptIDs) + len(generated), LatencyMS: latency.Milliseconds()}, PromptTokenIDs: promptIDs, GeneratedTokenIDs: generated, DiffusionSteps: steps, DiffusionStats: s.diffusionStats(req, res, latency)}
}

func (s *server) diffusionStats(req completionRequest, res diffusiongemma.InferenceResult, latency time.Duration) diffusionStats {
	effectiveCanvas := req.CanvasLength
	if effectiveCanvas <= 0 {
		effectiveCanvas = s.defaultCanvas
	}
	if effectiveCanvas <= 0 {
		effectiveCanvas = s.model.Shape.CanvasLength
	}
	effectiveSteps := req.DenoiseSteps
	if effectiveSteps <= 0 {
		effectiveSteps = s.defaultDenoiseStep
	}
	if effectiveSteps <= 0 {
		effectiveSteps = s.model.Denoising.MaxDenoisingSteps
	}
	effectiveMax := req.MaxNewTokens
	if effectiveMax <= 0 {
		effectiveMax = req.MaxTokens
	}
	if effectiveMax <= 0 {
		effectiveMax = s.defaultMaxNew
	}
	if effectiveMax <= 0 && effectiveCanvas > 0 {
		effectiveMax = effectiveCanvas
	}
	var executed, canvasPositions int
	for _, canvas := range res.Canvases {
		executed += len(canvas.Steps)
		canvasPositions += len(canvas.Canvas) * len(canvas.Steps)
	}
	latencySeconds := latency.Seconds()
	stats := diffusionStats{
		RequestedMaxTokens:        firstPositive(req.MaxNewTokens, req.MaxTokens),
		EffectiveMaxTokens:        effectiveMax,
		RequestedCanvasLength:     req.CanvasLength,
		EffectiveCanvasLength:     effectiveCanvas,
		RequestedDenoisingSteps:   req.DenoiseSteps,
		EffectiveDenoisingSteps:   effectiveSteps,
		GeneratedTokens:           len(res.Generated),
		Blocks:                    len(res.Canvases),
		DenoisingStepsExecuted:    executed,
		CanvasPositionEvaluations: canvasPositions,
		LatencyMS:                 latency.Milliseconds(),
	}
	if latencySeconds > 0 {
		stats.GeneratedTokensPerSecond = float64(stats.GeneratedTokens) / latencySeconds
		stats.CanvasPositionsPerSecond = float64(stats.CanvasPositionEvaluations) / latencySeconds
	}
	return stats
}

func firstPositive(values ...int) int {
	for _, v := range values {
		if v > 0 {
			return v
		}
	}
	return 0
}

func (s *server) decode(ids []int) string {
	if len(ids) == 0 {
		return ""
	}
	if s.tok != nil {
		return s.tok.Decode(ids)
	}
	if s.vocab != nil {
		return strings.Join(s.vocab.DecodeIDs(ids), "")
	}
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = fmt.Sprintf("<id:%d>", id)
	}
	return strings.Join(parts, "")
}

func parseIDs(csv string) ([]int, error) {
	csv = strings.TrimSpace(csv)
	if csv == "" {
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
			return nil, fmt.Errorf("invalid token id %q: %w", part, err)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func stringifyPrompt(v interface{}) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	case []interface{}:
		parts := make([]string, 0, len(x))
		for _, item := range x {
			parts = append(parts, stringifyPrompt(item))
		}
		return strings.Join(parts, "\n")
	default:
		b, _ := json.Marshal(x)
		return string(b)
	}
}

func stringifyContent(v interface{}) string { return stringifyPrompt(v) }

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeSSE(ctx context.Context, w http.ResponseWriter, flusher http.Flusher, event string, data interface{}) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func writeSSEData(ctx context.Context, w http.ResponseWriter, flusher http.Flusher, data interface{}) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
