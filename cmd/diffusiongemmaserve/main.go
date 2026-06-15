package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/rcarmo/go-pherence/cmd/internal/dgflags"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/diffusiongemma"
)

// --- OpenAI-compatible types ---

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	Seed        *int64        `json:"seed,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
	Timing  *timingInfo  `json:"timing,omitempty"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message,omitempty"`
	Delta        *chatDelta  `json:"delta,omitempty"`
	FinishReason *string     `json:"finish_reason"`
}

type chatDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type timingInfo struct {
	TotalMs    float64 `json:"total_ms"`
	TokensPerS float64 `json:"tokens_per_s,omitempty"`
}

type modelResponse struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

type modelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// --- Server ---

type server struct {
	mu          sync.Mutex
	eng         *diffusiongemma.Engine
	tok         *tokenizer.Tokenizer
	vocab       *diffusiongemma.Vocab
	meta        *diffusiongemma.Model
	specials    diffusiongemma.SpecialTokenIDs
	opts        inferOpts
	reqCount    int
	expertCache *diffusiongemma.ExpertLRUCache
}

type inferOpts struct {
	canvas       int
	denoiseSteps int
	lmHeadTopK   int
	seed         int64
}

func main() {
	modelDir := flag.String("model", "", "DiffusionGemma metadata/tokenizer model directory")
	fp8Model := flag.String("fp8-model", "", "FP8-dynamic checkpoint directory")
	ggufModel := flag.String("gguf-model", "", "GGUF model file (Q4_K_M, same as llama.cpp)")
	listen := flag.String("listen", ":8080", "HTTP listen address")
	canvas := flag.Int("canvas", 0, "canvas length (0 = model default)")
	denoiseSteps := flag.Int("denoise-steps", 0, "max denoising steps (0 = model default)")
	lmHeadTopK := flag.Int("lm-head-top-k", 0, "LM head top-K (0 = full vocabulary, parity with llama.cpp)")
	residencyBudgetGiB := flag.Float64("residency-budget-gib", 16, "float cache budget in GiB")
	fp8ExpertPrewarmLayers := flag.Int("fp8-expert-prewarm-layers", 9, "pre-upload and pin all FP8 experts for the first N layers to GPU (0 disables)")
	cpuExperts := flag.Bool("cpu-experts", false, "force CPU-only expert MoE (skip GPU LRU cache)")
	seed := flag.Int64("seed", 0, "default RNG seed")
	flag.Parse()

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: diffusiongemmaserve -model PATH [-fp8-model PATH] [-listen :8080]")
		os.Exit(2)
	}

	log.Printf("loading model from %s", *modelDir)
	m, err := diffusiongemma.LoadMetadata(*modelDir)
	if err != nil {
		log.Fatal(err)
	}

	tok, err := tokenizer.Load(*modelDir + "/tokenizer.json")
	if err != nil {
		log.Fatal(err)
	}

	vocab, err := diffusiongemma.LoadVocab(*modelDir)
	if err != nil {
		log.Fatal(err)
	}

	// ── Weight loading: GGUF-only or BF16+FP8 ────────────────────────────
	var weights *diffusiongemma.TextWeights
	var ggufFile *gguf.GGUF // keep alive for mmap

	if *ggufModel != "" {
		// GGUF-only path: use exactly the same Q4_K_M weights as llama.cpp
		log.Printf("loading ALL weights from GGUF: %s", *ggufModel)
		var err error
		ggufFile, err = gguf.Open(*ggufModel)
		if err != nil {
			log.Fatal(err)
		}
		weights, err = diffusiongemma.OpenTextWeightsFromGGUF(ggufFile, m.Shape)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		// BF16 safetensors path (legacy)
		var err error
		weights, err = diffusiongemma.OpenTextWeights(*modelDir, m.Shape)
		if err != nil {
			log.Fatal(err)
		}
	}

	// Residency budget
	residentLayers := 0
	if *residencyBudgetGiB > 0 && *ggufModel == "" {
		// Only estimate residency for safetensors path; GGUF is fully pre-cached
		budgetBytes := int64(*residencyBudgetGiB * 1024 * 1024 * 1024)
		budget := diffusiongemma.EstimateResidencyBudgetFromWeights(weights, true, budgetBytes)
		residentLayers = budget.ResidentLayers
		log.Printf("residency budget %.2f GiB → resident_layers=%d/%d", *residencyBudgetGiB, budget.ResidentLayers, budget.TotalLayers)
	}
	if residentLayers > 0 {
		if err := weights.PreloadLayerRangeLightweight(0, residentLayers); err != nil {
			log.Fatal(err)
		}
	}

	// GPU dispatcher
	gpuDisp := diffusiongemma.GPUDispatcher{
		ResidentLayerPrefix:   residentLayers,
		LMHeadTopK:            *lmHeadTopK,
		CPUExperts:            *cpuExperts || *ggufModel != "",
		FinalLogitSoftcapping: float32(m.Shape.FinalLogitSoftcapping),
	}

	if *ggufModel != "" {
		// GGUF expert index — same Q4_K weights as llama.cpp
		ggufIdx, err := diffusiongemma.BuildGGUFExpertIndex(ggufFile, m.Shape.TextLayers, m.Shape.NumExperts)
		if err != nil {
			log.Fatalf("GGUF expert index build failed: %v", err)
		}
		gpuDisp.GGUFExpertIndex = ggufIdx
		gpuDisp.SkipEviction = true // GGUF TextWeights are fully pre-cached and cannot reload evicted tensors.
		log.Printf("GGUF expert index ready: %d layers, %d experts, intermediate=%d",
			ggufIdx.NumLayers, ggufIdx.NumExperts, ggufIdx.Intermediate)
		if dgflags.GGUFGPULMHeadEnabled() {
			gpuDisp.F32LMHeadChunkSize = dgflags.GGUFGPULMHeadChunkSize()
			gpuDisp.F32LMHeadUseCache = dgflags.GGUFGPULMHeadUseF32Cache()
			source := "Q-row"
			if gpuDisp.F32LMHeadUseCache {
				source = "F32-cache"
			}
			log.Printf("GGUF F32 LM head chunked GPU mode enabled chunk=%d source=%s", gpuDisp.F32LMHeadChunkSize, source)
		}
	} else if *fp8Model != "" {
		log.Printf("loading FP8 weights from %s", *fp8Model)
		fp8Weights, err := diffusiongemma.OpenFP8TextWeights(*fp8Model, m.Shape)
		if err != nil {
			log.Fatal(err)
		}
		if n, err := fp8Weights.EagerLoad(); err == nil && n > 0 {
			log.Printf("eager-loaded FP8: %d bytes", n)
		}
		log.Printf("uploading FP8 layers to GPU")
		gpuModel, err := diffusiongemma.UploadFP8Layers(fp8Weights)
		if err != nil {
			log.Fatal(err)
		}
		gpuDisp.FP8Model = gpuModel
		gpuDisp.FP8Weights = fp8Weights
		lmHeadBuf, lmVocab, lmHidden, err := diffusiongemma.UploadFP8LMHeadBuffer(fp8Weights)
		if err != nil {
			gpuModel.Free()
			log.Fatal(err)
		}
		gpuDisp.FP8LMHead = lmHeadBuf
		gpuDisp.FP8LMHeadVocab = lmVocab
		gpuDisp.FP8LMHeadHidden = lmHidden
		log.Printf("FP8 GPU ready: %d layers; LM head [%d,%d] resident", len(gpuModel.Layers), lmVocab, lmHidden)
		var scEmbedBytes int64
		if dgflags.GPUSelfCondEnabled() {
			log.Printf("WARNING: GPU self-conditioning is experimental and not reference-correct yet")
			scBuf, scVocab, scHidden, err := diffusiongemma.UploadSelfConditioningEmbeddingBuffer(weights)
			if err != nil {
				gpuModel.Free()
				lmHeadBuf.Free()
				log.Fatal(err)
			}
			gpuDisp.SCEmbed = scBuf
			gpuDisp.SCEmbedVocab = scVocab
			gpuDisp.SCEmbedHidden = scHidden
			scEmbedBytes = int64(scVocab) * int64(scHidden) * 4
			log.Printf("SC embedding [%d,%d] resident (%.1f MB)", scVocab, scHidden, float64(scEmbedBytes)/1e6)
		}
		nExperts := m.Shape.NumExperts
		if nExperts <= 0 {
			nExperts = 128
		}
		prewarmedLayers := 0
		if !*cpuExperts {
			expertBudget := dgflags.ExpertCacheBudgetBytes(7200) - int64(lmVocab)*int64(lmHidden)*2 - scEmbedBytes - gpuModel.DenseTransposeBytes()
			if expertBudget > 0 {
				gpuDisp.ExpertCache = diffusiongemma.NewExpertLRUCache(expertBudget)
				if *fp8ExpertPrewarmLayers > 0 {
					layers, experts, err := gpuDisp.ExpertCache.PrewarmLayerPrefix(fp8Weights, *fp8ExpertPrewarmLayers, nExperts)
					if err != nil {
						log.Fatalf("FP8 expert prewarm failed: %v", err)
					}
					prewarmedLayers = layers
					log.Printf("FP8 expert prewarm: layers=%d/%d experts=%d pinned_prefix=%d cache=%s", layers, *fp8ExpertPrewarmLayers, experts, gpuDisp.ExpertCache.PinnedLayerPrefix(nExperts), gpuDisp.ExpertCache.Stats())
				}
			} else if *fp8ExpertPrewarmLayers > 0 {
				log.Printf("skipping FP8 expert prewarm; LM head leaves no expert cache budget")
			}
		}

		// Build pre-indexed FP8 expert lookup for CPU expert path and pinned-prefix overflow fallback.
		if *cpuExperts || *fp8ExpertPrewarmLayers > 0 || prewarmedLayers > 0 {
			idx, err := diffusiongemma.BuildFP8ExpertIndex(fp8Weights, m.Shape.TextLayers, nExperts)
			if err != nil {
				log.Printf("WARNING: FP8 expert index build failed: %v", err)
			} else {
				gpuDisp.ExpertIndex = idx
			}
		}
	}

	expertCacheRef := gpuDisp.ExpertCache

	denoiser, err := diffusiongemma.NewTextDenoiserWithDispatcher(m.Shape, weights, gpuDisp)
	if err != nil {
		log.Fatal(err)
	}
	eng, err := diffusiongemma.NewEngineWithTextWeights(m, weights, denoiser)
	if err != nil {
		log.Fatal(err)
	}

	specials := m.Tokenizer.SpecialTokenIDs(m.Processor)

	serverCanvas := *canvas
	if serverCanvas <= 0 {
		serverCanvas = m.Shape.CanvasLength
	}
	if serverCanvas <= 0 {
		serverCanvas = 256
	}
	serverDenoiseSteps := *denoiseSteps
	if serverDenoiseSteps <= 0 {
		serverDenoiseSteps = m.Denoising.MaxDenoisingSteps
	}
	if serverDenoiseSteps <= 0 {
		serverDenoiseSteps = diffusiongemma.DefaultDenoisingConfig().MaxDenoisingSteps
	}

	srv := &server{
		eng:         eng,
		tok:         tok,
		vocab:       vocab,
		meta:        m,
		specials:    specials,
		expertCache: expertCacheRef,
		opts: inferOpts{
			canvas:       serverCanvas,
			denoiseSteps: serverDenoiseSteps,
			lmHeadTopK:   *lmHeadTopK,
			seed:         *seed,
		},
	}

	http.HandleFunc("/v1/chat/completions", srv.handleChat)
	http.HandleFunc("/v1/models", srv.handleModels)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"ok"}`))
	})

	log.Printf("DiffusionGemma server on %s (canvas=%d, steps=%d, topk=%d)", *listen, serverCanvas, serverDenoiseSteps, *lmHeadTopK)
	log.Fatal(http.ListenAndServe(*listen, nil))
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelResponse{
		Object: "list",
		Data: []modelInfo{{
			ID: "diffusiongemma-26B-A4B-it", Object: "model",
			Created: time.Now().Unix(), OwnedBy: "google",
		}},
	})
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, `{"error":{"message":"method not allowed"}}`, 405)
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"bad json: %s"}}`, err), 400)
		return
	}

	// Convert messages
	textMsgs := make([]diffusiongemma.TextChatMessage, len(req.Messages))
	for i, m := range req.Messages {
		textMsgs[i] = diffusiongemma.TextChatMessage{Role: m.Role, Content: m.Content}
	}

	// Tokenize
	framed, err := diffusiongemma.BuildTemplateChatPromptIDs(textMsgs, s.specials, s.tok.Encode, diffusiongemma.ChatRenderOptions{
		AddBOS: true, AddGenerationPrompt: true,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"template: %s"}}`, err), 500)
		return
	}
	promptIDs, err := diffusiongemma.ExpandImagePlaceholderTokens(framed.InputIDs, s.specials, s.meta.Shape.VisionSoftTokens)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"image prompt: %s"}}`, err), 500)
		return
	}

	maxNew := 0 // 0 lets Engine use generation_config max_new_tokens or canvas length
	if req.MaxTokens > 0 {
		maxNew = req.MaxTokens
	}
	seed := s.opts.seed
	if req.Seed != nil {
		seed = *req.Seed
	}

	denoising := s.meta.Denoising
	denoising.MaxDenoisingSteps = s.opts.denoiseSteps
	denoising.SparseTopK = s.opts.lmHeadTopK

	s.mu.Lock()
	s.reqCount++
	reqNum := s.reqCount
	s.mu.Unlock()

	log.Printf("req #%d: %d msgs, %d prompt tokens, stream=%v", reqNum, len(req.Messages), len(promptIDs), req.Stream)

	if req.Stream {
		s.handleStreamingChat(w, reqNum, promptIDs, maxNew, seed, denoising)
	} else {
		s.handleNonStreamingChat(w, reqNum, promptIDs, maxNew, seed, denoising)
	}
}

func (s *server) handleNonStreamingChat(w http.ResponseWriter, reqNum int, promptIDs []int, maxNew int, seed int64, denoising diffusiongemma.DenoisingConfig) {
	opts := diffusiongemma.InferenceOptions{
		MaxNewTokens: maxNew, CanvasLength: s.opts.canvas,
		Seed: seed, Denoising: &denoising,
	}

	t0 := time.Now()
	s.mu.Lock()
	res, err := s.eng.GenerateTokenIDs(promptIDs, opts)
	s.mu.Unlock()
	elapsed := time.Since(t0)

	if err != nil {
		log.Printf("req #%d: error: %s", reqNum, err)
		http.Error(w, fmt.Sprintf(`{"error":{"message":"%s"}}`, err), 500)
		return
	}

	text := cleanOutput(s.tok.Decode(res.Generated))
	tokPerSec := float64(len(res.Generated)) / elapsed.Seconds()
	log.Printf("req #%d: %d tok in %.1fs (%.1f t/s) → %q", reqNum, len(res.Generated), elapsed.Seconds(), tokPerSec, text)
	if s.expertCache != nil {
		s.mu.Lock()
		cacheStats := s.expertCache.Stats()
		s.expertCache.ResetCounters()
		s.mu.Unlock()
		log.Printf("req #%d: cache %s", reqNum, cacheStats)
	}

	stop := "stop"
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(chatResponse{
		ID: fmt.Sprintf("chatcmpl-%d", reqNum), Object: "chat.completion",
		Created: time.Now().Unix(), Model: "diffusiongemma-26B-A4B-it",
		Choices: []chatChoice{{
			Index: 0, Message: chatMessage{Role: "assistant", Content: text},
			FinishReason: &stop,
		}},
		Usage: chatUsage{
			PromptTokens: len(promptIDs), CompletionTokens: len(res.Generated),
			TotalTokens: len(promptIDs) + len(res.Generated),
		},
		Timing: &timingInfo{TotalMs: float64(elapsed.Milliseconds()), TokensPerS: tokPerSec},
	})
}

func (s *server) handleStreamingChat(w http.ResponseWriter, reqNum int, promptIDs []int, maxNew int, seed int64, denoising diffusiongemma.DenoisingConfig) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":{"message":"streaming not supported"}}`, 500)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	id := fmt.Sprintf("chatcmpl-%d", reqNum)
	model := "diffusiongemma-26B-A4B-it"
	created := time.Now().Unix()

	// Send initial role delta
	sendSSE(w, flusher, id, model, created, &chatDelta{Role: "assistant"}, nil)

	// Track what we've already sent so we can emit incremental deltas
	var sentText string
	tok := s.tok

	t0 := time.Now()

	opts := diffusiongemma.InferenceOptions{
		MaxNewTokens: maxNew, CanvasLength: s.opts.canvas,
		Seed: seed, Denoising: &denoising,
		OnProgress: func(ev diffusiongemma.ProgressEvent) {
			switch ev.Type {
			case "step":
				// Emit a comment with step progress so the client sees activity
				step := ev.Step
				if step != nil {
					comment := fmt.Sprintf("canvas %d step %d: entropy=%.4f stopped=%v", ev.CanvasIndex, step.Step, step.MeanEntropy, step.Stopped)
					fmt.Fprintf(w, ": %s\n\n", comment)
					flusher.Flush()
				}
			case "canvas":
				// Don't emit token delta here — we'll do it after GenerateTokenIDs returns
				// because we need the full generated token list to decode properly.
			}
		},
	}

	s.mu.Lock()
	res, err := s.eng.GenerateTokenIDs(promptIDs, opts)
	s.mu.Unlock()
	elapsed := time.Since(t0)

	if err != nil {
		log.Printf("req #%d: stream error: %s", reqNum, err)
		errMsg := fmt.Sprintf(`{"error":{"message":"%s"}}`, err)
		fmt.Fprintf(w, "data: %s\n\n", errMsg)
		flusher.Flush()
		return
	}

	// Decode full output and send the text as a content delta
	fullText := cleanOutput(tok.Decode(res.Generated))
	if newContent := strings.TrimPrefix(fullText, sentText); newContent != "" {
		sendSSE(w, flusher, id, model, created, &chatDelta{Content: newContent}, nil)
		sentText = fullText
	}

	// Send final chunk with finish_reason
	stop := "stop"
	sendSSE(w, flusher, id, model, created, &chatDelta{}, &stop)

	// Send [DONE]
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()

	tokPerSec := float64(len(res.Generated)) / elapsed.Seconds()
	if s.expertCache != nil {
		s.mu.Lock()
		cacheStats := s.expertCache.Stats()
		s.expertCache.ResetCounters()
		s.mu.Unlock()
		log.Printf("req #%d: cache %s", reqNum, cacheStats)
	}
	log.Printf("req #%d: %d tok in %.1fs (%.1f t/s) streamed → %q", reqNum, len(res.Generated), elapsed.Seconds(), tokPerSec, fullText)
}

func sendSSE(w http.ResponseWriter, f http.Flusher, id, model string, created int64, delta *chatDelta, finishReason *string) {
	chunk := chatResponse{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []chatChoice{{Index: 0, Delta: delta, FinishReason: finishReason}},
	}
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	f.Flush()
}

// cleanOutput strips special tokens and thinking prefixes.
func cleanOutput(text string) string {
	for _, t := range []string{"<eos>", "<|channel>", "<channel|>", "<|turn>", "<turn|>"} {
		text = strings.ReplaceAll(text, t, "")
	}
	text = strings.TrimPrefix(text, "thoughtthought")
	text = strings.TrimPrefix(text, "thought")
	text = strings.TrimSpace(text)
	return text
}
