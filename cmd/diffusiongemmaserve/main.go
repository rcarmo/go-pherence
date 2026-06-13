package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

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
	modelDir := flag.String("model", "", "DiffusionGemma BF16 model directory")
	fp8Model := flag.String("fp8-model", "", "FP8-dynamic checkpoint directory")
	listen := flag.String("listen", ":8080", "HTTP listen address")
	canvas := flag.Int("canvas", 16, "canvas length")
	denoiseSteps := flag.Int("denoise-steps", 2, "max denoising steps")
	lmHeadTopK := flag.Int("lm-head-top-k", 512, "LM head top-K")
	residencyBudgetGiB := flag.Float64("residency-budget-gib", 16, "float cache budget in GiB")
	seed := flag.Int64("seed", 42, "default RNG seed")
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

	weights, err := diffusiongemma.OpenTextWeights(*modelDir, m.Shape)
	if err != nil {
		log.Fatal(err)
	}

	// Residency budget
	residentLayers := 0
	if *residencyBudgetGiB > 0 {
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

	// GPU dispatcher with FP8
	gpuDisp := diffusiongemma.GPUDispatcher{
		ResidentLayerPrefix: residentLayers,
		LMHeadTopK:          *lmHeadTopK,
	}
	if *fp8Model != "" {
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
		gpuDisp.ExpertCache = diffusiongemma.NewExpertLRUCache(7200 * 1024 * 1024)
		log.Printf("FP8 GPU ready: %d layers", len(gpuModel.Layers))
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

	srv := &server{
		eng:         eng,
		tok:         tok,
		vocab:       vocab,
		meta:        m,
		specials:    specials,
		expertCache: expertCacheRef,
		opts: inferOpts{
			canvas:       *canvas,
			denoiseSteps: *denoiseSteps,
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

	log.Printf("DiffusionGemma server on %s (canvas=%d, steps=%d, topk=%d)", *listen, *canvas, *denoiseSteps, *lmHeadTopK)
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
	promptIDs := framed.InputIDs

	maxNew := 16
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
		log.Printf("req #%d: cache %s", reqNum, s.expertCache.Stats())
		s.expertCache.ResetCounters()
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
