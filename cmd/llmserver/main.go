package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

// OpenAI API types

type ChatMessage struct {
	Role             string `json:"role"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature float32       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   Usage                  `json:"usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type StreamDelta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

type ModelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelListResponse struct {
	Object string        `json:"object"`
	Data   []ModelObject `json:"data"`
}

// Server

type Server struct {
	cpuModel    *model.LlamaModel
	gpuModel    *model.GPUModel
	tok         *tokenizer.Tokenizer
	mu          sync.Mutex
	modelID     string
	modelPath   string
	presets     map[string]ModelPreset
	created     int64
	maxCtx      int
	useGPU      bool
	gpuLayers   int
	speculative bool
	cacheTypeK  string
	cacheTypeV  string
	kvResidual  int
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	models := []ModelObject{{
		ID:      s.modelID,
		Object:  "model",
		Created: s.created,
		OwnedBy: "local",
	}}
	for id := range s.presets {
		if id == s.modelID {
			continue
		}
		models = append(models, ModelObject{ID: id, Object: "model", Created: s.created, OwnedBy: "local"})
	}
	resp := ModelListResponse{
		Object: "list",
		Data:   models,
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("models response encode failed: %v", err)
	}
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	defer r.Body.Close()
	dec := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	dec.DisallowUnknownFields()
	var req ChatCompletionRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	maxTokens := req.MaxTokens
	if maxTokens < 0 {
		http.Error(w, "max_tokens must be non-negative", http.StatusBadRequest)
		return
	}
	if maxTokens == 0 {
		// Reasoning models such as Qwen3 can spend hundreds or thousands of
		// tokens in <think> before producing user-visible content.
		maxTokens = 4096
	}

	if len(req.Messages) == 0 {
		http.Error(w, "messages must not be empty", http.StatusBadRequest)
		return
	}

	// Build prompt from messages
	var parts []string
	for _, msg := range req.Messages {
		parts = append(parts, msg.Content)
	}
	prompt := strings.Join(parts, "\n")

	s.mu.Lock()
	defer s.mu.Unlock()

	if req.Model != "" && req.Model != s.modelID {
		if err := s.switchModelLocked(req.Model); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}

	ids := s.tok.Encode(prompt)

	if req.Stream {
		s.streamResponse(w, r, ids, maxTokens)
	} else {
		s.nonStreamResponse(w, ids, maxTokens)
	}
}

func (s *Server) generate(ids []int, maxTokens int, emit func(token int, text string) bool) (int, string) {
	var output []int
	if s.gpuModel != nil {
		output = s.gpuModel.Generate(ids, maxTokens)
	} else if s.speculative {
		output = s.cpuModel.GenerateSpeculative(ids, maxTokens, model.SpeculativeConfigFromEnv())
	} else {
		output = s.cpuModel.Generate(ids, maxTokens)
	}

	// Find generated tokens (after prompt)
	promptLen := len(ids)
	// The Generate function includes prompt in output for CPU, but not for GPU
	// Normalize: extract only the new tokens
	generated := output
	if len(output) > promptLen {
		// CPU path includes prompt
		generated = output[promptLen:]
	}

	var out strings.Builder
	count := 0
	for _, tok := range generated {
		if tok < 0 || tok >= len(s.tok.InvVocab) {
			break
		}
		text := s.tok.InvVocab[tok]
		// Stop on EOS-like tokens
		if text == "<eos>" || text == "</s>" || tok == 0 {
			break
		}
		out.WriteString(text)
		count++
		if emit != nil && !emit(tok, text) {
			break
		}
	}
	return count, out.String()
}

func (s *Server) nonStreamResponse(w http.ResponseWriter, ids []int, maxTokens int) {
	generated, text := s.generate(ids, maxTokens, nil)
	content, reasoning := splitReasoningText(text)
	finishReason := "stop"
	if generated >= maxTokens {
		finishReason = "length"
	}

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.modelID,
		Choices: []ChatCompletionChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: content, ReasoningContent: reasoning},
			FinishReason: finishReason,
		}},
		Usage: Usage{
			PromptTokens:     len(ids),
			CompletionTokens: generated,
			TotalTokens:      len(ids) + generated,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("chat response encode failed: %v", err)
	}
}

func (s *Server) streamResponse(w http.ResponseWriter, r *http.Request, ids []int, maxTokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	// Initial chunk with role
	if !writeSSE(w, flusher, StreamChunk{
		ID: id, Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: s.modelID,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Role: "assistant"}}},
	}) {
		return
	}

	generated := 0
	finishReason := "stop"
	var split reasoningSplitter

	s.generate(ids, maxTokens, func(tok int, text string) bool {
		select {
		case <-r.Context().Done():
			return false
		default:
		}
		content, reasoning := split.Push(text)
		if content != "" || reasoning != "" {
			if !writeSSE(w, flusher, StreamChunk{
				ID: id, Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: s.modelID,
				Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Content: content, ReasoningContent: reasoning}}},
			}) {
				return false
			}
		}
		generated++
		return true
	})

	if content, reasoning := split.Flush(); content != "" || reasoning != "" {
		if !writeSSE(w, flusher, StreamChunk{
			ID: id, Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: s.modelID,
			Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Content: content, ReasoningContent: reasoning}}},
		}) {
			return
		}
	}

	if generated >= maxTokens {
		finishReason = "length"
	}

	if !writeSSE(w, flusher, StreamChunk{
		ID: id, Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: s.modelID,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{}, FinishReason: &finishReason}},
	}) {
		return
	}

	if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
		log.Printf("stream done write failed: %v", err)
		return
	}
	flusher.Flush()
}

func writeSSE(w io.Writer, flusher http.Flusher, chunk StreamChunk) bool {
	data, err := json.Marshal(chunk)
	if err != nil {
		log.Printf("stream chunk marshal failed: %v", err)
		return false
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		log.Printf("stream chunk write failed: %v", err)
		return false
	}
	flusher.Flush()
	return true
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	status := map[string]any{"status": "ok", "model": s.modelID, "models": len(s.presets) + 1, "gpu": s.gpuModel != nil, "ctx_size": s.maxCtx}
	if s.cacheTypeK != "" || s.cacheTypeV != "" || s.kvResidual >= 0 {
		status["turboquant"] = s.turboQuantHealthLocked()
	}
	if err := json.NewEncoder(w).Encode(status); err != nil {
		log.Printf("health response encode failed: %v", err)
	}
}

func (s *Server) turboQuantHealthLocked() map[string]any {
	out := map[string]any{"cache_type_k": s.cacheTypeK, "cache_type_v": s.cacheTypeV, "residual_window": s.kvResidual}
	cfg, enabled, err := kv.TurboQuantConfigFromCacheTypes(s.cacheTypeK, s.cacheTypeV, s.kvResidual)
	if err != nil {
		out["error"] = err.Error()
		return out
	}
	out["enabled"] = enabled
	out["key_bits"] = cfg.KeyBits
	out["value_bits"] = cfg.ValueBits
	if s.cpuModel != nil {
		kvHeads := s.cpuModel.Config.NumKVHeads
		if kvHeads == 0 {
			kvHeads = s.cpuModel.Config.NumHeads
		}
		headDim := s.cpuModel.Config.HeadDim
		if headDim == 0 && s.cpuModel.Config.NumHeads > 0 {
			headDim = s.cpuModel.Config.HiddenSize / s.cpuModel.Config.NumHeads
		}
		layers := s.cpuModel.Config.NumLayers
		maxSeq := s.maxCtx
		if maxSeq <= 0 {
			maxSeq = s.cpuModel.Config.MaxSeqLen
		}
		kvDim := kvHeads * headDim
		out["layers"] = layers
		out["kv_heads"] = kvHeads
		out["head_dim"] = headDim
		out["kv_dim"] = kvDim
		out["max_seq"] = maxSeq
		out["full_kv_bytes"] = int64(layers) * int64(maxSeq) * int64(kvDim) * 2 * 4
	}
	return out
}

func (s *Server) switchModelLocked(id string) error {
	preset, ok := s.presets[id]
	if !ok {
		return fmt.Errorf("unknown model %q", id)
	}
	log.Printf("Switching model from %s to %s (%s)", s.modelID, id, preset.Path)
	m, tok, gpu, err := loadRuntimeModelWithKV(preset.Path, s.useGPU, preset.GPULayers, preset.CacheTypeK, preset.CacheTypeV, -1)
	if err != nil {
		return fmt.Errorf("load model %q: %w", id, err)
	}
	if s.gpuModel != nil {
		s.gpuModel.Close()
	}
	s.cpuModel = m
	s.gpuModel = gpu
	s.tok = tok
	s.modelID = id
	s.modelPath = preset.Path
	s.cacheTypeK = preset.CacheTypeK
	s.cacheTypeV = preset.CacheTypeV
	s.kvResidual = -1
	if preset.CtxSize > 0 {
		s.maxCtx = preset.CtxSize
	}
	return nil
}

func loadRuntimeModel(path string, useGPU bool, gpuLayers int) (*model.LlamaModel, *tokenizer.Tokenizer, *model.GPUModel, error) {
	return loadRuntimeModelWithKV(path, useGPU, gpuLayers, "", "", -1)
}

func loadRuntimeModelWithKV(path string, useGPU bool, gpuLayers int, cacheTypeK, cacheTypeV string, residualWindow int) (*model.LlamaModel, *tokenizer.Tokenizer, *model.GPUModel, error) {
	m, err := model.LoadLlama(path)
	if err != nil {
		return nil, nil, nil, err
	}
	tok, err := tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		return nil, nil, nil, err
	}
	m.Tok = tok
	if cacheTypeK != "" || cacheTypeV != "" || residualWindow >= 0 {
		cfg, enabled, err := kv.TurboQuantConfigFromCacheTypes(cacheTypeK, cacheTypeV, residualWindow)
		if err != nil {
			return nil, nil, nil, err
		}
		if enabled || residualWindow >= 0 {
			m.EnableTurboQuant = enabled
			m.TurboQuantConfig = &cfg
		}
	}
	var gpu *model.GPUModel
	if useGPU {
		g, err := model.LoadGPUModelWithLayers(m, gpuLayers)
		if err != nil {
			log.Printf("GPU failed for %s: %v (using CPU)", path, err)
		} else {
			g.CPU.Tok = tok
			gpu = g
		}
	}
	return m, tok, gpu, nil
}

func main() {
	dir := flag.String("model", "", "model directory")
	modelPresets := flag.String("model-presets", "", "llama.cpp-compatible models.ini preset file")
	listen := flag.String("listen", ":8080", "address to listen on")
	useGPU := flag.Bool("gpu", false, "use GPU")
	gpuLayers := flag.Int("gpu-layers", 0, "number of layers on GPU (0=all)")
	threads := flag.Int("threads", 4, "decode CPU threads hint for llama.cpp-compatible deployments")
	batchSize := flag.Int("batch-size", 512, "prefill/ubatch size hint for llama.cpp-compatible deployments")
	ctxSize := flag.Int("ctx-size", 32768, "maximum context size advertised by the server")
	cacheTypeK := flag.String("cache-type-k", "", "KV cache key quantization hint (turbo4, q8_0, f16)")
	cacheTypeV := flag.String("cache-type-v", "", "KV cache value quantization hint (turbo2, q4_0, f16)")
	kvResidualWindow := flag.Int("kv-residual-window", -1, "TurboQuant full-precision residual KV tokens (-1=default)")
	turboQuant := flag.Bool("turbo-quant", false, "enable TurboQuant KV cache compression on CPU backend")
	speculative := flag.Bool("speculative", false, "enable opt-in stock-weight speculative decoding path (CPU backend)")
	specBlock := flag.Int("speculative-block", 8, "speculative proposal block size")
	specNGram := flag.Int("speculative-ngram", 4, "speculative prompt-lookup n-gram size")
	specMinProposal := flag.Int("speculative-min-proposal", 2, "minimum proposal length before verifier attempt")
	specProposer := flag.String("speculative-proposer", "prompt", "speculative proposer: prompt, repeat-last, none")
	specBackend := flag.String("speculative-backend", "replay", "speculative verifier backend: replay")
	specDebug := flag.Bool("speculative-debug", false, "print speculative proposal/acceptance stats")
	eagerLoad := flag.Bool("eager-load", false, "pre-fault mmap'd model weights at startup")
	flag.Parse()

	presets := make(map[string]ModelPreset)
	if *modelPresets != "" {
		parsed, err := ParseModelPresets(*modelPresets)
		if err != nil {
			log.Fatalf("model presets failed: %v", err)
		}
		for _, preset := range parsed {
			presets[preset.ID] = preset
		}
		if *dir == "" && len(parsed) > 0 {
			*dir = parsed[0].Path
		}
	}
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: llmserver -model <dir> [-model-presets models.ini] [-listen :8080] [-gpu]")
		os.Exit(1)
	}

	if *eagerLoad {
		os.Setenv("GO_PHERENCE_EAGER_LOAD", "1")
	}
	if *useGPU {
		model.ForceOnTheFly = true
		if *turboQuant {
			log.Printf("warning: --turbo-quant currently applies to the CPU backend only")
		}
		if *speculative {
			log.Printf("warning: --speculative currently applies to the CPU backend only")
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

	log.Printf("Loading model from %s...", *dir)
	t0 := time.Now()
	m, tok, gpu, err := loadRuntimeModelWithKV(*dir, *useGPU, *gpuLayers, *cacheTypeK, *cacheTypeV, *kvResidualWindow)
	if err != nil {
		log.Fatalf("Load failed: %v", err)
	}
	m.EnableTurboQuant = *turboQuant
	log.Printf("Model loaded in %.1fs (%d layers, h=%d)", time.Since(t0).Seconds(),
		m.Config.NumLayers, m.Config.HiddenSize)

	modelID := filepath.Base(*dir)
	for id, preset := range presets {
		if preset.Path == *dir {
			modelID = id
			break
		}
	}
	log.Printf("Runtime hints: threads=%d batch_size=%d ctx_size=%d cache_type_k=%q cache_type_v=%q kv_residual_window=%d presets=%d", *threads, *batchSize, *ctxSize, *cacheTypeK, *cacheTypeV, *kvResidualWindow, len(presets))
	srv := &Server{cpuModel: m, gpuModel: gpu, tok: tok, modelID: modelID, modelPath: *dir, presets: presets, created: time.Now().Unix(), maxCtx: *ctxSize, useGPU: *useGPU, gpuLayers: *gpuLayers, speculative: *speculative, cacheTypeK: *cacheTypeK, cacheTypeV: *cacheTypeV, kvResidual: *kvResidualWindow}
	if gpu != nil {
		defer gpu.Close()
		defer nvidia.Shutdown()
		log.Printf("GPU model ready")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/v1/models", srv.handleModels)
	mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)

	log.Printf("Listening on %s", *listen)
	log.Printf("  GET  /health")
	log.Printf("  POST /v1/chat/completions")
	log.Printf("  GET  /v1/models")
	if err := http.ListenAndServe(*listen, mux); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
