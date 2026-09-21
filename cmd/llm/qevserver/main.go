package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/qev"
	"github.com/rcarmo/go-pherence/webui"
)

const defaultModelID = "gemma-4-12b-it-qev"

type options struct {
	modelPath    string
	tokenizerDir string
	modelID      string
	listen       string
	backend      string
	verify       bool
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("qevserver", flag.ContinueOnError)
	var cfg options
	fs.StringVar(&cfg.modelPath, "model", "", "pinned Gemma 4 12B GGUF file")
	fs.StringVar(&cfg.tokenizerDir, "tokenizer-dir", "", "pinned Gemma 4 tokenizer sidecar directory")
	fs.StringVar(&cfg.modelID, "model-id", defaultModelID, "API model identifier")
	fs.StringVar(&cfg.listen, "listen", "127.0.0.1:8080", "HTTP listen address")
	fs.StringVar(&cfg.backend, "backend", "nvidia", "scoring backend: nvidia or simd")
	fs.BoolVar(&cfg.verify, "verify-artifacts", true, "verify exact QEV v1 model/tokenizer SHA-256 pins")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if cfg.modelPath == "" || cfg.tokenizerDir == "" {
		return fmt.Errorf("-model and -tokenizer-dir are required")
	}
	if cfg.modelID == "" {
		return fmt.Errorf("-model-id must not be empty")
	}
	if cfg.backend != "nvidia" && cfg.backend != "simd" {
		return fmt.Errorf("-backend must be nvidia or simd")
	}
	if cfg.verify {
		log.Printf("qevserver: verifying pinned artifacts")
		if err := qev.VerifyV1Artifacts(cfg.modelPath, cfg.tokenizerDir); err != nil {
			return fmt.Errorf("artifact verification: %w", err)
		}
	}

	log.Printf("qevserver: loading tokenizer")
	tok, err := tokenizer.Load(filepath.Join(cfg.tokenizerDir, "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}
	log.Printf("qevserver: loading %s", filepath.Base(cfg.modelPath))
	m, err := model.LoadGemma4GGUFAsLlama(cfg.modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	m.Tok = tok
	if tok.VocabSize() != m.Config.VocabSize {
		return fmt.Errorf("tokenizer vocab=%d, model vocab=%d", tok.VocabSize(), m.Config.VocabSize)
	}

	var scorer qev.ContextScorer
	var nvidiaScorer *qev.Gemma4NVIDIAScorer
	device := "CPU"
	residentBytes := int64(0)
	var gpu *model.Gemma4NVIDIA
	if cfg.backend == "nvidia" {
		gpu, err = model.NewGemma4NVIDIA(m)
		if err != nil {
			return fmt.Errorf("initialize NVIDIA scorer: %w", err)
		}
		defer gpu.Close()
		defer nvidia.Shutdown()
		device, residentBytes = gpu.DeviceName(), gpu.ResidentBytes()
		nvidiaScorer = &qev.Gemma4NVIDIAScorer{Model: m, GPU: gpu}
		defer nvidiaScorer.Close()
		scorer = nvidiaScorer
	} else {
		scorer = &qev.Gemma4SIMDBatchScorer{Model: m, Backend: model.InferenceBackendSIMD}
	}
	engine := &qev.Engine{Tokenizer: tok, Scorer: scorer, BOSToken: m.Config.BOSTokenID}
	decision := &qev.Handler{Engine: engine, ModelID: cfg.modelID}
	mux := http.NewServeMux()
	mux.Handle("/v1/decision", decision)
	webui.RegisterQEV(mux, webui.QEVConfig{ModelID: cfg.modelID, Backend: cfg.backend, Device: device, ResidentBytes: residentBytes, MaxContexts: qev.MaxContexts, MaxFields: qev.MaxFields, MaxCandidates: qev.MaxCandidates, Busy: decision.Busy})
	server := &http.Server{Addr: cfg.listen, Handler: logRequests(mux), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 32 << 10}

	errCh := make(chan error, 1)
	go func() { errCh <- server.ListenAndServe() }()
	log.Printf("qevserver: listening on http://%s/qev backend=%s device=%s resident_bytes=%d", cfg.listen, cfg.backend, device, residentBytes)
	select {
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdown); err != nil {
			return err
		}
		return nil
	}
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(started).Round(time.Millisecond))
	})
}
