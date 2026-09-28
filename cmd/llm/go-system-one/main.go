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
	gosystemone "github.com/rcarmo/go-pherence/model/gosystemone"
	"github.com/rcarmo/go-pherence/webui"
)

const defaultModelID = "gemma-4-12b-it-go-system-one"

type options struct {
	modelPath    string
	tokenizerDir string
	modelID      string
	listen       string
	backend      string
	verify       bool
	packedRows   int
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func parseOptions(args []string) (options, error) {
	fs := flag.NewFlagSet("go-system-one", flag.ContinueOnError)
	var cfg options
	fs.StringVar(&cfg.modelPath, "model", "", "pinned Gemma 4 12B GGUF file")
	fs.StringVar(&cfg.tokenizerDir, "tokenizer-dir", "", "pinned Gemma 4 tokenizer sidecar directory")
	fs.StringVar(&cfg.modelID, "model-id", defaultModelID, "API model identifier")
	fs.StringVar(&cfg.listen, "listen", "127.0.0.1:8080", "HTTP listen address")
	fs.StringVar(&cfg.backend, "backend", "nvidia", "scoring backend: nvidia or simd")
	fs.BoolVar(&cfg.verify, "verify-artifacts", true, "verify exact Go System One v1 model/tokenizer SHA-256 pins")
	fs.IntVar(&cfg.packedRows, "packed-token-rows", -1, "NVIDIA packed token budget (-1 automatic: 512 on NVIDIA, 0 on SIMD; 0 forces serial)")
	if err := fs.Parse(args); err != nil {
		return cfg, err
	}
	if cfg.modelPath == "" || cfg.tokenizerDir == "" {
		return cfg, fmt.Errorf("-model and -tokenizer-dir are required")
	}
	if cfg.modelID == "" {
		return cfg, fmt.Errorf("-model-id must not be empty")
	}
	if cfg.backend != "nvidia" && cfg.backend != "simd" {
		return cfg, fmt.Errorf("-backend must be nvidia or simd")
	}
	if cfg.packedRows == -1 {
		cfg.packedRows = 0
		if cfg.backend == "nvidia" {
			cfg.packedRows = model.Gemma4PackedRows
		}
	}
	if cfg.packedRows < 0 || cfg.packedRows > model.Gemma4PackedRows || (cfg.packedRows > 0 && cfg.backend != "nvidia") {
		return cfg, fmt.Errorf("-packed-token-rows requires nvidia and must be automatic (-1) or 0..%d", model.Gemma4PackedRows)
	}
	return cfg, nil
}

func run(ctx context.Context, args []string) error {
	cfg, err := parseOptions(args)
	if err != nil {
		return err
	}
	if cfg.verify {
		log.Printf("go-system-one: verifying pinned artifacts")
		if err := gosystemone.VerifyV1Artifacts(cfg.modelPath, cfg.tokenizerDir); err != nil {
			return fmt.Errorf("artifact verification: %w", err)
		}
	}

	log.Printf("go-system-one: loading tokenizer")
	tok, err := tokenizer.Load(filepath.Join(cfg.tokenizerDir, "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}
	log.Printf("go-system-one: loading %s", filepath.Base(cfg.modelPath))
	m, err := model.LoadGemma4GGUFAsLlama(cfg.modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	m.Tok = tok
	if tok.VocabSize() != m.Config.VocabSize {
		return fmt.Errorf("tokenizer vocab=%d, model vocab=%d", tok.VocabSize(), m.Config.VocabSize)
	}

	var scorer gosystemone.ContextScorer
	var nvidiaScorer *gosystemone.Gemma4NVIDIAScorer
	device := "CPU"
	residentBytes := int64(0)
	var gpu *model.Gemma4NVIDIA
	if cfg.backend == "nvidia" {
		gpu, err = model.NewGemma4NVIDIA(m)
		if err != nil {
			return fmt.Errorf("initialize NVIDIA scorer: %w", err)
		}
		defer nvidia.Shutdown()
		defer gpu.Close()
		device, residentBytes = gpu.DeviceName(), gpu.ResidentBytes()
		nvidiaScorer = &gosystemone.Gemma4NVIDIAScorer{Model: m, GPU: gpu, PackedTokenRows: cfg.packedRows}
		defer nvidiaScorer.Close()
		scorer = nvidiaScorer
	} else {
		scorer = &gosystemone.Gemma4SIMDBatchScorer{Model: m, Backend: model.InferenceBackendSIMD}
	}
	engine := &gosystemone.Engine{Tokenizer: tok, Scorer: scorer, BOSToken: m.Config.BOSTokenID}
	decision := &gosystemone.Handler{Engine: engine, ModelID: cfg.modelID}
	mux := http.NewServeMux()
	mux.Handle("/v1/decision", decision)
	mux.HandleFunc("/v1/systemone", decision.ServeSystemOne)
	webui.RegisterGoSystemOne(mux, webui.GoSystemOneConfig{ModelID: cfg.modelID, Backend: cfg.backend, Device: device, ResidentBytes: residentBytes, MaxContexts: gosystemone.MaxContexts, MaxFields: gosystemone.MaxFields, MaxCandidates: gosystemone.MaxCandidates, Busy: decision.Busy})
	server := &http.Server{Addr: cfg.listen, Handler: logRequests(mux), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 32 << 10}

	errCh := make(chan error, 1)
	go func() { errCh <- server.ListenAndServe() }()
	log.Printf("go-system-one: listening on http://%s/go-system-one backend=%s device=%s resident_bytes=%d packed_token_rows=%d", cfg.listen, cfg.backend, device, residentBytes, cfg.packedRows)
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
