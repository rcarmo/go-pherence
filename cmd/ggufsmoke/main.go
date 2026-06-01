package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	path := flag.String("model", "", "GGUF model path")
	loadOnly := flag.Bool("load-only", false, "load model and exit")
	maxNew := flag.Int("max-new", 0, "run greedy generation for N tokens from -prompt-ids")
	promptIDsCSV := flag.String("prompt-ids", "0", "comma-separated prompt token IDs for forward/generation smoke")
	quant := flag.Bool("ggml-quant", true, "keep quantized GGUF matrices instead of full F32 expansion")
	cacheTypeK := flag.String("cache-type-k", "", "native TurboQuant key cache type (turbo4, q8_0, f16)")
	cacheTypeV := flag.String("cache-type-v", "", "native TurboQuant value cache type (turbo2, q4_0, f16)")
	kvResidualWindow := flag.Int("kv-residual-window", -1, "native TurboQuant residual window")
	flag.Parse()
	if *path == "" {
		fmt.Fprintln(os.Stderr, "usage: ggufsmoke -model model.gguf [-load-only]")
		os.Exit(2)
	}
	if *quant {
		os.Setenv("GO_PHERENCE_GGML_QUANT", "1")
	}
	t0 := time.Now()
	m, err := model.LoadGGUFLlama(*path, k3.SIMDBackend{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "ggufsmoke: load failed: %v\n", err)
		os.Exit(1)
	}
	reap := 0.0
	if m.REAP != nil {
		reap = m.REAP.PruneRatio
	}
	fmt.Printf("loaded architecture=%s layers=%d hidden=%d experts=%d active=%d qwennext=%v reap=%.2f in %.2fs\n", m.Config.Architecture, m.Config.NumLayers, m.Config.HiddenSize, m.Config.NumExperts, m.Config.NumExpertsPerTok, m.Config.IsQwenNextHybridGGUF(), reap, time.Since(t0).Seconds())
	if *cacheTypeK != "" || *cacheTypeV != "" || *kvResidualWindow >= 0 {
		plan, err := m.TurboQuantPlan(*cacheTypeK, *cacheTypeV, *kvResidualWindow)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: turboquant plan failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("turboquant enabled=%v key_bits=%d value_bits=%d residual=%d layers=%d kv_dim=%d\n", plan.Enabled, plan.KeyBits, plan.ValueBits, plan.ResidualWindow, plan.Layers, plan.KVDim)
	}
	if *loadOnly {
		return
	}
	promptIDs, err := parsePromptIDs(*promptIDsCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ggufsmoke: bad -prompt-ids: %v\n", err)
		os.Exit(2)
	}
	if *maxNew > 0 {
		ids, err := m.Generate(promptIDs, *maxNew)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: generate failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("generated=%v\n", ids)
		return
	}
	state := m.NewForwardState()
	kvDim := m.Config.NumKVHeads * m.Config.HeadDim
	kvK := make([][]float32, m.Config.NumLayers)
	kvV := make([][]float32, m.Config.NumLayers)
	for i := range kvK {
		kvK[i] = make([]float32, kvDim)
		kvV[i] = make([]float32, kvDim)
	}
	logits := m.ForwardState(state, promptIDs[0], 0, kvK, kvV)
	fmt.Printf("forward logits=%d\n", len(logits))
}

func parsePromptIDs(csv string) ([]int, error) {
	parts := strings.Split(csv, ",")
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		id, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		if id < 0 {
			return nil, fmt.Errorf("negative token id %d", id)
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("no prompt ids")
	}
	return ids, nil
}
