package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	path := flag.String("model", "", "GGUF model path")
	loadOnly := flag.Bool("load-only", false, "load model and exit")
	quant := flag.Bool("ggml-quant", true, "keep quantized GGUF matrices instead of full F32 expansion")
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
	fmt.Printf("loaded architecture=%s layers=%d hidden=%d experts=%d active=%d qwennext=%v in %.2fs\n", m.Config.Architecture, m.Config.NumLayers, m.Config.HiddenSize, m.Config.NumExperts, m.Config.NumExpertsPerTok, m.Config.IsQwenNextHybridGGUF(), time.Since(t0).Seconds())
	if *loadOnly {
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
	logits := m.ForwardState(state, 0, 0, kvK, kvV)
	fmt.Printf("forward logits=%d\n", len(logits))
}
