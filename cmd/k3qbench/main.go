package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	path := flag.String("model", "", "GGUF model path")
	iters := flag.Int("iters", 5, "iterations")
	flag.Parse()
	if *path == "" {
		panic("-model required")
	}
	g, err := gguf.Open(*path)
	if err != nil {
		panic(err)
	}
	defer g.Close()
	bench := []string{
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_down.weight",
		"output.weight",
	}
	be := k3.SIMDBackend{}
	for _, name := range bench {
		t, ok := g.TensorByName(name)
		if !ok {
			continue
		}
		qm, err := g.MatrixFromTensor(t)
		if err != nil {
			panic(err)
		}
		x := make([]float32, qm.InDim)
		for i := range x {
			x[i] = rand.Float32()*2 - 1
		}
		outQ := make([]float32, qm.OutDim)
		outF := make([]float32, qm.OutDim)
		fmt.Printf("── %s q=%d shape=[%d,%d] ──\n", name, qm.QType, qm.InDim, qm.OutDim)

		// Experimental quantized GGUF RVV path.
		qStart := time.Now()
		for i := 0; i < *iters; i++ {
			if err := model.QuantGemvRVVBlocks(outQ, x, qm); err != nil {
				panic(err)
			}
		}
		qAvg := time.Since(qStart) / time.Duration(*iters)
		fmt.Printf("  quant-rvv: %s/iter\n", qAvg)

		// F32 baseline: one-time dequant then parallel RVV GEMV.
		fStartLoad := time.Now()
		w, err := g.DequantF32(t)
		if err != nil {
			panic(err)
		}
		load := time.Since(fStartLoad)
		fStart := time.Now()
		for i := 0; i < *iters; i++ {
			_ = be.GemvF32(outF, x, w, qm.InDim, qm.OutDim)
		}
		fAvg := time.Since(fStart) / time.Duration(*iters)
		fmt.Printf("  f32-rvv:   %s/iter  (one-time dequant %s)\n", fAvg, load.Round(time.Millisecond))
		fmt.Printf("  first: quant=%+.5f f32=%+.5f\n\n", outQ[0], outF[0])
	}
}
