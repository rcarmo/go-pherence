package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/rcarmo/go-pherence/backends/ggmlgraph"
	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	path := flag.String("model", "", "GGUF model")
	iters := flag.Int("iters", 10, "iters")
	threads := flag.Int("threads", 8, "ggml threads")
	flag.Parse()
	if *path == "" {
		panic("-model required")
	}
	g, err := gguf.Open(*path)
	if err != nil {
		panic(err)
	}
	defer g.Close()
	be := k3.SIMDBackend{}
	loadM := func(name string) *gguf.QuantMatrix {
		t, ok := g.TensorByName(name)
		if !ok {
			panic(name)
		}
		m, err := g.MatrixFromTensor(t)
		if err != nil {
			panic(err)
		}
		return m
	}
	loadF := func(name string) []float32 {
		t, ok := g.TensorByName(name)
		if !ok {
			panic(name)
		}
		w, err := g.DequantF32(t)
		if err != nil {
			panic(err)
		}
		return w
	}

	h, qd, kvd, ffn := 2048, 2048, 256, 5632
	x := make([]float32, h)
	for i := range x {
		x[i] = rand.Float32()*2 - 1
	}

	fmt.Println("== QKV graph vs 3x GEMV ==")
	wq, wk, wv := loadM("blk.0.attn_q.weight"), loadM("blk.0.attn_k.weight"), loadM("blk.0.attn_v.weight")
	fq, fk, fv := loadF("blk.0.attn_q.weight"), loadF("blk.0.attn_k.weight"), loadF("blk.0.attn_v.weight")
	q, k, v := make([]float32, qd), make([]float32, kvd), make([]float32, kvd)
	start := time.Now()
	for i := 0; i < *iters; i++ {
		_ = be.GemvF32(q, x, fq, h, qd)
		_ = be.GemvF32(k, x, fk, h, kvd)
		_ = be.GemvF32(v, x, fv, h, kvd)
	}
	cpu := time.Since(start) / time.Duration(*iters)
	fmt.Printf("cpu-rvv 3x gemv: %s first=%+.5f\n", cpu, q[0])
	qg, err := ggmlgraph.NewQKV(wq, wk, wv, *threads)
	if err != nil {
		panic(err)
	}
	defer qg.Close()
	q2, k2, v2 := make([]float32, qd), make([]float32, kvd), make([]float32, kvd)
	_ = qg.Run(x, q2, k2, v2)
	start = time.Now()
	for i := 0; i < *iters; i++ {
		if err := qg.Run(x, q2, k2, v2); err != nil {
			panic(err)
		}
	}
	graph := time.Since(start) / time.Duration(*iters)
	fmt.Printf("ggml qkv graph: %s first=%+.5f speedup=%.2fx\n", graph, q2[0], float64(cpu)/float64(graph))

	fmt.Println("\n== MLP graph vs gate/up/silu/down ==")
	wg, wu, wd := loadM("blk.0.ffn_gate.weight"), loadM("blk.0.ffn_up.weight"), loadM("blk.0.ffn_down.weight")
	fg, fu, fd := loadF("blk.0.ffn_gate.weight"), loadF("blk.0.ffn_up.weight"), loadF("blk.0.ffn_down.weight")
	gate, up, mid, y := make([]float32, ffn), make([]float32, ffn), make([]float32, ffn), make([]float32, h)
	start = time.Now()
	for i := 0; i < *iters; i++ {
		_ = be.GemvF32(gate, x, fg, h, ffn)
		_ = be.GemvF32(up, x, fu, h, ffn)
		_ = be.SiLUMulF32(mid, gate, up)
		_ = be.GemvF32(y, mid, fd, ffn, h)
	}
	cpu = time.Since(start) / time.Duration(*iters)
	fmt.Printf("cpu-rvv mlp:    %s first=%+.5f\n", cpu, y[0])
	mg, err := ggmlgraph.NewMLP(wg, wu, wd, *threads)
	if err != nil {
		panic(err)
	}
	defer mg.Close()
	y2 := make([]float32, h)
	_ = mg.Run(x, y2)
	start = time.Now()
	for i := 0; i < *iters; i++ {
		if err := mg.Run(x, y2); err != nil {
			panic(err)
		}
	}
	graph = time.Since(start) / time.Duration(*iters)
	fmt.Printf("ggml mlp graph: %s first=%+.5f speedup=%.2fx\n", graph, y2[0], float64(cpu)/float64(graph))
}
