package main

import (
	"flag"
	"fmt"
	"github.com/rcarmo/go-pherence/backends/ggmlgraph"
	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"math/rand"
	"time"
)

func rms(x, w []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	scale := float32(1.0) // avoid importing math? no
	_ = scale
	_ = ss
	_ = eps
	for i := range x {
		x[i] *= w[i]
	}
}
func main() {
	path := flag.String("model", "", "")
	iters := flag.Int("iters", 5, "")
	threads := flag.Int("threads", 8, "")
	flag.Parse()
	if *path == "" {
		panic("model")
	}
	g, err := gguf.Open(*path)
	if err != nil {
		panic(err)
	}
	defer g.Close()
	loadM := func(n string) *gguf.QuantMatrix { t, _ := g.TensorByName(n); m, _ := g.MatrixFromTensor(t); return m }
	loadF := func(n string) []float32 { t, _ := g.TensorByName(n); w, _ := g.DequantF32(t); return w }
	wg, wu, wd := loadM("blk.0.ffn_gate.weight"), loadM("blk.0.ffn_up.weight"), loadM("blk.0.ffn_down.weight")
	norm := loadF("blk.0.ffn_norm.weight")
	fg, fu, fd := loadF("blk.0.ffn_gate.weight"), loadF("blk.0.ffn_up.weight"), loadF("blk.0.ffn_down.weight")
	h, ffn := 2048, 5632
	x := make([]float32, h)
	for i := range x {
		x[i] = rand.Float32()*2 - 1
	}
	be := k3.SIMDBackend{}
	gate, up, mid, down := make([]float32, ffn), make([]float32, ffn), make([]float32, ffn), make([]float32, h)
	start := time.Now()
	for i := 0; i < *iters; i++ {
		xx := append([]float32(nil), x...)
		rms(xx, norm, 1e-5)
		_ = be.GemvF32(gate, xx, fg, h, ffn)
		_ = be.GemvF32(up, xx, fu, h, ffn)
		_ = be.SiLUMulF32(mid, gate, up)
		_ = be.GemvF32(down, mid, fd, ffn, h)
		for j := range down {
			down[j] += x[j]
		}
	}
	fmt.Println("cpu approx", time.Since(start)/time.Duration(*iters), down[0])
	gb, err := ggmlgraph.NewFFNBlock(norm, wg, wu, wd, 1e-5, *threads)
	if err != nil {
		panic(err)
	}
	defer gb.Close()
	y := make([]float32, h)
	_ = gb.Run(x, y)
	start = time.Now()
	for i := 0; i < *iters; i++ {
		_ = gb.Run(x, y)
	}
	fmt.Println("ggml ffnblock", time.Since(start)/time.Duration(*iters), y[0])
}
