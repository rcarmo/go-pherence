package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/rcarmo/go-pherence/backends/ggmlcompute"
	"github.com/rcarmo/go-pherence/backends/ggmlgraph"
	"github.com/rcarmo/go-pherence/backends/ggmlquant"
	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	path := flag.String("model", "", "GGUF model")
	iters := flag.Int("iters", 3, "iters")
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
	for _, name := range []string{"blk.0.attn_q.weight", "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight"} {
		t, ok := g.TensorByName(name)
		if !ok {
			continue
		}
		qm, _ := g.MatrixFromTensor(t)
		x := make([]float32, qm.InDim)
		for i := range x {
			x[i] = rand.Float32()*2 - 1
		}
		fmt.Printf("\n== %s q=%d/%s shape=[%d,%d] vecdot=%v vecdot_type=%d/%s nrows=%d ==\n", name, qm.QType, ggmlquant.TypeName(int(qm.QType)), qm.InDim, qm.OutDim, ggmlquant.HasVecDot(int(qm.QType)), ggmlquant.VecDotType(int(qm.QType)), ggmlquant.TypeName(ggmlquant.VecDotType(int(qm.QType))), ggmlquant.NRows(int(qm.QType)))

		if qm.QType == gguf.QuantQ2_K || qm.QType == gguf.QuantQ3_K || qm.QType == gguf.QuantQ6_K {
			out := make([]float32, qm.OutDim)
			start := time.Now()
			for i := 0; i < *iters; i++ {
				if err := ggmlComputeGemv(out, x, qm); err != nil {
					panic(err)
				}
			}
			fmt.Printf("  ggml direct:      %s first=%+.5f\n", time.Since(start)/time.Duration(*iters), out[0])
		}
		if bg, err := ggmlgraph.NewBackendMulMat(int(qm.QType), qm.Raw, qm.InDim, qm.OutDim); err == nil {
			out := make([]float32, qm.OutDim)
			_ = bg.Run(x, out)
			start := time.Now()
			for i := 0; i < *iters; i++ {
				if err := bg.Run(x, out); err != nil {
					panic(err)
				}
			}
			fmt.Printf("  ggml backend:     %s first=%+.5f\n", time.Since(start)/time.Duration(*iters), out[0])
			bg.Close()
		}
		if graph, err := ggmlgraph.NewMulMat(int(qm.QType), qm.Raw, qm.InDim, qm.OutDim, 8); err == nil {
			out := make([]float32, qm.OutDim)
			_ = graph.Run(x, out)
			start := time.Now()
			for i := 0; i < *iters; i++ {
				if err := graph.Run(x, out); err != nil {
					panic(err)
				}
			}
			fmt.Printf("  ggml graph:       %s first=%+.5f\n", time.Since(start)/time.Duration(*iters), out[0])
			graph.Close()
		}
		if ggmlquant.HasVecDot(int(qm.QType)) {
			out := make([]float32, qm.OutDim)
			start := time.Now()
			for i := 0; i < *iters; i++ {
				if err := ggmlVecDotGemvRows(out, x, qm); err != nil {
					panic(err)
				}
			}
			fmt.Printf("  ggml vecdot rows: %s first=%+.5f\n", time.Since(start)/time.Duration(*iters), out[0])
		}
		out := make([]float32, qm.OutDim)
		start := time.Now()
		for i := 0; i < *iters; i++ {
			_ = model.QuantGemvRVVBlocks(out, x, qm)
		}
		fmt.Printf("  ours scratch+rvv: %s\n", time.Since(start)/time.Duration(*iters))
		w, _ := g.DequantF32(t)
		outF := make([]float32, qm.OutDim)
		start = time.Now()
		for i := 0; i < *iters; i++ {
			_ = be.GemvF32(outF, x, w, qm.InDim, qm.OutDim)
		}
		fmt.Printf("  f32-rvv:          %s first=%+.5f\n", time.Since(start)/time.Duration(*iters), outF[0])
	}
}

func ggmlComputeGemv(out []float32, x []float32, qm *gguf.QuantMatrix) error {
	q8, err := ggmlcompute.QuantizeQ8K(x[:qm.InDim])
	if err != nil {
		return err
	}
	rowBytes, err := qm.RowBytes()
	if err != nil {
		return err
	}
	return ggmlcompute.VecDotRowsDirect(int(qm.QType), out, qm.Raw, rowBytes, q8, qm.InDim, qm.OutDim)
}

func ggmlVecDotGemvRows(out []float32, x []float32, qm *gguf.QuantMatrix) error {
	vt := ggmlquant.VecDotType(int(qm.QType))
	xRaw, err := ggmlquant.QuantizeFromFloat(vt, x[:qm.InDim])
	if err != nil {
		return err
	}
	rowBytes, err := qm.RowBytes()
	if err != nil {
		return err
	}
	return ggmlquant.VecDotRows(int(qm.QType), out, qm.Raw, rowBytes, xRaw, qm.InDim, qm.OutDim)
}
