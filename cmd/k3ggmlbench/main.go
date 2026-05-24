package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

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
		fmt.Printf("\n== %s q=%d/%s shape=[%d,%d] vecdot=%v vecdot_type=%d/%s ==\n", name, qm.QType, ggmlquant.TypeName(int(qm.QType)), qm.InDim, qm.OutDim, ggmlquant.HasVecDot(int(qm.QType)), ggmlquant.VecDotType(int(qm.QType)), ggmlquant.TypeName(ggmlquant.VecDotType(int(qm.QType))))
		out := make([]float32, qm.OutDim)
		graph, err := ggmlgraph.NewMulMat(int(qm.QType), qm.Raw, qm.InDim, qm.OutDim, 8)
		if err == nil {
			outGraph := make([]float32, qm.OutDim)
			_ = graph.Run(x, outGraph)
			graphStart := time.Now()
			for i := 0; i < *iters; i++ {
				if err := graph.Run(x, outGraph); err != nil {
					panic(err)
				}
			}
			fmt.Printf("  ggml graph:       %s first=%+.5f\n", time.Since(graphStart)/time.Duration(*iters), outGraph[0])
			graph.Close()
		} else {
			fmt.Printf("  ggml graph:       ERR %v\n", err)
		}

		if ggmlquant.HasVecDot(int(qm.QType)) {
			outG := make([]float32, qm.OutDim)
			gStart := time.Now()
			for i := 0; i < *iters; i++ {
				if err := ggmlVecDotGemv(outG, x, qm); err != nil {
					panic(err)
				}
			}
			fmt.Printf("  ggml vecdot row:  %s first=%+.5f\n", time.Since(gStart)/time.Duration(*iters), outG[0])
			outGR := make([]float32, qm.OutDim)
			grStart := time.Now()
			for i := 0; i < *iters; i++ {
				if err := ggmlVecDotGemvRows(outGR, x, qm); err != nil {
					panic(err)
				}
			}
			fmt.Printf("  ggml vecdot rows: %s first=%+.5f\n", time.Since(grStart)/time.Duration(*iters), outGR[0])
		}
		start := time.Now()
		for i := 0; i < *iters; i++ {
			_ = model.QuantGemvRVVBlocks(out, x, qm)
		}
		fmt.Printf("  ours scratch+rvv: %s\n", time.Since(start)/time.Duration(*iters))
		if qm.QType == gguf.QuantQ2_K || qm.QType == gguf.QuantQ3_K || qm.QType == gguf.QuantQ6_K {
			outC := make([]float32, qm.OutDim)
			start = time.Now()
			for i := 0; i < *iters; i++ {
				_ = model.QuantGemvCgoFused(outC, x, qm)
			}
			fmt.Printf("  ours fused-cgo:   %s first=%+.5f\n", time.Since(start)/time.Duration(*iters), outC[0])
		}
		w, _ := g.DequantF32(t)
		outF := make([]float32, qm.OutDim)
		start = time.Now()
		for i := 0; i < *iters; i++ {
			_ = be.GemvF32(outF, x, w, qm.InDim, qm.OutDim)
		}
		fmt.Printf("  f32-rvv:          %s first=%+.5f\n", time.Since(start)/time.Duration(*iters), outF[0])
		// ggml row dequant + f32-rvv only measures ggml dequant correctness/speed for one full GEMV.
		row := make([]float32, qm.InDim)
		start = time.Now()
		for r := 0; r < qm.OutDim; r++ {
			rb, _ := qm.RowBytes()
			_ = ggmlquant.DequantRow(int(qm.QType), qm.Raw[r*rb:(r+1)*rb], row)
			var s float32
			for j, v := range row {
				s += v * x[j]
			}
			out[r] = s
		}
		fmt.Printf("  ggml-deq+scalar:  %s first=%+.5f\n", time.Since(start), out[0])
	}
}

func ggmlVecDotGemv(out []float32, x []float32, qm *gguf.QuantMatrix) error {
	vt := ggmlquant.VecDotType(int(qm.QType))
	xRaw, err := ggmlquant.QuantizeFromFloat(vt, x[:qm.InDim])
	if err != nil {
		return err
	}
	rowBytes, err := qm.RowBytes()
	if err != nil {
		return err
	}
	for r := 0; r < qm.OutDim; r++ {
		v, err := ggmlquant.VecDot(int(qm.QType), qm.Raw[r*rowBytes:(r+1)*rowBytes], xRaw, qm.InDim)
		if err != nil {
			return err
		}
		out[r] = v
	}
	return nil
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
