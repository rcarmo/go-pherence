package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/backends/spacemit/board"
	"github.com/rcarmo/go-pherence/backends/spacemit/ort"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	modelPath := flag.String("model", "", "GGUF model path")
	outDir := flag.String("out", "/tmp/go-pherence-ort-layer-cache", "cache/output dir")
	iters := flag.Int("iters", 10, "iterations")
	threads := flag.Int("threads", 2, "SpaceMIT intra threads")
	batch := flag.Int("batch", 8, "batch/prefill rows")
	flag.Parse()
	if *modelPath == "" {
		panic("-model required")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		panic(err)
	}

	g, err := gguf.Open(*modelPath)
	if err != nil {
		panic(err)
	}
	defer g.Close()
	be := board.SIMDBackend{}
	tensors := []string{
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
	}
	fmt.Printf("batch=%d iters=%d threads=%d\n", *batch, *iters, *threads)
	for _, name := range tensors {
		t, ok := g.TensorByName(name)
		if !ok {
			continue
		}
		qm, err := g.MatrixFromTensor(t)
		if err != nil {
			panic(err)
		}
		fmt.Printf("\n== %s q=%d shape=[in=%d,out=%d] ==\n", name, qm.QType, qm.InDim, qm.OutDim)
		w, err := g.DequantF32(t)
		if err != nil {
			panic(err)
		}
		onnxPath := filepath.Join(*outDir, safe(name)+fmt.Sprintf("_b%d.onnx", *batch))
		if _, err := os.Stat(onnxPath); err != nil {
			fmt.Printf("  exporting ONNX...\n")
			if err := exportMatMulONNX(*outDir, onnxPath, w, qm.InDim, qm.OutDim, *batch); err != nil {
				panic(err)
			}
		}

		input := make([]float32, (*batch)*qm.InDim)
		for i := range input {
			input[i] = rand.Float32()*2 - 1
		}

		// CPU baseline: run each row through current row-vector GEMV.
		cpuOut := make([]float32, (*batch)*qm.OutDim)
		cpuStart := time.Now()
		for it := 0; it < *iters; it++ {
			for b := 0; b < *batch; b++ {
				_ = be.GemvF32(cpuOut[b*qm.OutDim:(b+1)*qm.OutDim], input[b*qm.InDim:(b+1)*qm.InDim], w, qm.InDim, qm.OutDim)
			}
		}
		cpuAvg := time.Since(cpuStart) / time.Duration(*iters)
		fmt.Printf("  cpu-rvv:      %s/iter\n", cpuAvg)

		sessStart := time.Now()
		sess, err := ort.NewSession(onnxPath, ort.Options{IntraThreadNum: *threads})
		if err != nil {
			fmt.Printf("  ort session ERR: %v\n", err)
			continue
		}
		fmt.Printf("  ort session:   %s\n", time.Since(sessStart).Round(time.Millisecond))
		ortOut, err := sess.Run1("X", input, []int64{int64(*batch), int64(qm.InDim)}, "Y", (*batch)*qm.OutDim)
		if err != nil {
			fmt.Printf("  ort run ERR: %v\n", err)
			sess.Close()
			continue
		}
		ortStart := time.Now()
		for it := 0; it < *iters; it++ {
			ortOut, err = sess.Run1("X", input, []int64{int64(*batch), int64(qm.InDim)}, "Y", (*batch)*qm.OutDim)
			if err != nil {
				panic(err)
			}
		}
		ortAvg := time.Since(ortStart) / time.Duration(*iters)
		sess.Close()
		fmt.Printf("  spacemit-ort: %s/iter\n", ortAvg)
		fmt.Printf("  speedup:      %.2fx vs CPU\n", float64(cpuAvg)/float64(ortAvg))
		fmt.Printf("  first:        cpu=%+.5f ort=%+.5f diff=%.5g\n", cpuOut[0], ortOut[0], math.Abs(float64(cpuOut[0]-ortOut[0])))
	}
}

func safe(s string) string { return strings.NewReplacer("/", "_", ".", "_").Replace(s) }

func exportMatMulONNX(dir, outPath string, wRowMajorOutIn []float32, inDim, outDim, batch int) error {
	binPath := outPath + ".w.bin"
	f, err := os.Create(binPath)
	if err != nil {
		return err
	}
	// ONNX wants W [K,N] = [in,out]. GGUF dequant is row-major [out,in]. Transpose while writing.
	buf := make([]byte, 4)
	for k := 0; k < inDim; k++ {
		for o := 0; o < outDim; o++ {
			binary.LittleEndian.PutUint32(buf, math.Float32bits(wRowMajorOutIn[o*inDim+k]))
			if _, err := f.Write(buf); err != nil {
				f.Close()
				return err
			}
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	script := filepath.Join(dir, "gen_matmul.py")
	code := fmt.Sprintf(`import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
in_dim=%d; out_dim=%d; batch=%d
W=np.fromfile(%q, dtype=np.float32).reshape(in_dim,out_dim)
X=helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch,in_dim])
Y=helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch,out_dim])
Wt=numpy_helper.from_array(W, "W")
node=helper.make_node("MatMul", ["X","W"], ["Y"], name="MatMul")
graph=helper.make_graph([node], "go_gguf_matmul", [X], [Y], [Wt])
model=helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)], ir_version=10)
onnx.save(model, %q)
`, inDim, outDim, batch, binPath, outPath)
	if err := os.WriteFile(script, []byte(code), 0o644); err != nil {
		return err
	}
	cmd := exec.Command("python3", script)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
