package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/rcarmo/go-pherence/backends/spacemit/ort"
)

func main() {
	outDir := flag.String("out", "/tmp/go-pherence-ort-probes", "probe directory")
	iters := flag.Int("iters", 20, "iterations")
	threads := flag.Int("threads", 2, "SpaceMIT intra threads")
	flag.Parse()
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		panic(err)
	}
	if err := generateProbes(*outDir); err != nil {
		panic(err)
	}

	type shape struct {
		name    string
		m, k, n int
	}
	shapes := []shape{{"q_2048x2048", 1, 2048, 2048}, {"kv_2048x256", 1, 2048, 256}, {"ffn_up_2048x5632", 1, 2048, 5632}, {"ffn_down_5632x2048", 1, 5632, 2048}, {"batch8_q_2048x2048", 8, 2048, 2048}, {"batch32_q_2048x2048", 32, 2048, 2048}, {"batch8_4096x4096", 8, 4096, 4096}}
	for _, s := range shapes {
		path := filepath.Join(*outDir, s.name+".onnx")
		input := make([]float32, s.m*s.k)
		for i := range input {
			input[i] = rand.Float32()*2 - 1
		}
		fmt.Printf("\n== %s [%d,%d]x[%d,%d] ==\n", s.name, s.m, s.k, s.k, s.n)
		sess, err := ort.NewSession(path, ort.Options{IntraThreadNum: *threads})
		if err != nil {
			fmt.Println("session ERR", err)
			continue
		}
		// warmup
		out, err := sess.Run1("X", input, []int64{int64(s.m), int64(s.k)}, "Y", s.m*s.n)
		if err != nil {
			fmt.Println("run ERR", err)
			sess.Close()
			continue
		}
		t0 := time.Now()
		for i := 0; i < *iters; i++ {
			out, err = sess.Run1("X", input, []int64{int64(s.m), int64(s.k)}, "Y", s.m*s.n)
			if err != nil {
				panic(err)
			}
		}
		avg := time.Since(t0) / time.Duration(*iters)
		fmt.Printf("spacemit-ort: %s/iter first=%+.5f\n", avg, out[0])
		sess.Close()
	}
}

func generateProbes(dir string) error {
	script := filepath.Join(dir, "gen.py")
	code := `import os, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
shapes=[("q_2048x2048",1,2048,2048),("kv_2048x256",1,2048,256),("ffn_up_2048x5632",1,2048,5632),("ffn_down_5632x2048",1,5632,2048),("batch8_q_2048x2048",8,2048,2048),("batch32_q_2048x2048",32,2048,2048),("batch8_4096x4096",8,4096,4096)]
for name,m,k,n in shapes:
 X=helper.make_tensor_value_info("X", TensorProto.FLOAT, [m,k]); Y=helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m,n])
 W=numpy_helper.from_array((np.random.randn(k,n).astype(np.float32)*0.01), "W")
 model=helper.make_model(helper.make_graph([helper.make_node("MatMul",["X","W"],["Y"],name="MatMul")],name,[X],[Y],[W]),opset_imports=[helper.make_opsetid("",17)],ir_version=10)
 onnx.save(model, os.path.join("` + dir + `", name+".onnx"))
`
	if err := os.WriteFile(script, []byte(code), 0o644); err != nil {
		return err
	}
	cmd := exec.Command("python3", script)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
