// cmd/k3bench benchmarks all available compute backends on the K3 SoC.
//
// It runs a micro-benchmark suite (GEMV, RMSNorm, SiLU, RoPE, Attention) across
// every available backend tier, then optionally runs llama-bench on a GGUF model.
//
// Usage:
//
//	k3bench [-model /path/to/model.gguf] [-threads N] [-size N]
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/backends/k3"
)

func main() {
	model := flag.String("model", "", "GGUF model path for end-to-end llama-bench (optional)")
	threads := flag.Int("threads", 8, "threads for llama-bench")
	size := flag.Int("size", 4096, "hidden dim for micro-benchmarks")
	outDim := flag.Int("out-dim", 4096, "output dim for GEMV micro-bench")
	seqLen := flag.Int("seq", 32, "KV sequence length for attention micro-bench")
	nHeads := flag.Int("heads", 32, "number of query heads")
	nKVHeads := flag.Int("kv-heads", 8, "number of KV heads (GQA)")
	headDim := flag.Int("head-dim", 128, "head dimension")
	iters := flag.Int("iters", 20, "iterations per micro-benchmark")
	flag.Parse()

	caps := k3.Probe()
	fmt.Println(caps.Summary())
	fmt.Println("SpacemiT runtime:")
	fmt.Println(k3.RVVCaps())
	fmt.Println()

	backends := k3.SelectAll()

	for _, b := range backends {
		fmt.Printf("── %s ──────────────────────────\n", b.Name())
		benchAll(b, *size, *outDim, *seqLen, *nHeads, *nKVHeads, *headDim, *iters)
		fmt.Println()
	}

	if *model != "" {
		fmt.Println("── SpacemiT llama-bench ────────────────────────────")
		res, err := k3.RunGGUF(*model, *threads, 128, 64, 180*time.Second)
		if err != nil {
			fmt.Fprintf(os.Stderr, "llama-bench failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("  PP-128 : %.2f t/s\n", res.PP)
		fmt.Printf("  TG-64  : %.2f t/s\n", res.TG)
		fmt.Println()
		for _, l := range res.RawLines {
			fmt.Println("  ", l)
		}
	}
}

func benchAll(b k3.OpBackend, sz, outDim, seqLen, nH, nKV, hDim, iters int) {
	rng := rand.New(rand.NewSource(42))
	mkVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}

	// GEMV: [outDim × sz] · [sz] → [outDim]
	w := mkVec(outDim * sz)
	x := mkVec(sz)
	out := make([]float32, outDim)
	benchOp(b.Name(), "GEMV", iters, func() error {
		return b.GemvF32(out, x, w, sz, outDim)
	})

	// RMSNorm
	vec := mkVec(sz)
	weight := mkVec(sz)
	for i := range weight {
		weight[i] = 1
	}
	benchOp(b.Name(), "RMSNorm", iters, func() error {
		return b.RMSNormF32(vec, weight, 1e-5)
	})

	// SiLUMul
	gate := mkVec(sz)
	up := mkVec(sz)
	dst := make([]float32, sz)
	benchOp(b.Name(), "SiLUMul", iters, func() error {
		return b.SiLUMulF32(dst, gate, up)
	})

	// RoPE
	rotHalf := hDim / 2
	qvec := mkVec(nH * hDim)
	freqs := mkVec(rotHalf)
	for i := range freqs {
		freqs[i] = float32(math.Exp(-float64(i) * 0.1))
	}
	benchOp(b.Name(), "RoPE", iters, func() error {
		return b.RoPEPartialF32(qvec, freqs, 0, nH, hDim, rotHalf)
	})

	// Attention scores
	q := mkVec(nH * hDim)
	kCache := mkVec(seqLen * nKV * hDim)
	scores := make([]float32, nH*seqLen)
	scale := float32(1.0 / math.Sqrt(float64(hDim)))
	benchOp(b.Name(), "AttnScores", iters, func() error {
		return b.AttentionScoresF32(scores, q, kCache, seqLen, nH, nKV, hDim, scale)
	})
}

func benchOp(backend, name string, iters int, fn func() error) {
	// Warmup
	for i := 0; i < 3; i++ {
		if err := fn(); err != nil {
			fmt.Printf("  %-12s  SKIP (%v)\n", name, err)
			return
		}
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		if err := fn(); err != nil {
			fmt.Printf("  %-12s  ERR  (%v)\n", name, err)
			return
		}
	}
	elapsed := time.Since(start)
	avg := elapsed / time.Duration(iters)
	fmt.Printf("  %-12s  %10s/iter  (%d iters)\n", name, avg, iters)
}
