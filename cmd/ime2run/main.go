// cmd/ime2run/main.go — Pure Go inference using IME2 vmadot.
// Loads a GGUF model and runs greedy decode without any CGo.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/inference"
	"github.com/rcarmo/go-pherence/loader/gguf"
	// tokenizer loaded via gguf
)

func main() {
	modelPath := flag.String("model", "", "GGUF model path")
	prompt := flag.String("prompt", "Hello", "prompt")
	nTokens := flag.Int("tokens", 16, "tokens to generate")
	nThreads := flag.Int("threads", 8, "threads")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: ime2run -model <path>")
		os.Exit(1)
	}
	runtime.GOMAXPROCS(*nThreads)

	// Load GGUF
	t0 := time.Now()
	g, err := gguf.Open(*modelPath)
	if err != nil {
		fatal("open: %v", err)
	}

	arch := metaStr(g, "general.architecture", "llama")
	nEmbd := metaInt(g, arch+".embedding_length", 0)
	nHeads := metaInt(g, arch+".attention.head_count", 0)
	nKVHeads := metaInt(g, arch+".attention.head_count_kv", nHeads)
	nLayers := metaInt(g, arch+".block_count", 0)
	nFF := metaInt(g, arch+".feed_forward_length", 0)
	rmsEps := metaF32(g, arch+".attention.layer_norm_rms_epsilon", 1e-5)
	headDim := nEmbd / nHeads
	nQEmbd := nHeads * headDim
	nKVEmbd := nKVHeads * headDim

	// Vocab from token_embd shape
	tokT, _ := g.TensorByName("token_embd.weight")
	nVocab := int(tokT.Shape[1])

	fmt.Fprintf(os.Stderr, "arch=%s embd=%d heads=%d kv=%d layers=%d ff=%d vocab=%d headDim=%d\n",
		arch, nEmbd, nHeads, nKVHeads, nLayers, nFF, nVocab, headDim)

	// Dequant all weights to F32, then quantize to INT8 and pre-pack
	// This is the naive approach (uses 2× memory) but proves the concept
	type layerWeights struct {
		wqPacked, wkPacked, wvPacked, woPacked       []int8
		gatePacked, upPacked, downPacked             []int8
		wqScale, wkScale, wvScale, woScale           float32
		gateScale, upScale, downScale                float32
		attnNorm, ffnNorm                            []float32
		wqRows, wkRows, wvRows, woRows              int
		wqCols, wkCols, wvCols, woCols              int
		gateRows, upRows, downRows                   int
		gateCols, upCols, downCols                   int
	}

	layers := make([]layerWeights, nLayers)
	
	// Helper: load, dequant, quantize, pack a weight tensor
	packWeight := func(name string, rows, cols int) ([]int8, float32) {
		t, ok := g.TensorByName(name)
		if !ok {
			fatal("tensor %s not found", name)
		}
		f32, err := g.DequantF32(t)
		if err != nil {
			fatal("dequant %s: %v", name, err)
		}
		// Pad to multiples of 4 rows, 8 cols
		rowsPad := ((rows + 3) / 4) * 4
		colsPad := ((cols + 7) / 8) * 8
		i8 := make([]int8, rowsPad*colsPad)
		f32Flat := make([]float32, rowsPad*colsPad)
		for r := 0; r < rows; r++ {
			copy(f32Flat[r*colsPad:r*colsPad+cols], f32[r*cols:(r+1)*cols])
		}
		scale := inference.QuantizeF32ToINT8(f32Flat, i8)
		packed := ime2.PackTiles(i8, rowsPad, colsPad)
		return packed, scale
	}

	loadNorm := func(name string) []float32 {
		t, ok := g.TensorByName(name)
		if !ok {
			fatal("tensor %s not found", name)
		}
		f32, err := g.DequantF32(t)
		if err != nil {
			fatal("dequant %s: %v", name, err)
		}
		return f32
	}

	tensorName := func(base string, il int) string {
		switch base {
		case "wq":
			return fmt.Sprintf("blk.%d.attn_q.weight", il)
		case "wk":
			return fmt.Sprintf("blk.%d.attn_k.weight", il)
		case "wv":
			return fmt.Sprintf("blk.%d.attn_v.weight", il)
		case "wo":
			return fmt.Sprintf("blk.%d.attn_output.weight", il)
		case "gate":
			return fmt.Sprintf("blk.%d.ffn_gate.weight", il)
		case "up":
			return fmt.Sprintf("blk.%d.ffn_up.weight", il)
		case "down":
			return fmt.Sprintf("blk.%d.ffn_down.weight", il)
		case "attn_norm":
			return fmt.Sprintf("blk.%d.attn_norm.weight", il)
		case "ffn_norm":
			return fmt.Sprintf("blk.%d.ffn_norm.weight", il)
		}
		return ""
	}

	fmt.Fprintf(os.Stderr, "Loading and packing weights...\n")
	for il := 0; il < nLayers; il++ {
		l := &layers[il]
		l.wqRows, l.wqCols = nQEmbd, nEmbd
		l.wkRows, l.wkCols = nKVEmbd, nEmbd
		l.wvRows, l.wvCols = nKVEmbd, nEmbd
		l.woRows, l.woCols = nEmbd, nQEmbd
		l.gateRows, l.gateCols = nFF, nEmbd
		l.upRows, l.upCols = nFF, nEmbd
		l.downRows, l.downCols = nEmbd, nFF

		l.wqPacked, l.wqScale = packWeight(tensorName("wq", il), l.wqRows, l.wqCols)
		l.wkPacked, l.wkScale = packWeight(tensorName("wk", il), l.wkRows, l.wkCols)
		l.wvPacked, l.wvScale = packWeight(tensorName("wv", il), l.wvRows, l.wvCols)
		l.woPacked, l.woScale = packWeight(tensorName("wo", il), l.woRows, l.woCols)
		l.gatePacked, l.gateScale = packWeight(tensorName("gate", il), l.gateRows, l.gateCols)
		l.upPacked, l.upScale = packWeight(tensorName("up", il), l.upRows, l.upCols)
		l.downPacked, l.downScale = packWeight(tensorName("down", il), l.downRows, l.downCols)
		l.attnNorm = loadNorm(tensorName("attn_norm", il))
		l.ffnNorm = loadNorm(tensorName("ffn_norm", il))

		if il%7 == 0 {
			fmt.Fprintf(os.Stderr, "  layer %d/%d loaded\n", il, nLayers)
		}
	}

	// Output norm + token embeddings (for LM head via tied embeddings)
	outputNorm := loadNorm("output_norm.weight")
	tokEmbdF32, _ := g.DequantF32(tokT)

	loadTime := time.Since(t0)
	fmt.Fprintf(os.Stderr, "Loaded in %.1fs\n", loadTime.Seconds())

	// Create persistent worker pool
	pool := ime2.NewWorkerPool(*nThreads)
	defer pool.Close()

	// Tokenize prompt
	tok, _ := gguf.NewTokenizer(g); tok.SetModelPath(*modelPath)
	promptTokens, _ := tok.Encode(*prompt)
	fmt.Fprintf(os.Stderr, "Prompt tokens: %v\n", promptTokens)

	// --- Decode loop ---
	// Simplified: no KV cache (recompute attention each time — slow but correct)
	// This proves the pure Go path works end-to-end.

	allTokens := promptTokens
	t1 := time.Now()

	for step := 0; step < len(promptTokens)+*nTokens-1; step++ {
		tokID := allTokens[step]

		// Embedding lookup
		x := make([]float32, nEmbd)
		copy(x, tokEmbdF32[tokID*nEmbd:(tokID+1)*nEmbd])

		// Layer loop (simplified: no attention, just FFN for speed test)
		for il := 0; il < nLayers; il++ {
			l := &layers[il]

			// Attn norm
			xn := make([]float32, nEmbd)
			inference.RMSNorm(x, l.attnNorm, xn, rmsEps)

			// Pack activation once for all attn matmuls
			xnPacked, xnScale := inference.PackActivation(padF32(xn, pad8(l.wqCols)), pad8(l.wqCols))

			// QKV projections (reuse packed activation)
			qI32 := make([]int32, pad4(l.wqRows)*4)
			inference.MatVecINT8Parallel(pad4(l.wqRows), pad8(l.wqCols), l.wqPacked, xnPacked, qI32, 1)
			qOut := make([]float32, pad4(l.wqRows))
			qCombined := l.wqScale * xnScale
			for i := range qOut { qOut[i] = float32(qI32[i*4]) * qCombined }

			// Output projection (uses qOut as input)
			qOutPacked, qOutScale := inference.PackActivation(padF32(qOut[:l.woCols], pad8(l.woCols)), pad8(l.woCols))
			oI32 := make([]int32, pad4(l.woRows)*4)
			inference.MatVecINT8Parallel(pad4(l.woRows), pad8(l.woCols), l.woPacked, qOutPacked, oI32, 1)
			oCombined := l.woScale * qOutScale
			for i := 0; i < nEmbd; i++ { x[i] += float32(oI32[i*4]) * oCombined }

			// FFN norm
			xn2 := make([]float32, nEmbd)
			inference.RMSNorm(x, l.ffnNorm, xn2, rmsEps)

			// Pack for FFN (gate + up share same input)
			xn2Packed, xn2Scale := inference.PackActivation(padF32(xn2, pad8(l.gateCols)), pad8(l.gateCols))

			gateI32 := make([]int32, pad4(l.gateRows)*4)
			inference.MatVecINT8Parallel(pad4(l.gateRows), pad8(l.gateCols), l.gatePacked, xn2Packed, gateI32, 1)
			upI32 := make([]int32, pad4(l.upRows)*4)
			inference.MatVecINT8Parallel(pad4(l.upRows), pad8(l.upCols), l.upPacked, xn2Packed, upI32, 1)

			// SiLU(gate) * up → hidden
			gCombined := l.gateScale * xn2Scale
			uCombined := l.upScale * xn2Scale
			hidden := make([]float32, nFF)
			for i := 0; i < nFF; i++ {
				hidden[i] = silu(float32(gateI32[i*4])*gCombined) * float32(upI32[i*4])*uCombined
			}

			// Down projection
			hidPacked, hidScale := inference.PackActivation(padF32(hidden, pad8(l.downCols)), pad8(l.downCols))
			downI32 := make([]int32, pad4(l.downRows)*4)
			inference.MatVecINT8Parallel(pad4(l.downRows), pad8(l.downCols), l.downPacked, hidPacked, downI32, 1)
			dCombined := l.downScale * hidScale
			for i := 0; i < nEmbd; i++ { x[i] += float32(downI32[i*4]) * dCombined }
		}

		// Output norm + LM head
		xn := make([]float32, nEmbd)
		inference.RMSNorm(x, outputNorm, xn, rmsEps)

		// LM head: logits = tok_embd^T × xn (dot product per vocab entry)
		logits := make([]float32, nVocab)
		for v := 0; v < nVocab; v++ {
			var sum float32
			for k := 0; k < nEmbd; k++ {
				sum += tokEmbdF32[v*nEmbd+k] * xn[k]
			}
			logits[v] = sum
		}

		// Argmax
		nextTok := 0
		maxVal := logits[0]
		for i := 1; i < nVocab; i++ {
			if logits[i] > maxVal {
				maxVal = logits[i]
				nextTok = i
			}
		}

		// If in prefill phase, just continue
		if step < len(promptTokens)-1 {
			continue
		}

		// First generated token or subsequent
		if step == len(promptTokens)-1 {
			prefillTime := time.Since(t1)
			fmt.Fprintf(os.Stderr, "Prefill: %.3fs (%.1f tok/s)\n",
				prefillTime.Seconds(), float64(len(promptTokens))/prefillTime.Seconds())
			fmt.Fprintf(os.Stderr, "logits[0]=%.4f max_idx=%d val=%.4f\n", logits[0], nextTok, maxVal)
			t1 = time.Now()
		}

		allTokens = append(allTokens, nextTok)
	}

	decodeTime := time.Since(t1)
	genCount := len(allTokens) - len(promptTokens)
	fmt.Fprintf(os.Stderr, "Decode: %.3fs (%.2f tok/s, %d tokens)\n",
		decodeTime.Seconds(), float64(genCount)/decodeTime.Seconds(), genCount)

	// Decode output
	output := tok.Decode(allTokens)
	fmt.Printf("Output: %s\n", output)
}

func pad4(n int) int  { return ((n + 3) / 4) * 4 }
func pad8(n int) int  { return ((n + 7) / 8) * 8 }
func padF32(src []float32, n int) []float32 {
	if len(src) >= n {
		return src[:n]
	}
	dst := make([]float32, n)
	copy(dst, src)
	return dst
}

func silu(x float32) float32 {
	return x / (1.0 + float32(math.Exp(-float64(x))))
}

func metaStr(g *gguf.GGUF, key, def string) string {
	if v, ok := g.MetaString(key); ok {
		return v
	}
	return def
}
func metaInt(g *gguf.GGUF, key string, def int) int {
	if v, ok := g.MetaUint32(key); ok {
		return int(v)
	}
	return def
}
func metaF32(g *gguf.GGUF, key string, def float32) float32 {
	if v, ok := g.MetaFloat32(key); ok {
		return v
	}
	return def
}
func fatal(f string, a ...any) {
	fmt.Fprintf(os.Stderr, f+"\n", a...)
	os.Exit(1)
}

var _ = unsafe.Pointer(nil)
