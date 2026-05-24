//go:build cgo && linux

// cmd/k3graphrun benchmarks the llamagraph full GGML decode-graph backend,
// wiring go-pherence's own GGUF loader to the new gpll_model executor.
// Target: match cmd/k3llama (~12 tok/s decode on MilkV Jupiter K3).
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/backends/llamagraph"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file (required)")
	prompt    := flag.String("prompt", "Once upon a time", "prompt text")
	nTokens   := flag.Int("tokens", 64, "tokens to generate")
	nThreads  := flag.Int("threads", 8, "CPU threads")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: k3graphrun -model <path> [-prompt <text>] [-tokens N] [-threads N]")
		os.Exit(1)
	}

	// --- Load GGUF ---
	fmt.Printf("Loading %s …\n", *modelPath)
	t0 := time.Now()
	g, err := gguf.Open(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gguf open: %v\n", err)
		os.Exit(1)
	}

	getUint32 := func(key string, def int) int {
		if v, ok := g.MetaUint32(key); ok { return int(v) }
		return def
	}
	getFloat32 := func(key string, def float32) float32 {
		if v, ok := g.MetaFloat32(key); ok { return v }
		return def
	}

	nVocab   := getUint32("tokenizer.ggml.tokens", 32000)
	nVocab2  := getUint32("llama.vocab_size", nVocab)
	if nVocab2 > nVocab { nVocab = nVocab2 }
	nEmbd    := getUint32("llama.embedding_length", 2048)
	nHeads   := getUint32("llama.attention.head_count", 32)
	nHeadsKV := getUint32("llama.attention.head_count_kv", nHeads)
	nLayers  := getUint32("llama.block_count", 22)
	nFF      := getUint32("llama.feed_forward_length", 5632)
	nCtx     := 2048
	ropeBase := getFloat32("llama.rope.freq_base", 10000.0)
	rmsEps   := getFloat32("llama.attention.layer_norm_rms_epsilon", 1e-5)
	ropeDims := nEmbd / nHeads

	fmt.Printf("Config: vocab=%d embd=%d heads=%d kv=%d layers=%d ff=%d\n",
		nVocab, nEmbd, nHeads, nHeadsKV, nLayers, nFF)

	qtType := func(name string) int {
		t, ok := g.TensorByName(name)
		if !ok { return llamagraph.GGMLTypeF32 }
		return int(t.QType)
	}

	cfg := llamagraph.Config{
		NVocab: nVocab, NEmbd: nEmbd, NHeads: nHeads, NHeadsKV: nHeadsKV,
		NLayers: nLayers, NFF: nFF, NCtx: nCtx,
		RopeBase: ropeBase, RmsEps: rmsEps, RopeDims: ropeDims,
		NThreads:    *nThreads,
		TokEmbdType: qtType("token_embd.weight"),
		OutputType:  qtType("output.weight"),
		WQType:      make([]int, nLayers),
		WKType:      make([]int, nLayers),
		WVType:      make([]int, nLayers),
		WOType:      make([]int, nLayers),
		FFNGateType: make([]int, nLayers),
		FFNUpType:   make([]int, nLayers),
		FFNDownType: make([]int, nLayers),
	}
	for il := 0; il < nLayers; il++ {
		pfx := fmt.Sprintf("blk.%d.", il)
		cfg.WQType[il]      = qtType(pfx + "attn_q.weight")
		cfg.WKType[il]      = qtType(pfx + "attn_k.weight")
		cfg.WVType[il]      = qtType(pfx + "attn_v.weight")
		cfg.WOType[il]      = qtType(pfx + "attn_output.weight")
		cfg.FFNGateType[il] = qtType(pfx + "ffn_gate.weight")
		cfg.FFNUpType[il]   = qtType(pfx + "ffn_up.weight")
		cfg.FFNDownType[il] = qtType(pfx + "ffn_down.weight")
	}

	// --- Init model ---
	m, err := llamagraph.New(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "llamagraph.New: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	rawBytes := func(name string) []byte {
		t, ok := g.TensorByName(name)
		if !ok { return nil }
		raw, err := g.Raw(t)
		if err != nil { return nil }
		return raw
	}

	m.SetTokEmbd(rawBytes("token_embd.weight"))
	m.SetOutputNorm(rawBytes("output_norm.weight"))
	outW := rawBytes("output.weight")
	if outW == nil { outW = rawBytes("token_embd.weight") }
	m.SetOutput(outW)

	for il := 0; il < nLayers; il++ {
		pfx := fmt.Sprintf("blk.%d.", il)
		m.SetLayerAttnNorm(il, rawBytes(pfx+"attn_norm.weight"))
		m.SetLayerWQ(il, rawBytes(pfx+"attn_q.weight"))
		m.SetLayerWK(il, rawBytes(pfx+"attn_k.weight"))
		m.SetLayerWV(il, rawBytes(pfx+"attn_v.weight"))
		m.SetLayerWO(il, rawBytes(pfx+"attn_output.weight"))
		m.SetLayerFFNNorm(il, rawBytes(pfx+"ffn_norm.weight"))
		m.SetLayerFFNGate(il, rawBytes(pfx+"ffn_gate.weight"))
		m.SetLayerFFNUp(il, rawBytes(pfx+"ffn_up.weight"))
		m.SetLayerFFNDown(il, rawBytes(pfx+"ffn_down.weight"))
	}

	loadTime := time.Since(t0)
	fmt.Printf("Loaded in %.3fs\n", loadTime.Seconds())

	// --- Tokenise ---
	tokenizer, err := gguf.NewTokenizer(g)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(1)
	}
	tokenizer.SetModelPath(*modelPath)
	tokens, err := tokenizer.Encode(*prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Prompt tokens: %d\n", len(tokens))

	// --- Prefill ---
	t1 := time.Now()
	var lastLogits []float32
	for i, tok := range tokens {
		logits, err := m.Decode(tok)
		if err != nil {
			fmt.Fprintf(os.Stderr, "decode[%d]: %v\n", i, err)
			os.Exit(1)
		}
		lastLogits = logits

	}
	prefillTime := time.Since(t1)
	ppTPS := float64(len(tokens)) / prefillTime.Seconds()
	fmt.Printf("Prefill: %.3fs → %.2f tok/s\n", prefillTime.Seconds(), ppTPS)

	// --- Decode loop ---
	t2 := time.Now()
	output := *prompt
	for step := 0; step < *nTokens; step++ {
		nextTok := argmax(lastLogits)
		text := tokenizer.Decode([]int{nextTok})
		output += text

		logits, err := m.Decode(nextTok)
		if err != nil {
			fmt.Fprintf(os.Stderr, "decode step %d: %v\n", step, err)
			break
		}
		lastLogits = logits
	}
	decodeTime := time.Since(t2)
	tgTPS := float64(*nTokens) / decodeTime.Seconds()

	fmt.Printf("Decode:  %.3fs → %.2f tok/s (%d tokens)\n",
		decodeTime.Seconds(), tgTPS, *nTokens)
	fmt.Printf("Output: %s\n", output)
}

func argmax(v []float32) int {
	best, idx := float32(math.Inf(-1)), 0
	for i, x := range v {
		if x > best { best = x; idx = i }
	}
	return idx
}
