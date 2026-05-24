//go:build ggml && cgo && linux

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


// tensorName maps canonical weight names to model-specific GGUF tensor names.
// Supports: llama (TinyLlama/Llama), qwen3 (Qwen3-0.6B+)
func tensorName(arch, canonical string, il int) string {
	pfx := fmt.Sprintf("blk.%d.", il)
	switch canonical {
	case "attn_norm":
		return pfx + "attn_norm.weight"
	case "wq":
		if arch == "qwen3" { return pfx + "attn_q.weight" }
		return pfx + "attn_q.weight"
	case "wk":
		if arch == "qwen3" { return pfx + "attn_k.weight" }
		return pfx + "attn_k.weight"
	case "wv":
		if arch == "qwen3" { return pfx + "attn_v.weight" }
		return pfx + "attn_v.weight"
	case "wo":
		if arch == "qwen3" { return pfx + "attn_output.weight" }
		return pfx + "attn_output.weight"
	case "ffn_norm":
		return pfx + "ffn_norm.weight"
	case "ffn_gate":
		return pfx + "ffn_gate.weight"
	case "ffn_up":
		return pfx + "ffn_up.weight"
	case "ffn_down":
		return pfx + "ffn_down.weight"
	case "q_norm":
		return pfx + "attn_q_norm.weight"
	case "k_norm":
		return pfx + "attn_k_norm.weight"
	}
	return canonical
}

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

	// Detect model architecture from metadata
	arch := "llama" // default
	if _, ok := g.MetaUint32("qwen3.block_count"); ok {
		arch = "qwen3"
	}
	hasQKNorm := false
	if arch == "qwen3" {
		hasQKNorm = true
	}

	// Get vocab size from token_embd tensor shape (most reliable)
	prefix := arch + "."
	nVocab := 32000
	if t, ok := g.TensorByName("token_embd.weight"); ok && len(t.Shape) >= 2 {
		nVocab = int(t.Shape[1]) // [n_embd, n_vocab]
	}
	if v := getUint32(prefix+"vocab_size", 0); v > 0 { nVocab = v }
	nEmbd    := getUint32(prefix+"embedding_length", getUint32("llama.embedding_length", 2048))
	nHeads   := getUint32(prefix+"attention.head_count", getUint32("llama.attention.head_count", 32))
	nHeadsKV := getUint32(prefix+"attention.head_count_kv", getUint32("llama.attention.head_count_kv", nHeads))
	nLayers  := getUint32(prefix+"block_count", getUint32("llama.block_count", 22))
	nFF      := getUint32(prefix+"feed_forward_length", getUint32("llama.feed_forward_length", 5632))
	nCtx     := 2048
	ropeBase := getFloat32(prefix+"rope.freq_base", getFloat32("llama.rope.freq_base", 10000.0))
	rmsEps   := getFloat32(prefix+"attention.layer_norm_rms_epsilon", getFloat32("llama.attention.layer_norm_rms_epsilon", 1e-5))
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
		OutputType:  func() int { t := qtType("output.weight"); if t == llamagraph.GGMLTypeF32 { t = qtType("token_embd.weight") }; return t }(),
		WQType:      make([]int, nLayers),
		WKType:      make([]int, nLayers),
		WVType:      make([]int, nLayers),
		WOType:      make([]int, nLayers),
		FFNGateType: make([]int, nLayers),
		FFNUpType:   make([]int, nLayers),
		FFNDownType: make([]int, nLayers),
		HasQKNorm:   hasQKNorm,
	}
	for il := 0; il < nLayers; il++ {
		cfg.WQType[il]      = qtType(tensorName(arch, "wq", il))
		cfg.WKType[il]      = qtType(tensorName(arch, "wk", il))
		cfg.WVType[il]      = qtType(tensorName(arch, "wv", il))
		cfg.WOType[il]      = qtType(tensorName(arch, "wo", il))
		cfg.FFNGateType[il] = qtType(tensorName(arch, "ffn_gate", il))
		cfg.FFNUpType[il]   = qtType(tensorName(arch, "ffn_up", il))
		cfg.FFNDownType[il] = qtType(tensorName(arch, "ffn_down", il))
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
	if outW := rawBytes("output.weight"); outW != nil {
		m.SetOutput(outW)
	} else {
		// tie_word_embeddings: output shares token_embd — signal to C layer
		m.SetOutput(rawBytes("token_embd.weight"))
	}

	for il := 0; il < nLayers; il++ {
		m.SetLayerAttnNorm(il, rawBytes(tensorName(arch, "attn_norm", il)))
		m.SetLayerWQ(il, rawBytes(tensorName(arch, "wq", il)))
		m.SetLayerWK(il, rawBytes(tensorName(arch, "wk", il)))
		m.SetLayerWV(il, rawBytes(tensorName(arch, "wv", il)))
		m.SetLayerWO(il, rawBytes(tensorName(arch, "wo", il)))
		m.SetLayerFFNNorm(il, rawBytes(tensorName(arch, "ffn_norm", il)))
		m.SetLayerFFNGate(il, rawBytes(tensorName(arch, "ffn_gate", il)))
		m.SetLayerFFNUp(il, rawBytes(tensorName(arch, "ffn_up", il)))
		m.SetLayerFFNDown(il, rawBytes(tensorName(arch, "ffn_down", il)))
		if hasQKNorm {
			m.SetLayerQNorm(il, rawBytes(tensorName(arch, "q_norm", il)))
			m.SetLayerKNorm(il, rawBytes(tensorName(arch, "k_norm", il)))
		}
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
