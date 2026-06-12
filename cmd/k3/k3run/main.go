// k3run — run LLaMA inference from a GGUF file using the k3 multi-backend stack.
//
// Backend selection order: auto (best available) → SpacemiT → Vulkan → CPU SIMD.
// Each tier is benchmarked on the first forward pass; subsequent tokens are timed.
//
// Usage:
//
//	k3run -model /path/to/model.gguf [-prompt "text"] [-tokens 64] [-backend auto|simd|vulkan|spacemit] [-threads N]
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/backends/spacemit/board"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file (required)")
	promptStr := flag.String("prompt", "Hello! What are you?", "text prompt")
	maxNew := flag.Int("tokens", 64, "maximum new tokens to generate")
	backendName := flag.String("backend", "auto", "backend: auto|simd|vulkan|spacemit")
	benchAll := flag.Bool("bench-all", false, "run all available backends and compare (skips generation output)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "k3run: -model is required")
		os.Exit(1)
	}

	// ── detect available backends ──────────────────────────────────────────
	caps := board.Probe()
	fmt.Println(caps.Summary())
	fmt.Println()

	if *benchAll {
		runBenchAll(*modelPath, *promptStr, *maxNew)
		return
	}

	be := selectBackend(*backendName, caps)
	fmt.Printf("Selected backend: %s\n\n", be.Name())

	runInference(*modelPath, *promptStr, *maxNew, be)
}

func selectBackend(name string, caps board.Capabilities) board.OpBackend {
	switch strings.ToLower(name) {
	case "simd", "cpu":
		return board.SIMDBackend{}
	case "vulkan", "gpu":
		if !caps.Vulkan {
			fmt.Fprintln(os.Stderr, "k3run: Vulkan not available, falling back to SIMD")
			return board.SIMDBackend{}
		}
		return board.VulkanBackend{}
	case "spacemit", "npu", "ort":
		if !caps.SpacemiT {
			fmt.Fprintln(os.Stderr, "k3run: SpacemiT ORT not available, falling back to SIMD")
			return board.SIMDBackend{}
		}
		return board.SpacemiTBackend{}
	default: // auto
		be, _ := board.Select()
		return be
	}
}

func runInference(modelPath, prompt string, maxNew int, be board.OpBackend) {
	fmt.Printf("Loading model from %s …\n", modelPath)
	t0 := time.Now()
	m, err := model.LoadGGUFLlama(modelPath, be)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded in %s  (layers=%d hidden=%d heads=%d/%d)\n\n",
		time.Since(t0).Round(time.Millisecond),
		m.Config.NumLayers, m.Config.HiddenSize,
		m.Config.NumHeads, m.Config.NumKVHeads)

	// Tokenize
	tok, err := gguf.Open(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer open: %v\n", err)
		os.Exit(1)
	}
	tokenizer, err := gguf.NewTokenizer(tok)
	tok.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(1)
	}
	tokenizer.SetModelPath(modelPath)

	fmt.Printf("Prompt: %q\n", prompt)
	ids, err := tokenizer.Encode(prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Tokens: %v (%d)\n\n", ids, len(ids))

	// Generate
	genStart := time.Now()
	var tokenTimes []time.Duration

	// Prefill
	prefillStart := time.Now()
	kvK := make([][]float32, m.Config.NumLayers)
	kvV := make([][]float32, m.Config.NumLayers)
	kvDim := m.Config.NumKVHeads * m.Config.HeadDim
	maxSeq := len(ids) + maxNew
	if maxSeq > m.Config.MaxSeqLen {
		maxSeq = m.Config.MaxSeqLen
	}
	for i := range kvK {
		kvK[i] = make([]float32, maxSeq*kvDim)
		kvV[i] = make([]float32, maxSeq*kvDim)
	}
	state := m.NewForwardState()
	var lastLogits []float32
	for step, tokID := range ids {
		lastLogits = m.ForwardState(state, tokID, step, kvK, kvV)
	}
	prefillDur := time.Since(prefillStart)
	ppTPS := float64(len(ids)) / prefillDur.Seconds()

	// Decode
	generated := make([]int, 0, maxNew)
	step := len(ids)
	curLogits := lastLogits
	for range maxNew {
		next := argmax(curLogits)
		generated = append(generated, next)
		if next == tokenizer.EOS() {
			break
		}
		tStart := time.Now()
		curLogits = m.ForwardState(state, next, step, kvK, kvV)
		tokenTimes = append(tokenTimes, time.Since(tStart))
		step++
		if step >= maxSeq {
			break
		}
	}
	genDur := time.Since(genStart)

	// Results
	output := tokenizer.Decode(generated)
	fmt.Printf("Output: %s\n\n", output)

	var avgTG float64
	if len(tokenTimes) > 0 {
		var total time.Duration
		for _, d := range tokenTimes {
			total += d
		}
		avg := total / time.Duration(len(tokenTimes))
		avgTG = 1.0 / avg.Seconds()
		fmt.Printf("Performance:\n")
		fmt.Printf("  Prefill  (PP-%d): %s  →  %.2f t/s\n", len(ids), prefillDur.Round(time.Millisecond), ppTPS)
		fmt.Printf("  Decode   (TG-%d): avg %s/tok  →  %.2f t/s\n", len(generated), avg.Round(time.Microsecond), avgTG)
		fmt.Printf("  Total:           %s\n", genDur.Round(time.Millisecond))
	} else {
		fmt.Printf("Generated %d tokens in %s\n", len(generated), genDur.Round(time.Millisecond))
	}
}

func runBenchAll(modelPath, prompt string, maxNew int) {
	fmt.Printf("Loading model for benchmark (will be reloaded per backend) …\n\n")

	type result struct {
		name   string
		ppTPS  float64
		tgTPS  float64
		loadMs int64
	}
	var results []result

	backends := []struct {
		name string
		be   board.OpBackend
	}{
		{"CPU-SIMD-RVV", board.SIMDBackend{}},
	}
	caps := board.Probe()
	if caps.Vulkan {
		backends = append(backends, struct {
			name string
			be   board.OpBackend
		}{"Vulkan-PowerVR", board.VulkanBackend{}})
	}
	if caps.SpacemiT {
		backends = append(backends, struct {
			name string
			be   board.OpBackend
		}{"SpacemiT-ORT", board.SpacemiTBackend{}})
	}

	// Tokenize once
	tok, err := gguf.Open(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer open: %v\n", err)
		os.Exit(1)
	}
	tokenizer, err := gguf.NewTokenizer(tok)
	tok.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(1)
	}
	tokenizer.SetModelPath(modelPath)
	ids, err := tokenizer.Encode(prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Prompt: %q  (%d tokens)\n\n", prompt, len(ids))

	for _, b := range backends {
		fmt.Printf("── %s ──────────────────────────────────────\n", b.name)
		t0 := time.Now()
		m, err := model.LoadGGUFLlama(modelPath, b.be)
		loadMs := time.Since(t0).Milliseconds()
		if err != nil {
			fmt.Printf("  LOAD FAILED: %v\n\n", err)
			continue
		}
		fmt.Printf("  loaded in %dms\n", loadMs)

		kvK, kvV, maxSeq := allocKV(m, len(ids)+maxNew)

		// Prefill
		state := m.NewForwardState()
		var lastLogits []float32
		ppStart := time.Now()
		for step, tokID := range ids {
			lastLogits = m.ForwardState(state, tokID, step, kvK, kvV)
		}
		ppDur := time.Since(ppStart)
		ppTPS := float64(len(ids)) / ppDur.Seconds()

		// Decode
		var tgTimes []time.Duration
		step := len(ids)
		curLogits := lastLogits
		for range maxNew {
			next := argmax(curLogits)
			if next == tokenizer.EOS() {
				break
			}
			tStart := time.Now()
			curLogits = m.ForwardState(state, next, step, kvK, kvV)
			tgTimes = append(tgTimes, time.Since(tStart))
			step++
			if step >= maxSeq {
				break
			}
		}
		var tgTPS float64
		if len(tgTimes) > 0 {
			var total time.Duration
			for _, d := range tgTimes {
				total += d
			}
			avg := total / time.Duration(len(tgTimes))
			tgTPS = 1.0 / avg.Seconds()
		}
		fmt.Printf("  PP-%d : %.2f t/s\n", len(ids), ppTPS)
		fmt.Printf("  TG-%d : %.2f t/s\n", maxNew, tgTPS)
		fmt.Println()

		results = append(results, result{b.name, ppTPS, tgTPS, loadMs})
	}

	// Summary table
	fmt.Println("── Summary ──────────────────────────────────────")
	fmt.Printf("  %-24s  %10s  %10s  %8s\n", "Backend", "PP t/s", "TG t/s", "Load ms")
	fmt.Printf("  %-24s  %10s  %10s  %8s\n", strings.Repeat("-", 24), strings.Repeat("-", 10), strings.Repeat("-", 10), strings.Repeat("-", 8))
	for _, r := range results {
		fmt.Printf("  %-24s  %10.2f  %10.2f  %8d\n", r.name, r.ppTPS, r.tgTPS, r.loadMs)
	}
	// SpacemiT llama-bench reference
	fmt.Printf("\n  SpacemiT llama-bench ref:   PP-128=137.47 t/s   TG-64=36.60 t/s\n")
}

func allocKV(m *model.GGUFLlama, maxSeq int) (kvK, kvV [][]float32, _ int) {
	kvDim := m.Config.NumKVHeads * m.Config.HeadDim
	if maxSeq > m.Config.MaxSeqLen {
		maxSeq = m.Config.MaxSeqLen
	}
	kvK = make([][]float32, m.Config.NumLayers)
	kvV = make([][]float32, m.Config.NumLayers)
	for i := range kvK {
		kvK[i] = make([]float32, maxSeq*kvDim)
		kvV[i] = make([]float32, maxSeq*kvDim)
	}
	return kvK, kvV, maxSeq
}

func argmax(x []float32) int {
	best := 0
	for i, v := range x[1:] {
		if v > x[best] {
			best = i + 1
		}
	}
	return best
}

var _ = math.Sqrt // ensure math is used
