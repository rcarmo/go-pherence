package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

func main() {
	path := flag.String("model", "", "GGUF model path")
	loadOnly := flag.Bool("load-only", false, "load model and exit")
	bench := flag.Bool("bench", false, "print prefill/decode timing for smoke generation")
	maxNew := flag.Int("max-new", 0, "run greedy generation for N tokens from -prompt-ids")
	promptText := flag.String("prompt", "", "text prompt to tokenize via GGUF/llama-tokenize; overrides -prompt-ids")
	decodeOutput := flag.Bool("decode", false, "decode generated token IDs using GGUF tokenizer vocabulary")
	promptIDsCSV := flag.String("prompt-ids", "0", "comma-separated prompt token IDs for forward/generation smoke")
	quant := flag.Bool("ggml-quant", true, "keep quantized GGUF matrices instead of full F32 expansion")
	cacheTypeK := flag.String("cache-type-k", "", "native TurboQuant key cache type (turbo4, q8_0, f16)")
	cacheTypeV := flag.String("cache-type-v", "", "native TurboQuant value cache type (turbo2, q4_0, f16)")
	kvResidualWindow := flag.Int("kv-residual-window", -1, "native TurboQuant residual window")
	kvSmokeTokens := flag.Int("kv-smoke-tokens", 0, "append N synthetic K/V positions to native TurboQuant GGUF caches")
	flag.Parse()
	if *path == "" {
		fmt.Fprintln(os.Stderr, "usage: ggufsmoke -model model.gguf [-load-only]")
		os.Exit(2)
	}
	if *quant {
		os.Setenv("GO_PHERENCE_GGML_QUANT", "1")
	}
	t0 := time.Now()
	m, err := model.LoadGGUFLlama(*path, k3.SIMDBackend{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "ggufsmoke: load failed: %v\n", err)
		os.Exit(1)
	}
	reap := m.REAP.Summary()
	fmt.Printf("loaded architecture=%s layers=%d hidden=%d experts=%d active=%d qwennext=%v reap=%.2f reap_static_masks=%v reap_default_experts=%d reap_layer_masks=%d reap_layer_experts=%d bos=%d eos=%d in %.2fs\n", m.Config.Architecture, m.Config.NumLayers, m.Config.HiddenSize, m.Config.NumExperts, m.Config.NumExpertsPerTok, m.Config.IsQwenNextHybridGGUF(), reap.PruneRatio, reap.HasStaticMasks, reap.DefaultExperts, reap.LayerMasks, reap.LayerExpertTotal, m.Config.BOSTokenID, m.Config.EOSTokenID, time.Since(t0).Seconds())
	if *cacheTypeK != "" || *cacheTypeV != "" || *kvResidualWindow >= 0 {
		plan, err := m.TurboQuantPlan(*cacheTypeK, *cacheTypeV, *kvResidualWindow)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: turboquant plan failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("turboquant enabled=%v key_bits=%d value_bits=%d residual=%d layers=%d cache_layers=%d protected_cache_layers=%d max_seq=%d kv_dim=%d full_kv_bytes=%d estimated_kv_bytes=%d estimated_saved_kv_bytes=%d estimated_kv_ratio=%.4f\n", plan.Enabled, plan.KeyBits, plan.ValueBits, plan.ResidualWindow, plan.Layers, plan.CacheLayers, plan.ProtectedCacheLayers, plan.MaxSeqLen, plan.KVDim, plan.FullKVBytes, plan.EstimatedKVBytes, plan.EstimatedSavedKVBytes, plan.EstimatedKVRatio)
		planPromptIDs, _ := resolvePromptIDs(*path, *promptText, *promptIDsCSV)
		if rt, err := m.GenerationKVRuntimePlan(len(planPromptIDs), *maxNew, model.GGUFGenerationOptions{CacheTypeK: *cacheTypeK, CacheTypeV: *cacheTypeV, KVResidualWindow: *kvResidualWindow}); err == nil {
			fmt.Printf("turboquant_runtime_kv max_seq=%d float_layers=%d compressed_layers=%d protected_compressed_layers=%d float_alloc_bytes=%d compressed_full_bytes=%d compressed_estimated_bytes=%d compressed_saved_bytes=%d compressed_ratio=%.4f\n", rt.MaxSeq, rt.FloatKVLayers, rt.CompressedKVLayers, rt.ProtectedCompressedLayers, rt.FloatKVBytesAllocated, rt.FullCompressedKVBytes, rt.EstimatedCompressedKVBytes, rt.SavedCompressedKVBytes, rt.CompressedKVRatio)
		}
		if *kvSmokeTokens > 0 {
			caches, err := m.NewTurboQuantKVCache(*cacheTypeK, *cacheTypeV, *kvResidualWindow)
			if err != nil {
				fmt.Fprintf(os.Stderr, "ggufsmoke: turboquant cache failed: %v\n", err)
				os.Exit(1)
			}
			if len(caches) > 0 {
				k := make([]float32, plan.KVDim)
				v := make([]float32, plan.KVDim)
				for i := 0; i < *kvSmokeTokens; i++ {
					for j := range k {
						k[j] = float32(i+1) / float32(j+1)
						v[j] = float32(i+2) / float32(j+2)
					}
					for _, c := range caches {
						if c != nil {
							c.Append(k, v)
						}
					}
				}
				idx := -1
				for i, c := range caches {
					if c != nil && c.CompressedCount() > 0 {
						idx = i
						break
					}
				}
				if idx < 0 {
					for i, c := range caches {
						if c != nil {
							idx = i
							break
						}
					}
				}
				if idx >= 0 {
					c := caches[idx]
					fmt.Printf("turboquant_cache_smoke tokens=%d layer=%d seq=%d compressed=%d full=%d bytes=%d\n", *kvSmokeTokens, idx, c.SeqLen(), c.CompressedCount(), c.FullCount(), c.MemoryBytes())
				}
			}
		}
	}
	if *loadOnly {
		return
	}
	promptIDs, tok := resolvePromptIDs(*path, *promptText, *promptIDsCSV)
	if *decodeOutput && tok == nil {
		tok = loadGGUFTokenizerForDecode(*path)
	}
	if len(promptIDs) == 0 {
		fmt.Fprintln(os.Stderr, "ggufsmoke: empty prompt")
		os.Exit(2)
	}
	if *maxNew > 0 {
		if *bench {
			ids, stats, err := runGenerationBench(m, promptIDs, *maxNew, model.GGUFGenerationOptions{CacheTypeK: *cacheTypeK, CacheTypeV: *cacheTypeV, KVResidualWindow: *kvResidualWindow})
			if err != nil {
				fmt.Fprintf(os.Stderr, "ggufsmoke: generate failed: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("generated=%v\n", ids)
			if *decodeOutput && tok != nil {
				fmt.Printf("decoded=%q\n", tok.Decode(ids))
			}
			fmt.Printf("prefill_tokens=%d prefill_s=%.3f prefill_tps=%.2f decode_tokens=%d decode_s=%.3f decode_tps=%.2f kv_float_bytes=%d kv_compressed_bytes=%d\n", stats.PrefillTokens, stats.PrefillSeconds, stats.PrefillTPS(), stats.DecodeTokens, stats.DecodeSeconds, stats.DecodeTPS(), stats.KVFloatBytes, stats.KVCompressedBytes)
			return
		}
		ids, err := m.GenerateWithOptions(promptIDs, *maxNew, model.GGUFGenerationOptions{CacheTypeK: *cacheTypeK, CacheTypeV: *cacheTypeV, KVResidualWindow: *kvResidualWindow})
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: generate failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("generated=%v\n", ids)
		if *decodeOutput && tok != nil {
			fmt.Printf("decoded=%q\n", tok.Decode(ids))
		}
		return
	}
	state := m.NewForwardState()
	kvDim := m.Config.NumKVHeads * m.Config.HeadDim
	kvK := make([][]float32, m.Config.NumLayers)
	kvV := make([][]float32, m.Config.NumLayers)
	for i := range kvK {
		kvK[i] = make([]float32, kvDim)
		kvV[i] = make([]float32, kvDim)
	}
	logits := m.ForwardState(state, promptIDs[0], 0, kvK, kvV)
	fmt.Printf("forward logits=%d\n", len(logits))
}

func parsePromptIDs(csv string) ([]int, error) {
	parts := strings.Split(csv, ",")
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		id, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		if id < 0 {
			return nil, fmt.Errorf("negative token id %d", id)
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("no prompt ids")
	}
	return ids, nil
}

type generationBenchStats struct {
	PrefillTokens     int
	PrefillSeconds    float64
	DecodeTokens      int
	DecodeSeconds     float64
	KVFloatBytes      int64
	KVCompressedBytes int64
}

func (s generationBenchStats) PrefillTPS() float64 {
	if s.PrefillSeconds <= 0 {
		return 0
	}
	return float64(s.PrefillTokens) / s.PrefillSeconds
}

func (s generationBenchStats) DecodeTPS() float64 {
	if s.DecodeSeconds <= 0 {
		return 0
	}
	return float64(s.DecodeTokens) / s.DecodeSeconds
}

func runGenerationBench(m *model.GGUFLlama, promptIDs []int, maxNew int, opts model.GGUFGenerationOptions) ([]int, generationBenchStats, error) {
	var stats generationBenchStats
	if m == nil || len(promptIDs) == 0 || maxNew <= 0 {
		return nil, stats, nil
	}
	cfg := m.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	maxSeq := len(promptIDs) + maxNew
	if maxSeq > cfg.MaxSeqLen {
		maxSeq = cfg.MaxSeqLen
	}
	var compressedKV []*kv.CompressedKVCache
	if opts.CacheTypeK != "" || opts.CacheTypeV != "" || opts.KVResidualWindow >= 0 {
		caches, err := m.NewTurboQuantKVCache(opts.CacheTypeK, opts.CacheTypeV, opts.KVResidualWindow)
		if err != nil {
			return nil, stats, err
		}
		compressedKV = caches
	}
	kvK := make([][]float32, cfg.NumLayers)
	kvV := make([][]float32, cfg.NumLayers)
	for i := range kvK {
		if i < len(compressedKV) && compressedKV[i] != nil {
			continue
		}
		kvK[i] = make([]float32, maxSeq*kvDim)
		kvV[i] = make([]float32, maxSeq*kvDim)
	}
	state := m.NewForwardState()
	state.SetCompressedKVForSmoke(compressedKV)
	var logits []float32
	p0 := time.Now()
	for step, tok := range promptIDs {
		logits = m.ForwardState(state, tok, step, kvK, kvV)
	}
	stats.PrefillTokens = len(promptIDs)
	stats.PrefillSeconds = time.Since(p0).Seconds()
	var generated []int
	step := len(promptIDs) - 1
	d0 := time.Now()
	for range maxNew {
		next := argmaxLocal(logits)
		generated = append(generated, next)
		if cfg.IsEOS(next) {
			break
		}
		step++
		if step >= maxSeq {
			break
		}
		logits = m.ForwardState(state, next, step, kvK, kvV)
	}
	stats.DecodeTokens = len(generated)
	stats.DecodeSeconds = time.Since(d0).Seconds()
	stats.KVFloatBytes, stats.KVCompressedBytes = ggufBenchKVBytes(kvK, kvV, compressedKV)
	return generated, stats, nil
}

func ggufBenchKVBytes(kvK, kvV [][]float32, compressedKV []*kv.CompressedKVCache) (floatBytes, compressedBytes int64) {
	for _, x := range kvK {
		floatBytes += int64(len(x)) * 4
	}
	for _, x := range kvV {
		floatBytes += int64(len(x)) * 4
	}
	for _, c := range compressedKV {
		if c != nil {
			compressedBytes += c.MemoryBytes()
		}
	}
	return floatBytes, compressedBytes
}

func argmaxLocal(x []float32) int {
	best := 0
	for i, v := range x[1:] {
		if v > x[best] {
			best = i + 1
		}
	}
	return best
}

func resolvePromptIDs(modelPath, promptText, promptIDsCSV string) ([]int, *gguf.Tokenizer) {
	if promptText != "" {
		g, err := gguf.Open(modelPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: tokenizer open failed: %v\n", err)
			os.Exit(1)
		}
		defer g.Close()
		tok, err := gguf.NewTokenizer(g)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: tokenizer load failed: %v\n", err)
			os.Exit(1)
		}
		tok.SetModelPath(modelPath)
		ids, err := tok.Encode(promptText)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: tokenize failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("prompt_tokens=%v\n", ids)
		return ids, tok
	}
	ids, err := parsePromptIDs(promptIDsCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ggufsmoke: bad -prompt-ids: %v\n", err)
		os.Exit(2)
	}
	return ids, nil
}

func loadGGUFTokenizerForDecode(modelPath string) *gguf.Tokenizer {
	g, err := gguf.Open(modelPath)
	if err != nil {
		return nil
	}
	defer g.Close()
	tok, err := gguf.NewTokenizer(g)
	if err != nil {
		return nil
	}
	tok.SetModelPath(modelPath)
	return tok
}
