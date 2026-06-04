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
	expectKVSmokeLayer := flag.Int("expect-kv-smoke-layer", -1, "fail unless TurboQuant synthetic KV smoke selects this layer")
	expectKVSmokeCompressed := flag.Int("expect-kv-smoke-compressed", -1, "fail unless TurboQuant synthetic KV smoke compressed count matches")
	expectKVSmokeFull := flag.Int("expect-kv-smoke-full", -1, "fail unless TurboQuant synthetic KV smoke full count matches")
	expectKVSmokeBytes := flag.Int64("expect-kv-smoke-bytes", -1, "fail unless TurboQuant synthetic KV smoke stored memory bytes match")
	expectKVSmokeScratchBytes := flag.Int64("expect-kv-smoke-scratch-bytes", -1, "fail unless TurboQuant synthetic KV smoke scratch bytes match")
	expectKVSmokeTotalBytes := flag.Int64("expect-kv-smoke-total-bytes", -1, "fail unless TurboQuant synthetic KV smoke total bytes match")
	expectGeneratedCSV := flag.String("expect-generated", "", "comma-separated generated token IDs expected from generation smoke")
	expectDecoded := flag.String("expect-decoded", "", "decoded generated text expected from generation smoke; implies -decode validation when tokenizer is available")
	expectRuntimeFloatBytes := flag.Int64("expect-runtime-float-bytes", -1, "fail unless planned runtime F32 KV bytes match this value")
	expectRuntimeCompressedBytes := flag.Int64("expect-runtime-compressed-bytes", -1, "fail unless planned runtime compressed KV bytes match this value")
	expectRuntimeScratchBytes := flag.Int64("expect-runtime-scratch-bytes", -1, "fail unless planned runtime TurboQuant scratch bytes match this value")
	expectRuntimeTotalBytes := flag.Int64("expect-runtime-total-bytes", -1, "fail unless planned runtime total KV+scratch bytes match this value")
	expectKVFloatBytes := flag.Int64("expect-kv-float-bytes", -1, "fail unless benchmark F32 KV allocated bytes match this value")
	expectKVCompressedBytes := flag.Int64("expect-kv-compressed-bytes", -1, "fail unless benchmark compressed KV stored bytes match this value")
	expectKVScratchBytes := flag.Int64("expect-kv-scratch-bytes", -1, "fail unless benchmark compressed KV scratch bytes match this value")
	expectKVTotalBytes := flag.Int64("expect-kv-total-bytes", -1, "fail unless benchmark total KV+scratch bytes match this value")
	expectSIMDRotation := flag.Bool("expect-simd-rotation", false, "fail unless native SIMD dot-product rotation support is available")
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
	fmt.Printf("loaded architecture=%s layers=%d hidden=%d experts=%d active=%d qwennext=%v reap=%.2f reap_source=%s reap_static_masks=%v reap_default_experts=%d reap_layer_masks=%d reap_layer_experts=%d bos=%d eos=%d in %.2fs\n", m.Config.Architecture, m.Config.NumLayers, m.Config.HiddenSize, m.Config.NumExperts, m.Config.NumExpertsPerTok, m.Config.IsQwenNextHybridGGUF(), reap.PruneRatio, reap.Source, reap.HasStaticMasks, reap.DefaultExperts, reap.LayerMasks, reap.LayerExpertTotal, m.Config.BOSTokenID, m.Config.EOSTokenID, time.Since(t0).Seconds())
	if *cacheTypeK != "" || *cacheTypeV != "" || *kvResidualWindow >= 0 {
		plan, err := m.TurboQuantPlan(*cacheTypeK, *cacheTypeV, *kvResidualWindow)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: turboquant plan failed: %v\n", err)
			os.Exit(1)
		}
		if err := checkExpectedSIMDRotation(plan.SIMDArch, plan.SIMDRotation, plan.SIMDVec, *expectSIMDRotation); err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("turboquant enabled=%v key_bits=%d value_bits=%d residual=%d layers=%d cache_layers=%d protected_cache_layers=%d max_seq=%d kv_dim=%d full_kv_bytes=%d estimated_kv_bytes=%d estimated_saved_kv_bytes=%d estimated_kv_ratio=%.4f simd_arch=%s simd_rotation=%v simd_vec=%v simd_avx2=%v simd_neon=%v simd_rvv=%v\n", plan.Enabled, plan.KeyBits, plan.ValueBits, plan.ResidualWindow, plan.Layers, plan.CacheLayers, plan.ProtectedCacheLayers, plan.MaxSeqLen, plan.KVDim, plan.FullKVBytes, plan.EstimatedKVBytes, plan.EstimatedSavedKVBytes, plan.EstimatedKVRatio, plan.SIMDArch, plan.SIMDRotation, plan.SIMDVec, plan.SIMDAVX2, plan.SIMDNEON, plan.SIMDRVv)
		planPromptIDs, _ := resolvePromptIDs(*path, *promptText, *promptIDsCSV)
		if rt, err := m.GenerationKVRuntimePlan(len(planPromptIDs), *maxNew, model.GGUFGenerationOptions{CacheTypeK: *cacheTypeK, CacheTypeV: *cacheTypeV, KVResidualWindow: *kvResidualWindow}); err == nil {
			if err := checkExpectedRuntimeKV(rt, *expectRuntimeFloatBytes, *expectRuntimeCompressedBytes, *expectRuntimeScratchBytes, *expectRuntimeTotalBytes); err != nil {
				fmt.Fprintf(os.Stderr, "ggufsmoke: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("turboquant_runtime_kv max_seq=%d float_layers=%d compressed_layers=%d protected_compressed_layers=%d float_alloc_bytes=%d compressed_full_bytes=%d compressed_estimated_bytes=%d compressed_saved_bytes=%d compressed_ratio=%.4f estimated_scratch_bytes=%d estimated_total_bytes=%d simd_arch=%s simd_rotation=%v simd_vec=%v simd_avx2=%v simd_neon=%v simd_rvv=%v\n", rt.MaxSeq, rt.FloatKVLayers, rt.CompressedKVLayers, rt.ProtectedCompressedLayers, rt.FloatKVBytesAllocated, rt.FullCompressedKVBytes, rt.EstimatedCompressedKVBytes, rt.SavedCompressedKVBytes, rt.CompressedKVRatio, rt.EstimatedScratchBytes, rt.EstimatedTotalBytes, rt.SIMDArch, rt.SIMDRotation, rt.SIMDVec, rt.SIMDAVX2, rt.SIMDNEON, rt.SIMDRVv)
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
					stats := c.Stats()
					if err := checkExpectedKVSmoke(idx, stats.CompressedCount, stats.FullCount, stats.StoredBytes, stats.ScratchBytes, stats.TotalBytes, *expectKVSmokeLayer, *expectKVSmokeCompressed, *expectKVSmokeFull, *expectKVSmokeBytes, *expectKVSmokeScratchBytes, *expectKVSmokeTotalBytes); err != nil {
						fmt.Fprintf(os.Stderr, "ggufsmoke: %v\n", err)
						os.Exit(1)
					}
					fmt.Printf("turboquant_cache_smoke tokens=%d layer=%d seq=%d compressed=%d full=%d bytes=%d stored_bytes=%d scratch_bytes=%d total_bytes=%d\n", *kvSmokeTokens, idx, stats.SeqLen, stats.CompressedCount, stats.FullCount, stats.StoredBytes, stats.StoredBytes, stats.ScratchBytes, stats.TotalBytes)
				}
			}
		}
	}
	if *loadOnly {
		return
	}
	promptIDs, tok := resolvePromptIDs(*path, *promptText, *promptIDsCSV)
	if (*decodeOutput || *expectDecoded != "") && tok == nil {
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
			if err := checkExpectedGenerated(ids, *expectGeneratedCSV); err != nil {
				fmt.Fprintf(os.Stderr, "ggufsmoke: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("generated=%v\n", ids)
			if err := checkExpectedDecoded(ids, tok, *expectDecoded); err != nil {
				fmt.Fprintf(os.Stderr, "ggufsmoke: %v\n", err)
				os.Exit(1)
			}
			if *decodeOutput && tok != nil {
				fmt.Printf("decoded=%q\n", tok.Decode(ids))
			}
			if err := checkExpectedBenchKV(stats, *expectKVFloatBytes, *expectKVCompressedBytes, *expectKVScratchBytes, *expectKVTotalBytes); err != nil {
				fmt.Fprintf(os.Stderr, "ggufsmoke: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("prefill_tokens=%d prefill_s=%.3f prefill_tps=%.2f decode_tokens=%d decode_s=%.3f decode_tps=%.2f kv_compressed_layers=%d kv_seq=%d kv_compressed_count=%d kv_full_count=%d kv_float_bytes=%d kv_compressed_bytes=%d kv_scratch_bytes=%d kv_total_bytes=%d\n", stats.PrefillTokens, stats.PrefillSeconds, stats.PrefillTPS(), stats.DecodeTokens, stats.DecodeSeconds, stats.DecodeTPS(), stats.KVCompressedLayers, stats.KVSeqLen, stats.KVCompressedCount, stats.KVFullCount, stats.KVFloatBytes, stats.KVCompressedBytes, stats.KVScratchBytes, stats.KVTotalBytes)
			return
		}
		ids, err := m.GenerateWithOptions(promptIDs, *maxNew, model.GGUFGenerationOptions{CacheTypeK: *cacheTypeK, CacheTypeV: *cacheTypeV, KVResidualWindow: *kvResidualWindow})
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: generate failed: %v\n", err)
			os.Exit(1)
		}
		if err := checkExpectedGenerated(ids, *expectGeneratedCSV); err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("generated=%v\n", ids)
		if err := checkExpectedDecoded(ids, tok, *expectDecoded); err != nil {
			fmt.Fprintf(os.Stderr, "ggufsmoke: %v\n", err)
			os.Exit(1)
		}
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

func checkExpectedSIMDRotation(arch string, rotation, vec, expect bool) error {
	if expect && !rotation {
		return fmt.Errorf("SIMD rotation unavailable arch=%s vec=%v", arch, vec)
	}
	if expect {
		fmt.Printf("expected_simd_rotation_ok=%v arch=%s\n", rotation, arch)
	}
	return nil
}

func checkExpectedKVSmoke(layer, compressed, full int, storedBytes, scratchBytes, totalBytes int64, expectLayer, expectCompressed, expectFull int, expectBytes, expectScratchBytes, expectTotalBytes int64) error {
	if expectLayer >= 0 && layer != expectLayer {
		return fmt.Errorf("KV smoke layer mismatch got=%d want=%d", layer, expectLayer)
	}
	if expectCompressed >= 0 && compressed != expectCompressed {
		return fmt.Errorf("KV smoke compressed count mismatch got=%d want=%d", compressed, expectCompressed)
	}
	if expectFull >= 0 && full != expectFull {
		return fmt.Errorf("KV smoke full count mismatch got=%d want=%d", full, expectFull)
	}
	if expectBytes >= 0 && storedBytes != expectBytes {
		return fmt.Errorf("KV smoke stored bytes mismatch got=%d want=%d", storedBytes, expectBytes)
	}
	if expectScratchBytes >= 0 && scratchBytes != expectScratchBytes {
		return fmt.Errorf("KV smoke scratch bytes mismatch got=%d want=%d", scratchBytes, expectScratchBytes)
	}
	if expectTotalBytes >= 0 && totalBytes != expectTotalBytes {
		return fmt.Errorf("KV smoke total bytes mismatch got=%d want=%d", totalBytes, expectTotalBytes)
	}
	if expectLayer >= 0 {
		fmt.Printf("expected_kv_smoke_layer_ok=%d\n", expectLayer)
	}
	if expectCompressed >= 0 {
		fmt.Printf("expected_kv_smoke_compressed_ok=%d\n", expectCompressed)
	}
	if expectFull >= 0 {
		fmt.Printf("expected_kv_smoke_full_ok=%d\n", expectFull)
	}
	if expectBytes >= 0 {
		fmt.Printf("expected_kv_smoke_bytes_ok=%d\n", expectBytes)
	}
	if expectScratchBytes >= 0 {
		fmt.Printf("expected_kv_smoke_scratch_bytes_ok=%d\n", expectScratchBytes)
	}
	if expectTotalBytes >= 0 {
		fmt.Printf("expected_kv_smoke_total_bytes_ok=%d\n", expectTotalBytes)
	}
	return nil
}

func checkExpectedRuntimeKV(plan model.GGUFGenerationKVRuntimePlan, expectFloat, expectCompressed, expectScratch, expectTotal int64) error {
	if expectFloat >= 0 && plan.FloatKVBytesAllocated != expectFloat {
		return fmt.Errorf("runtime F32 KV bytes mismatch got=%d want=%d", plan.FloatKVBytesAllocated, expectFloat)
	}
	if expectCompressed >= 0 && plan.EstimatedCompressedKVBytes != expectCompressed {
		return fmt.Errorf("runtime compressed KV bytes mismatch got=%d want=%d", plan.EstimatedCompressedKVBytes, expectCompressed)
	}
	if expectScratch >= 0 && plan.EstimatedScratchBytes != expectScratch {
		return fmt.Errorf("runtime scratch bytes mismatch got=%d want=%d", plan.EstimatedScratchBytes, expectScratch)
	}
	if expectTotal >= 0 && plan.EstimatedTotalBytes != expectTotal {
		return fmt.Errorf("runtime total bytes mismatch got=%d want=%d", plan.EstimatedTotalBytes, expectTotal)
	}
	if expectFloat >= 0 {
		fmt.Printf("expected_runtime_float_bytes_ok=%d\n", expectFloat)
	}
	if expectCompressed >= 0 {
		fmt.Printf("expected_runtime_compressed_bytes_ok=%d\n", expectCompressed)
	}
	if expectScratch >= 0 {
		fmt.Printf("expected_runtime_scratch_bytes_ok=%d\n", expectScratch)
	}
	if expectTotal >= 0 {
		fmt.Printf("expected_runtime_total_bytes_ok=%d\n", expectTotal)
	}
	return nil
}

func checkExpectedBenchKV(stats generationBenchStats, expectFloat, expectCompressed, expectScratch, expectTotal int64) error {
	if expectFloat >= 0 && stats.KVFloatBytes != expectFloat {
		return fmt.Errorf("benchmark F32 KV bytes mismatch got=%d want=%d", stats.KVFloatBytes, expectFloat)
	}
	if expectCompressed >= 0 && stats.KVCompressedBytes != expectCompressed {
		return fmt.Errorf("benchmark compressed KV bytes mismatch got=%d want=%d", stats.KVCompressedBytes, expectCompressed)
	}
	if expectScratch >= 0 && stats.KVScratchBytes != expectScratch {
		return fmt.Errorf("benchmark scratch KV bytes mismatch got=%d want=%d", stats.KVScratchBytes, expectScratch)
	}
	if expectTotal >= 0 && stats.KVTotalBytes != expectTotal {
		return fmt.Errorf("benchmark total KV bytes mismatch got=%d want=%d", stats.KVTotalBytes, expectTotal)
	}
	if expectFloat >= 0 {
		fmt.Printf("expected_kv_float_bytes_ok=%d\n", expectFloat)
	}
	if expectCompressed >= 0 {
		fmt.Printf("expected_kv_compressed_bytes_ok=%d\n", expectCompressed)
	}
	if expectScratch >= 0 {
		fmt.Printf("expected_kv_scratch_bytes_ok=%d\n", expectScratch)
	}
	if expectTotal >= 0 {
		fmt.Printf("expected_kv_total_bytes_ok=%d\n", expectTotal)
	}
	return nil
}

func checkExpectedDecoded(ids []int, tok *gguf.Tokenizer, expected string) error {
	if expected == "" {
		return nil
	}
	if tok == nil {
		return fmt.Errorf("-expect-decoded requires a GGUF tokenizer")
	}
	got := tok.Decode(ids)
	if got != expected {
		return fmt.Errorf("decoded mismatch got=%q want=%q", got, expected)
	}
	fmt.Printf("expected_decoded_ok=%q\n", expected)
	return nil
}

func checkExpectedGenerated(got []int, expectedCSV string) error {
	if strings.TrimSpace(expectedCSV) == "" {
		return nil
	}
	want, err := parsePromptIDs(expectedCSV)
	if err != nil {
		return fmt.Errorf("bad -expect-generated: %w", err)
	}
	if len(got) != len(want) {
		return fmt.Errorf("generated mismatch got=%v want=%v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("generated mismatch got=%v want=%v", got, want)
		}
	}
	fmt.Printf("expected_generated_ok=%v\n", want)
	return nil
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
	PrefillTokens      int
	PrefillSeconds     float64
	DecodeTokens       int
	DecodeSeconds      float64
	KVCompressedLayers int
	KVSeqLen           int
	KVCompressedCount  int
	KVFullCount        int
	KVFloatBytes       int64
	KVCompressedBytes  int64
	KVScratchBytes     int64
	KVTotalBytes       int64
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
	stats.KVFloatBytes, stats.KVCompressedLayers, stats.KVSeqLen, stats.KVCompressedCount, stats.KVFullCount, stats.KVCompressedBytes, stats.KVScratchBytes, stats.KVTotalBytes = ggufBenchKVStats(kvK, kvV, compressedKV)
	return generated, stats, nil
}

func ggufBenchKVStats(kvK, kvV [][]float32, compressedKV []*kv.CompressedKVCache) (floatBytes int64, compressedLayers, seqLen, compressedCount, fullCount int, compressedBytes, scratchBytes, totalBytes int64) {
	for _, x := range kvK {
		floatBytes += int64(len(x)) * 4
	}
	for _, x := range kvV {
		floatBytes += int64(len(x)) * 4
	}
	compressedStats := kv.AggregateCompressedKVCacheStats(compressedKV)
	compressedLayers = compressedStats.Layers
	seqLen = compressedStats.SeqLen
	compressedCount = compressedStats.CompressedCount
	fullCount = compressedStats.FullCount
	compressedBytes = compressedStats.StoredBytes
	scratchBytes = compressedStats.ScratchBytes
	return floatBytes, compressedLayers, seqLen, compressedCount, fullCount, compressedBytes, scratchBytes, floatBytes + compressedBytes + scratchBytes
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
