package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
	"github.com/rcarmo/go-pherence/runtime/kv"
	"github.com/rcarmo/go-pherence/tensor"
)

var qwen36UseGPULMHead bool
var qwen36LMHeadLogitsScratch []float32
var qwen36LMHeadStats Qwen36LMHeadStats

type QwenGPUWindowEstimate struct {
	Tokens                   int     `json:"tokens"`
	TransientBytesPerToken   float64 `json:"transient_bytes_per_token"`
	TransientUploadsPerToken float64 `json:"transient_uploads_per_token"`
	ByteReduction            float64 `json:"byte_reduction"`
	UploadReduction          float64 `json:"upload_reduction"`
}

type Qwen36LMHeadStats struct {
	Calls        int64   `json:"calls,omitempty"`
	GPUMillis    int64   `json:"gpu_ms,omitempty"`
	CPUMillis    int64   `json:"cpu_ms,omitempty"`
	DownloadRows int     `json:"download_rows,omitempty"`
	AvgMS        float64 `json:"avg_ms,omitempty"`
}

type TopLogit struct {
	ID    int     `json:"id"`
	Logit float32 `json:"logit"`
}

type Qwen36Summary struct {
	KVHit             bool    `json:"kv_hit,omitempty"`
	KVReuseEfficiency float64 `json:"kv_reuse_efficiency,omitempty"`
	KVSpeedupVsCold   float64 `json:"kv_speedup_vs_cold,omitempty"`
	MTPAcceptanceRate float64 `json:"mtp_acceptance_rate,omitempty"`
	MTPSpeedupVsSeq   float64 `json:"mtp_speedup_vs_sequential,omitempty"`
	DecodeTPS         float64 `json:"decode_tokens_per_second,omitempty"`
}

type mtpGenerateStats struct {
	Accepted            int
	Drafted             int
	Rounds              int
	BonusTokens         int
	VerifierChunks      int
	VerifierLayerChunks int
	AdaptiveFallback    bool
}

type Report struct {
	ModelDir                    string                     `json:"model_dir"`
	Prompt                      string                     `json:"prompt,omitempty"`
	InputIDs                    []int                      `json:"input_ids"`
	GeneratedIDs                []int                      `json:"generated_ids,omitempty"`
	Decoded                     string                     `json:"decoded,omitempty"`
	TokenID                     int                        `json:"token_id,omitempty"`
	NextID                      int                        `json:"next_id"`
	Logit                       float32                    `json:"logit"`
	HiddenAbsSum                float32                    `json:"hidden_abs_sum"`
	MTPOutputLen                int                        `json:"mtp_output_len,omitempty"`
	MTPAbsSum                   float32                    `json:"mtp_abs_sum,omitempty"`
	MTPNextID                   int                        `json:"mtp_next_id,omitempty"`
	MTPLogit                    float32                    `json:"mtp_logit,omitempty"`
	MTPVerifierNextID           int                        `json:"mtp_verifier_next_id,omitempty"`
	MTPAcceptedByGreedy         bool                       `json:"mtp_accepted_by_greedy"`
	PrefillMTPNextID            int                        `json:"prefill_mtp_next_id,omitempty"`
	PrefillMTPLogit             float32                    `json:"prefill_mtp_logit,omitempty"`
	PrefillMTPAccepted          bool                       `json:"prefill_mtp_accepted"`
	PrefillGreedySeedMTPNextID  int                        `json:"prefill_greedy_seed_mtp_next_id,omitempty"`
	PrefillGreedySeedAccepted   bool                       `json:"prefill_greedy_seed_accepted"`
	VerifierLogitForMTP         float32                    `json:"verifier_logit_for_mtp,omitempty"`
	VerifierBestMinusMTP        float32                    `json:"verifier_best_minus_mtp,omitempty"`
	MTPLogitForVerifier         float32                    `json:"mtp_logit_for_verifier,omitempty"`
	MTPBestMinusVerifier        float32                    `json:"mtp_best_minus_verifier,omitempty"`
	MTPDraftIDs                 []int                      `json:"mtp_draft_ids,omitempty"`
	MTPVerifierIDs              []int                      `json:"mtp_verifier_ids,omitempty"`
	MTPAcceptedPrefix           int                        `json:"mtp_accepted_prefix,omitempty"`
	MTPCommittedTokens          []int                      `json:"mtp_committed_tokens,omitempty"`
	MTPCommitStatePos           int                        `json:"mtp_commit_state_pos,omitempty"`
	MTPGenerate                 bool                       `json:"mtp_generate,omitempty"`
	MTPGeneratedIDs             []int                      `json:"mtp_generated_ids,omitempty"`
	MTPGeneratedAccepted        int                        `json:"mtp_generated_accepted,omitempty"`
	MTPGeneratedDrafted         int                        `json:"mtp_generated_drafted,omitempty"`
	MTPGeneratedRounds          int                        `json:"mtp_generated_rounds,omitempty"`
	MTPGeneratedBonusTokens     int                        `json:"mtp_generated_bonus_tokens,omitempty"`
	MTPVerifierChunks           int                        `json:"mtp_verifier_chunks,omitempty"`
	MTPVerifierLayerChunks      int                        `json:"mtp_verifier_layer_chunks,omitempty"`
	MTPGeneratedAcceptanceRate  float64                    `json:"mtp_generated_acceptance_rate,omitempty"`
	MTPAdaptiveFallback         bool                       `json:"mtp_adaptive_fallback,omitempty"`
	SequentialDecoded           string                     `json:"sequential_decoded,omitempty"`
	SequentialDurationMS        int64                      `json:"sequential_duration_ms,omitempty"`
	SequentialDecodeTPS         float64                    `json:"sequential_decode_tokens_per_second,omitempty"`
	SequentialLinearStats       qwen.Qwen35LinearStats     `json:"sequential_linear_stats,omitempty"`
	SequentialLMHeadStats       Qwen36LMHeadStats          `json:"sequential_lm_head_stats,omitempty"`
	MTPSpeedupVsSequential      float64                    `json:"mtp_speedup_vs_sequential,omitempty"`
	KVCachedDurationMS          int64                      `json:"kv_cached_duration_ms,omitempty"`
	KVColdDurationMS            int64                      `json:"kv_cold_duration_ms,omitempty"`
	KVSpeedupVsCold             float64                    `json:"kv_speedup_vs_cold,omitempty"`
	KVColdNextID                int                        `json:"kv_cold_next_id,omitempty"`
	KVCachedNextID              int                        `json:"kv_cached_next_id,omitempty"`
	KVColdMatchesCached         bool                       `json:"kv_cold_matches_cached,omitempty"`
	MmapEagerBytes              int64                      `json:"mmap_eager_bytes,omitempty"`
	MmapEagerMS                 int64                      `json:"mmap_eager_ms,omitempty"`
	GPUPrewarm                  qwen.Qwen35GPUPrewarmStats `json:"gpu_prewarm,omitempty"`
	GPUPrewarmMS                int64                      `json:"gpu_prewarm_ms,omitempty"`
	GPUCache                    qwen.Qwen35GPUCacheStats   `json:"gpu_cache,omitempty"`
	GPUTransientBytesPerToken   float64                    `json:"gpu_transient_bytes_per_token,omitempty"`
	GPUTransientUploadsPerToken float64                    `json:"gpu_transient_uploads_per_token,omitempty"`
	GPUWindowEstimates          []QwenGPUWindowEstimate    `json:"gpu_window_estimates,omitempty"`
	GPUVerify                   qwen.Qwen35GPUVerifyStats  `json:"gpu_verify,omitempty"`
	LinearStats                 qwen.Qwen35LinearStats     `json:"linear_stats,omitempty"`
	LMHeadStats                 Qwen36LMHeadStats          `json:"lm_head_stats,omitempty"`
	PrewarmTokensPerSecond      float64                    `json:"prewarm_tokens_per_second,omitempty"`
	DecodeTokensPerSecond       float64                    `json:"decode_tokens_per_second,omitempty"`
	GPULMHead                   bool                       `json:"gpu_lm_head,omitempty"`
	DurationMS                  int64                      `json:"duration_ms"`
	TokensProcessed             int                        `json:"tokens_processed"`
	TokensPerSecond             float64                    `json:"tokens_per_second"`
	BaseTop                     []TopLogit                 `json:"base_top,omitempty"`
	MTPTop                      []TopLogit                 `json:"mtp_top,omitempty"`
	Summary                     Qwen36Summary              `json:"summary,omitempty"`
	KVReuse                     bool                       `json:"kv_reuse,omitempty"`
	KVCacheHit                  bool                       `json:"kv_cache_hit,omitempty"`
	KVLookupAttempts            int                        `json:"kv_lookup_attempts,omitempty"`
	KVLookupHits                int                        `json:"kv_lookup_hits,omitempty"`
	KVLookupMisses              int                        `json:"kv_lookup_misses,omitempty"`
	KVStoreAttempts             int                        `json:"kv_store_attempts,omitempty"`
	KVEvictedStores             int                        `json:"kv_evicted_stores,omitempty"`
	KVReusedTokens              int                        `json:"kv_reused_tokens,omitempty"`
	KVPrefillTokens             int                        `json:"kv_prefill_tokens,omitempty"`
	KVSuffixTokens              int                        `json:"kv_suffix_tokens,omitempty"`
	KVSkippedPrefillTokens      int                        `json:"kv_skipped_prefill_tokens,omitempty"`
	KVReuseEfficiency           float64                    `json:"kv_reuse_efficiency,omitempty"`
	KVPrimePrompt               string                     `json:"kv_prime_prompt,omitempty"`
	KVPrimeTokens               int                        `json:"kv_prime_tokens,omitempty"`
	KVPrimeStoredChunks         int                        `json:"kv_prime_stored_chunks,omitempty"`
	KVStoredChunks              int                        `json:"kv_stored_chunks,omitempty"`
	KVChunkSize                 int                        `json:"kv_chunk_size,omitempty"`
	KVRepeat                    int                        `json:"kv_repeat,omitempty"`
	KVCacheMaxBytes             int64                      `json:"kv_cache_max_bytes,omitempty"`
	KVCacheUsedBytes            int64                      `json:"kv_cache_used_bytes,omitempty"`
	KVCacheEntries              int                        `json:"kv_cache_entries,omitempty"`
	KVGPUCacheMaxBytes          int64                      `json:"kv_gpu_cache_max_bytes,omitempty"`
	KVGPUCacheUsedBytes         int64                      `json:"kv_gpu_cache_used_bytes,omitempty"`
	KVGPUCacheEntries           int                        `json:"kv_gpu_cache_entries,omitempty"`
	KVGPUHeadroomBytes          uint64                     `json:"kv_gpu_headroom_bytes,omitempty"`
	KVGPUCompressed             bool                       `json:"kv_gpu_compressed,omitempty"`
	KVGPUStateEstimateBytes     int64                      `json:"kv_gpu_state_estimate_bytes,omitempty"`
	KVGPUFreeBytes              uint64                     `json:"kv_gpu_free_bytes,omitempty"`
	KVGPUUploadFailures         int64                      `json:"kv_gpu_upload_failures,omitempty"`
	KVGPUBudgetRejections       int64                      `json:"kv_gpu_budget_rejections,omitempty"`
	KVGPUHeadroomRejections     int64                      `json:"kv_gpu_headroom_rejections,omitempty"`
	KVGPUStoreAttempts          int                        `json:"kv_gpu_store_attempts,omitempty"`
	KVGPUStoredChunks           int                        `json:"kv_gpu_stored_chunks,omitempty"`
	KVGPUHit                    bool                       `json:"kv_gpu_hit,omitempty"`
	KVGPULookupAttempts         int                        `json:"kv_gpu_lookup_attempts,omitempty"`
	KVGPUHits                   int                        `json:"kv_gpu_hits,omitempty"`
	KVGPUHitsMissing            int                        `json:"kv_gpu_misses,omitempty"`
	KVGPUHitPromotions          int                        `json:"kv_gpu_hit_promotions,omitempty"`
	KVGPUHitPromotionFailures   int                        `json:"kv_gpu_hit_promotion_failures,omitempty"`
	KVGPURejectedStores         int                        `json:"kv_gpu_rejected_stores,omitempty"`
	KVGPUVerifyAttempts         int                        `json:"kv_gpu_verify_attempts,omitempty"`
	KVGPUVerifyMatches          int                        `json:"kv_gpu_verify_matches,omitempty"`
	KVGPUVerifyFailures         int                        `json:"kv_gpu_verify_failures,omitempty"`
	KVGPUUsedForRestore         bool                       `json:"kv_gpu_used_for_restore,omitempty"`
	KVGPURestoreAttempts        int                        `json:"kv_gpu_restore_attempts,omitempty"`
	KVGPURestoreFailures        int                        `json:"kv_gpu_restore_failures,omitempty"`
	KVGPURestoreMatchesCold     bool                       `json:"kv_gpu_restore_matches_cold,omitempty"`
	LayerStreamedPrefill        bool                       `json:"layer_streamed_prefill,omitempty"`
	Passed                      bool                       `json:"passed"`
}

type SweepReport struct {
	ModelDir         string   `json:"model_dir"`
	Prompts          []string `json:"prompts"`
	Runs             []Report `json:"runs"`
	Accepted         int      `json:"accepted"`
	Total            int      `json:"total"`
	AcceptanceRate   float64  `json:"acceptance_rate"`
	AcceptedPrefixes int      `json:"accepted_prefixes"`
	DurationMS       int64    `json:"duration_ms"`
	TokensProcessed  int      `json:"tokens_processed"`
	TokensPerSecond  float64  `json:"tokens_per_second"`
}

type rawTensor struct {
	raw   []byte
	dtype string
	shape []int
	mlx   *mlx.QuantWeight
}

var qwenPromptStateCache = qwen.NewPromptCache(2 << 30)

type runner struct {
	bundle       *qwen.Qwen35NativeMTPBundle
	state        qwen.Qwen35BaseForwardState
	emb          rawTensor
	normW        []float32
	lm           rawTensor
	lmGPU        *nvidia.Buffer
	mtpHead      *qwen.QwenNativeMTPHead
	promptTokens []int
	promptHidden [][]float32
	mtpPastK     []float32
	mtpPastV     []float32
}

func main() {
	dir := flag.String("model", "", "Qwen3.6 model directory")
	token := flag.Int("token", 0, "single token id to run when -prompt is empty")
	prompt := flag.String("prompt", "", "text prompt to encode and run")
	promptFile := flag.String("prompt-file", "", "read prompt text from file")
	steps := flag.Int("steps", 1, "greedy decode steps after prompt/token")
	mtp := flag.Bool("mtp", false, "also run native MTP head from last base hidden state and generated token")
	mtpSteps := flag.Int("mtp-steps", 1, "native MTP draft steps for diagnostics/generation")
	mtpGenerate := flag.Bool("mtp-generate", false, "use native MTP draft/verify/commit loop for generation after prompt prefill")
	mtpAdaptive := flag.Bool("mtp-adaptive", false, "fall back to plain verifier decode if native-MTP acceptance is too low")
	mtpMinAcceptance := flag.Float64("mtp-min-acceptance", 0.75, "minimum accepted/drafted ratio before -mtp-adaptive falls back after warmup")
	mtpWarmupRounds := flag.Int("mtp-warmup-rounds", 4, "native-MTP rounds to observe before -mtp-adaptive can fall back")
	mtpVerifyChunk := flag.Bool("mtp-verify-chunk", false, "experimental: precompute verifier states for each native-MTP draft chunk")
	mtpVerifyLayerChunk := flag.Bool("mtp-verify-layer-chunk", false, "experimental: compare native-MTP drafts against a layer-streamed verifier chunk")
	compareSequential := flag.Bool("compare-sequential", false, "after MTP generation, run a sequential decode baseline from the same prefill state")
	topK := flag.Int("topk", 0, "include top-K base/MTP logits in reports; 0 disables")
	greedySeed := flag.Bool("greedy-seed", false, "also run the more expensive prefill MTP diagnostic seeded with the base greedy token")
	useGPU := flag.Bool("gpu", false, "use CUDA for Qwen3.6 NVFP4 GEMV when available")
	gpuCacheMB := flag.Int("gpu-cache-mb", 10600, "GPU cache budget for packed Qwen3.6 weights; tuned below full VRAM to leave transient MLX upload scratch")
	gpuWindowReserveMB := flag.Int("gpu-window-reserve-mb", 0, "MiB to subtract from Qwen resident weight cache for experimental suffix layer-window headroom")
	gpuWindowStartLayer := flag.Int("gpu-window-start-layer", -1, "minimum Qwen layer index admitted to the overflow-window cache; -1 admits any overflow")
	gpuWindowSticky := flag.Bool("gpu-window-sticky", true, "keep admitted overflow-window weights resident when the window is full instead of LRU-evicting them")
	gpuPlacement := flag.String("gpu-placement", "prefix", "Qwen GPU placement policy: prefix, mlp-suffix, or mlp-first")
	gpuCacheHeadroomMB := flag.Int("gpu-cache-headroom-mb", 512, "free-VRAM headroom kept when auto-clamping the Qwen3.6 GPU weight cache")
	eagerMmap := flag.Bool("eager-mmap", false, "prefault safetensors mmap before timed generation")
	gpuPrewarm := flag.Bool("gpu-prewarm", true, "pre-upload GPU cache before timed generation")
	gpuTransientDetail := flag.Bool("gpu-transient-detail", false, "include top transient NVFP4 upload tensor names in GPU cache stats")
	gpuTiming := flag.Bool("gpu-timing", false, "collect per-linear GPU upload/kernel timing; adds hot-path time.Now overhead")
	gpuMLP := flag.Bool("gpu-mlp", true, "use GPU-resident Qwen3.6 MLP hot path when weights are cached")
	gpuMLXOverflow := flag.Bool("gpu-mlx-overflow", true, "transient-upload MLX weights that do not fit in the resident Qwen GPU cache")
	kvReuse := flag.Bool("kv-reuse", false, "reuse in-process Qwen prompt state across -kv-repeat runs")
	kvChunkSize := flag.Int("kv-chunk-size", 32, "token chunk size for Qwen prompt-state reuse")
	kvCacheMB := flag.Int("kv-cache-mb", 2048, "in-process Qwen prompt-state cache budget in MiB for -kv-reuse")
	kvGPUCacheMB := flag.Int("kv-gpu-cache-mb", 0, "experimental NVIDIA hot-tier Qwen prompt-state cache budget in MiB; 0 disables")
	kvGPUHeadroomMB := flag.Int("kv-gpu-headroom-mb", 256, "free VRAM MiB to reserve before accepting Qwen GPU prompt-state cache promotion")
	kvGPUCompressed := flag.Bool("kv-gpu-compressed", false, "store Qwen GPU prompt-state cache entries as BF16-packed buffers")
	kvGPUPromoteOnHit := flag.Bool("kv-gpu-promote-on-hit", true, "promote CPU prompt-cache hits into the GPU hot tier when missing")
	kvGPURestoreOnHit := flag.Bool("kv-gpu-restore-on-hit", false, "experimental: restore CPU state from matching GPU hot-tier entry on cache hits")
	kvGPUVerify := flag.Bool("kv-gpu-verify", false, "download promoted Qwen GPU prompt-cache entries and compare metadata with CPU snapshots")
	kvPrimePrompt := flag.String("kv-prime-prompt", "", "prime Qwen prompt-state cache with this prompt before running -prompt")
	kvPrimePromptFile := flag.String("kv-prime-prompt-file", "", "read Qwen KV prime prompt text from file")
	kvCompareCold := flag.Bool("kv-compare-cold", false, "after a cached -kv-reuse run, run a cold prefill/decode baseline for the same prompt")
	kvStoreEvery := flag.Int("kv-store-every", 1, "store every N eligible Qwen prompt chunks; 1 stores every chunk")
	kvStoreFinalOnly := flag.Bool("kv-store-final-only", false, "store only final prompt snapshots for Qwen KV reuse")
	kvMinStoreTokens := flag.Int("kv-min-store-tokens", 1, "minimum prefix length before storing a Qwen prompt snapshot")
	kvRepeat := flag.Int("kv-repeat", 1, "repeat Qwen prompt prefill N times to validate -kv-reuse hits")
	layerStreamedPrefill := flag.Bool("layer-streamed-prefill", false, "process prompt prefill chunks layer-by-layer instead of token-by-token")
	prefillChunkSize := flag.Int("prefill-chunk-size", 16, "prompt chunk size for -layer-streamed-prefill")
	gpuVerify := flag.Int("gpu-verify", 0, "verify first N GPU NVFP4 GEMVs against CPU reference")
	gpuVerifyTol := flag.Float64("gpu-verify-tol", 1e-4, "GPU NVFP4 verification max-diff tolerance")
	gpuLMHead := flag.Bool("gpu-lm-head", true, "run BF16 LM head on GPU when -gpu is enabled; set -gpu-lm-head=false to disable")
	sweep := flag.String("sweep", "", "newline-separated prompt file for MTP acceptance sweep")
	sweepLimit := flag.Int("sweep-limit", 0, "maximum prompts to run from -sweep; 0 means all")
	flag.Parse()
	if *kvCacheMB < 0 {
		*kvCacheMB = 0
	}
	qwenPromptStateCache = qwen.NewPromptCache(int64(*kvCacheMB) * 1024 * 1024)
	var qwenGPUPromptStateCache *qwen.GPUPromptCache
	if *kvGPUCacheMB > 0 {
		if *kvGPUHeadroomMB < 0 {
			*kvGPUHeadroomMB = 0
		}
		qwenGPUPromptStateCache = qwen.NewGPUPromptCacheWithOptions(int64(*kvGPUCacheMB)*1024*1024, uint64(*kvGPUHeadroomMB)*1024*1024, *kvGPUCompressed)
		defer qwenGPUPromptStateCache.Free()
	}
	if *promptFile != "" {
		data, err := os.ReadFile(*promptFile)
		check("prompt-file", err)
		*prompt = string(data)
	}
	if *kvPrimePromptFile != "" {
		data, err := os.ReadFile(*kvPrimePromptFile)
		check("kv-prime-prompt-file", err)
		*kvPrimePrompt = string(data)
	}
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen36run -model <dir> [-token id | -prompt text] [-steps n]")
		os.Exit(2)
	}
	if *steps < 1 {
		fmt.Fprintln(os.Stderr, "steps must be >= 1")
		os.Exit(2)
	}
	if *mtpSteps < 1 {
		fmt.Fprintln(os.Stderr, "mtp-steps must be >= 1")
		os.Exit(2)
	}
	qwen36UseGPULMHead = *gpuLMHead
	qwen.SetQwen35GPUEnabled(*useGPU)
	qwen.SetQwen35GPUPlacement(*gpuPlacement)
	qwen.SetQwen35GPUWindowBudget(int64(*gpuWindowReserveMB) * 1024 * 1024)
	qwen.SetQwen35GPUWindowMinLayer(*gpuWindowStartLayer)
	qwen.SetQwen35GPUWindowSticky(*gpuWindowSticky)
	qwen.SetQwen35GPUCacheHeadroom(int64(*gpuCacheHeadroomMB) * 1024 * 1024)
	qwen.SetQwen35GPUTransientDetail(*gpuTransientDetail)
	qwen.SetQwen35LinearTiming(*gpuTiming)
	qwen.SetQwen35GPUMLPEnabled(*gpuMLP)
	qwen.SetQwen35GPUMXOverflowEnabled(*gpuMLXOverflow)
	qwen.SetQwen35GPUVerify(*gpuVerify, float32(*gpuVerifyTol))
	qwen.ResetQwen35LinearStats()
	qwen36LMHeadStats = Qwen36LMHeadStats{}
	defer qwen.ResetQwen35GPUCache()
	data, err := os.ReadFile(filepath.Join(*dir, "config.json"))
	check("config", err)
	meta, err := loaderconfig.ParseQwenNativeMTPMetadata(data)
	check("parse config", err)
	bundle, err := qwen.LoadQwen35NativeMTPBundleFromDir(*dir)
	check("load bundle", err)
	defer bundle.Close()
	var mmapEagerBytes int64
	var mmapEagerMS int64
	if *eagerMmap {
		eagerStart := time.Now()
		mmapEagerBytes, err = bundle.EagerLoad()
		check("eager mmap", err)
		mmapEagerMS = time.Since(eagerStart).Milliseconds()
	}
	state, err := bundle.NewForwardState()
	check("state", err)
	src, err := qwen.OpenQwenNativeMTPSafetensorsSource(*dir)
	check("open tensors", err)
	defer src.Close()
	r := runner{bundle: bundle, state: state, emb: mustEmbedding(src, meta), normW: bf16All(mustRawCandidate(src, "model.language_model.norm.weight", "language_model.model.norm.weight")), lm: mustLMHead(src, meta)}
	residentCacheMB := *gpuCacheMB - *gpuWindowReserveMB
	if residentCacheMB < 0 {
		residentCacheMB = 0
	}
	qwen.ConfigureQwen35GPUCache(int64(residentCacheMB) * 1024 * 1024)
	if *gpuLMHead && *useGPU && r.lm.mlx == nil {
		r.lmGPU, err = nvidia.UploadBF16LMHead(r.lm.raw, r.lm.shape[0], r.lm.shape[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "upload GPU LM head failed, falling back to CPU LM head: %v\n", err)
			r.lmGPU = nil
		}
		if r.lmGPU != nil {
			defer r.lmGPU.Free()
		}
	}
	if *gpuLMHead && *useGPU && r.lm.mlx != nil {
		if _, err := qwen.CacheQwen35MLXWeight(r.lm.mlx); err != nil {
			fmt.Fprintf(os.Stderr, "cache GPU MLX LM head failed, falling back to CPU LM head: %v\n", err)
		}
	}
	var prewarmStats qwen.Qwen35GPUPrewarmStats
	var prewarmMS int64
	if *useGPU && *gpuPrewarm {
		prewarmStart := time.Now()
		prewarmStats = qwen.PrewarmQwen35GPUCache(bundle.Base)
		prewarmMS = time.Since(prewarmStart).Milliseconds()
	}
	if *mtp {
		r.mtpHead, err = qwen.LoadQwenNativeMTPHeadFromSafetensorsDir(*dir, meta)
		check("load MTP head", err)
	}
	ropeMax := meta.MaxPositionEmbeddings
	if ropeMax <= 0 || ropeMax > 4096 {
		ropeMax = 4096
	}
	ropeFreqs := qwen.NewQwen35RoPEFreqs(meta, ropeMax)
	if *sweep != "" {
		tok, err := tokenizer.Load(filepath.Join(*dir, "tokenizer.json"))
		check("tokenizer", err)
		prompts := applySweepLimit(loadSweepPrompts(*sweep), *sweepLimit)
		if len(prompts) == 0 {
			fmt.Fprintln(os.Stderr, "sweep prompt file is empty")
			os.Exit(2)
		}
		sweepStart := time.Now()
		sweepReport := SweepReport{ModelDir: *dir, Prompts: prompts, Total: len(prompts)}
		for _, p := range prompts {
			run := newRunner(bundle, state, r.emb, r.normW, r.lm, r.lmGPU, r.mtpHead)
			report, err := runPrompt(run, tok, p, *steps, *mtp, *mtpSteps, *topK, *greedySeed, ropeFreqs, meta, *dir)
			check("sweep prompt", err)
			sweepReport.Runs = append(sweepReport.Runs, report)
			if report.MTPAcceptedByGreedy || report.PrefillMTPAccepted {
				sweepReport.Accepted++
			}
			sweepReport.AcceptedPrefixes += report.MTPAcceptedPrefix
		}
		if sweepReport.Total > 0 {
			sweepReport.AcceptanceRate = float64(sweepReport.Accepted) / float64(sweepReport.Total)
		}
		for _, run := range sweepReport.Runs {
			sweepReport.TokensProcessed += run.TokensProcessed
		}
		sweepReport.DurationMS = time.Since(sweepStart).Milliseconds()
		sweepReport.TokensPerSecond = tokensPerSecond(sweepReport.TokensProcessed, sweepReport.DurationMS)
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(sweepReport)
		return
	}
	inputIDs := []int{*token}
	var tok *tokenizer.Tokenizer
	if *prompt != "" {
		tok, err = tokenizer.Load(filepath.Join(*dir, "tokenizer.json"))
		check("tokenizer", err)
		inputIDs = tok.Encode(*prompt)
		if len(inputIDs) == 0 {
			fmt.Fprintln(os.Stderr, "prompt encoded to zero tokens")
			os.Exit(2)
		}
	}
	runStart := time.Now()
	cachedPromptStart := runStart
	var next int
	var logit float32
	var h []float32
	var preNormHidden []float32
	cacheHit := false
	kvLookupAttempts := 0
	kvLookupHits := 0
	kvStoreAttempts := 0
	kvEvictedStores := 0
	kvGPUStoreAttempts := 0
	kvGPUStoredChunks := 0
	kvGPURejectedStores := 0
	kvGPUHit := false
	kvGPULookupAttempts := 0
	kvGPUHits := 0
	kvGPUMisses := 0
	kvGPUHitPromotions := 0
	kvGPUHitPromotionFailures := 0
	kvGPUVerifyAttempts := 0
	kvGPUVerifyMatches := 0
	kvGPUVerifyFailures := 0
	kvGPUUsedForRestore := false
	kvGPURestoreAttempts := 0
	kvGPURestoreFailures := 0
	kvReusedTokens := 0
	kvStoredChunks := 0
	kvPrefillTokens := 0
	if *kvRepeat < 1 {
		*kvRepeat = 1
	}
	if *kvChunkSize <= 0 {
		*kvChunkSize = len(inputIDs)
	}
	modelID := filepath.Base(*dir)
	layout := fmt.Sprintf("%d:%d", meta.NumHiddenLayers, meta.HiddenSize)
	kvPrimeTokens := 0
	kvPrimeStoredChunks := 0
	if *kvReuse && *kvPrimePrompt != "" {
		if tok == nil {
			tok, err = tokenizer.Load(filepath.Join(*dir, "tokenizer.json"))
			check("tokenizer", err)
		}
		primeIDs := tok.Encode(*kvPrimePrompt)
		if len(primeIDs) == 0 {
			fmt.Fprintln(os.Stderr, "kv-prime-prompt encoded to zero tokens")
			os.Exit(2)
		}
		kvPrimeTokens = len(primeIDs)
		primeRunner := newRunner(bundle, state, r.emb, r.normW, r.lm, r.lmGPU, r.mtpHead)
		for idx, id := range primeIDs {
			pnext, plogit, ph, ppre, err := primeRunner.step(id, ropeFreqs)
			check("kv prime prefill", err)
			if shouldStoreQwenPromptPrefix(idx+1, len(primeIDs), *kvChunkSize, *kvStoreEvery, *kvStoreFinalOnly, *kvMinStoreTokens) {
				kvStoreAttempts++
				snap := qwen.PromptSnapshot{State: qwen.CloneQwen35BaseForwardState(primeRunner.state), Next: pnext, Logit: plogit, Hidden: append([]float32(nil), ph...), PreNorm: append([]float32(nil), ppre...), EndPos: idx + 1}
				if qwenStorePromptPrefix(modelID, layout, primeIDs[:idx+1], *kvChunkSize, snap) {
					kvPrimeStoredChunks++
					if qwenGPUPromptStateCache != nil {
						kvGPUStoreAttempts++
						if qwenStorePromptPrefixGPU(qwenGPUPromptStateCache, modelID, layout, primeIDs[:idx+1], *kvChunkSize, snap) {
							kvGPUStoredChunks++
							if *kvGPUVerify {
								kvGPUVerifyAttempts++
								if qwenVerifyPromptPrefixGPU(qwenGPUPromptStateCache, modelID, layout, primeIDs[:idx+1], *kvChunkSize, snap) {
									kvGPUVerifyMatches++
								} else {
									kvGPUVerifyFailures++
								}
							}
						} else {
							kvGPURejectedStores++
						}
					}
				} else {
					kvEvictedStores++
				}
			}
		}
	}
	cachedPromptStart = time.Now()
	var cachedPrefillDurationMS int64
	for rep := 0; rep < *kvRepeat; rep++ {
		startAt := 0
		if *kvReuse {
			kvLookupAttempts++
			if snap, hitKey, ok := qwenFindLongestPromptPrefixWithKey(modelID, layout, inputIDs, *kvChunkSize); ok {
				kvLookupHits++
				if qwenGPUPromptStateCache != nil {
					kvGPULookupAttempts++
					if qwenGPUPromptStateCache.Contains(hitKey) {
						kvGPUHit = true
						kvGPUHits++
						if *kvGPURestoreOnHit {
							kvGPURestoreAttempts++
							if gpuState, ok, err := qwenGPUPromptStateCache.Download(hitKey); err == nil && ok {
								snap.State = gpuState
								kvGPUUsedForRestore = true
							} else {
								kvGPURestoreFailures++
							}
						}
					} else {
						kvGPUMisses++
						if *kvGPUPromoteOnHit {
							kvGPUStoreAttempts++
							if qwenGPUPromptStateCache.Put(hitKey, snap.State) {
								kvGPUStoredChunks++
								kvGPUHitPromotions++
							} else {
								kvGPURejectedStores++
								kvGPUHitPromotionFailures++
							}
						}
					}
				}
				r.state = qwen.CloneQwen35BaseForwardState(snap.State)
				next, logit = snap.Next, snap.Logit
				h = append([]float32(nil), snap.Hidden...)
				preNormHidden = append([]float32(nil), snap.PreNorm...)
				startAt = snap.EndPos
				if startAt > kvReusedTokens {
					kvReusedTokens = startAt
				}
				cacheHit = true
			}
		}
		kvPrefillTokens += len(inputIDs) - startAt
		if *layerStreamedPrefill && startAt < len(inputIDs) {
			next, logit, h, preNormHidden, err = r.prefillLayerStreamed(inputIDs[startAt:], *prefillChunkSize, ropeFreqs)
			check("streamed prefill", err)
			if *kvReuse && shouldStoreQwenPromptPrefix(len(inputIDs), len(inputIDs), *kvChunkSize, *kvStoreEvery, *kvStoreFinalOnly, *kvMinStoreTokens) {
				kvStoreAttempts++
				snap := qwen.PromptSnapshot{State: qwen.CloneQwen35BaseForwardState(r.state), Next: next, Logit: logit, Hidden: append([]float32(nil), h...), PreNorm: append([]float32(nil), preNormHidden...), EndPos: len(inputIDs)}
				if qwenStorePromptPrefix(modelID, layout, inputIDs, *kvChunkSize, snap) {
					kvStoredChunks++
					if qwenGPUPromptStateCache != nil {
						kvGPUStoreAttempts++
						if qwenStorePromptPrefixGPU(qwenGPUPromptStateCache, modelID, layout, inputIDs, *kvChunkSize, snap) {
							kvGPUStoredChunks++
							if *kvGPUVerify {
								kvGPUVerifyAttempts++
								if qwenVerifyPromptPrefixGPU(qwenGPUPromptStateCache, modelID, layout, inputIDs, *kvChunkSize, snap) {
									kvGPUVerifyMatches++
								} else {
									kvGPUVerifyFailures++
								}
							}
						} else {
							kvGPURejectedStores++
						}
					}
				} else {
					kvEvictedStores++
				}
			}
		} else {
			for idx := startAt; idx < len(inputIDs); idx++ {
				next, logit, h, preNormHidden, err = r.step(inputIDs[idx], ropeFreqs)
				check("prefill", err)
				r.promptTokens = append(r.promptTokens, inputIDs[idx])
				r.promptHidden = append(r.promptHidden, append([]float32(nil), preNormHidden...))
				if *kvReuse && shouldStoreQwenPromptPrefix(idx+1, len(inputIDs), *kvChunkSize, *kvStoreEvery, *kvStoreFinalOnly, *kvMinStoreTokens) {
					kvStoreAttempts++
					snap := qwen.PromptSnapshot{State: qwen.CloneQwen35BaseForwardState(r.state), Next: next, Logit: logit, Hidden: append([]float32(nil), h...), PreNorm: append([]float32(nil), preNormHidden...), EndPos: idx + 1}
					if qwenStorePromptPrefix(modelID, layout, inputIDs[:idx+1], *kvChunkSize, snap) {
						kvStoredChunks++
						if qwenGPUPromptStateCache != nil {
							kvGPUStoreAttempts++
							if qwenStorePromptPrefixGPU(qwenGPUPromptStateCache, modelID, layout, inputIDs[:idx+1], *kvChunkSize, snap) {
								kvGPUStoredChunks++
								if *kvGPUVerify {
									kvGPUVerifyAttempts++
									if qwenVerifyPromptPrefixGPU(qwenGPUPromptStateCache, modelID, layout, inputIDs[:idx+1], *kvChunkSize, snap) {
										kvGPUVerifyMatches++
									} else {
										kvGPUVerifyFailures++
									}
								}
							} else {
								kvGPURejectedStores++
							}
						}
					} else {
						kvEvictedStores++
					}
				}
			}
		}
	}
	cachedPrefillDurationMS = time.Since(cachedPromptStart).Milliseconds()
	prefillVerifierNext := next
	prefillHidden := append([]float32(nil), preNormHidden...)
	prefillToken := inputIDs[len(inputIDs)-1]
	prefillPos := r.state.Pos - 1
	baselineState := qwen.CloneQwen35BaseForwardState(r.state)
	baselineNext := next
	baselineH := append([]float32(nil), h...)
	baselinePreNorm := append([]float32(nil), preNormHidden...)
	generated := make([]int, 0, *steps)
	mtpGenStats := mtpGenerateStats{}
	if *mtpGenerate {
		if r.mtpHead == nil {
			fmt.Fprintln(os.Stderr, "-mtp-generate requires -mtp")
			os.Exit(2)
		}
		check("mtp prompt KV", r.buildMTPPromptKV(ropeFreqs, meta))
		generated, mtpGenStats, next, logit, h, preNormHidden, err = r.generateWithNativeMTP(next, preNormHidden, *steps, *mtpSteps, ropeFreqs, meta, *mtpAdaptive, *mtpMinAcceptance, *mtpWarmupRounds, *mtpVerifyChunk, *mtpVerifyLayerChunk)
		check("mtp generate", err)
	} else {
		cur := next
		for i := 0; i < *steps; i++ {
			generated = append(generated, cur)
			if i == *steps-1 {
				break
			}
			next, logit, h, preNormHidden, err = r.step(cur, ropeFreqs)
			check("decode", err)
			cur = next
		}
	}
	var sum float32
	for _, v := range h {
		if v < 0 {
			sum -= v
		} else {
			sum += v
		}
	}
	decoded := ""
	if tok != nil {
		decoded = tok.Decode(generated)
	}
	mtpGeneratedAcceptanceRate := 0.0
	if *mtpGenerate && mtpGenStats.Drafted > 0 {
		mtpGeneratedAcceptanceRate = float64(mtpGenStats.Accepted) / float64(mtpGenStats.Drafted)
	}
	mtpLinearStats := qwen.Qwen35LinearStatsSnapshot()
	mtpLMHeadStats := qwen36LMHeadStatsSnapshot()
	kvTotalPromptVisits := len(inputIDs) * *kvRepeat
	kvSkippedPrefillTokens := kvTotalPromptVisits - kvPrefillTokens
	if kvSkippedPrefillTokens < 0 {
		kvSkippedPrefillTokens = 0
	}
	kvReuseEfficiency := 0.0
	if kvTotalPromptVisits > 0 {
		kvReuseEfficiency = float64(kvSkippedPrefillTokens) / float64(kvTotalPromptVisits)
	}
	cachedDurationMS := time.Since(runStart).Milliseconds()
	cachedPromptDurationMS := cachedPrefillDurationMS
	gpuPromptStats := qwen.GPUPromptCacheStats{}
	if qwenGPUPromptStateCache != nil {
		gpuPromptStats = qwenGPUPromptStateCache.Stats()
	}
	cacheStats := qwenPromptStateCache.Stats()
	rep := Report{ModelDir: *dir, Prompt: *prompt, InputIDs: inputIDs, GeneratedIDs: generated, Decoded: decoded, TokenID: inputIDs[len(inputIDs)-1], NextID: next, Logit: logit, HiddenAbsSum: sum, DurationMS: cachedDurationMS, TokensProcessed: len(inputIDs) + len(generated), KVReuse: *kvReuse, KVCacheHit: cacheHit, KVLookupAttempts: kvLookupAttempts, KVLookupHits: kvLookupHits, KVLookupMisses: kvLookupAttempts - kvLookupHits, KVStoreAttempts: kvStoreAttempts, KVEvictedStores: kvEvictedStores, KVReusedTokens: kvReusedTokens, KVPrefillTokens: kvPrefillTokens, KVSuffixTokens: kvPrefillTokens, KVSkippedPrefillTokens: kvSkippedPrefillTokens, KVReuseEfficiency: kvReuseEfficiency, KVPrimePrompt: *kvPrimePrompt, KVPrimeTokens: kvPrimeTokens, KVPrimeStoredChunks: kvPrimeStoredChunks, KVStoredChunks: kvStoredChunks, KVChunkSize: *kvChunkSize, KVRepeat: *kvRepeat, KVCacheMaxBytes: cacheStats.MaxBytes, KVCacheUsedBytes: cacheStats.UsedBytes, KVCacheEntries: cacheStats.Entries, KVGPUCacheMaxBytes: gpuPromptStats.MaxBytes, KVGPUCacheUsedBytes: gpuPromptStats.UsedBytes, KVGPUCacheEntries: gpuPromptStats.Entries, KVGPUHeadroomBytes: gpuPromptStats.HeadroomBytes, KVGPUCompressed: gpuPromptStats.Compressed, KVGPUStateEstimateBytes: gpuPromptStats.LastEstimateBytes, KVGPUFreeBytes: gpuPromptStats.LastFreeBytes, KVGPUUploadFailures: gpuPromptStats.UploadFailures, KVGPUBudgetRejections: gpuPromptStats.BudgetRejections, KVGPUHeadroomRejections: gpuPromptStats.HeadroomRejections, KVGPUStoreAttempts: kvGPUStoreAttempts, KVGPUStoredChunks: kvGPUStoredChunks, KVGPUHit: kvGPUHit, KVGPULookupAttempts: kvGPULookupAttempts, KVGPUHits: kvGPUHits, KVGPUHitsMissing: kvGPUMisses, KVGPUHitPromotions: kvGPUHitPromotions, KVGPUHitPromotionFailures: kvGPUHitPromotionFailures, KVGPURejectedStores: kvGPURejectedStores, KVGPUVerifyAttempts: kvGPUVerifyAttempts, KVGPUVerifyMatches: kvGPUVerifyMatches, KVGPUVerifyFailures: kvGPUVerifyFailures, KVGPUUsedForRestore: kvGPUUsedForRestore, KVGPURestoreAttempts: kvGPURestoreAttempts, KVGPURestoreFailures: kvGPURestoreFailures, KVCachedDurationMS: cachedPromptDurationMS, LayerStreamedPrefill: *layerStreamedPrefill, MTPGenerate: *mtpGenerate, MTPGeneratedIDs: generated, MTPGeneratedAccepted: mtpGenStats.Accepted, MTPGeneratedDrafted: mtpGenStats.Drafted, MTPGeneratedRounds: mtpGenStats.Rounds, MTPGeneratedBonusTokens: mtpGenStats.BonusTokens, MTPVerifierChunks: mtpGenStats.VerifierChunks, MTPVerifierLayerChunks: mtpGenStats.VerifierLayerChunks, MTPGeneratedAcceptanceRate: mtpGeneratedAcceptanceRate, MTPAdaptiveFallback: mtpGenStats.AdaptiveFallback, Passed: next >= 0 && len(h) == meta.HiddenSize}
	if *kvCompareCold && *kvReuse {
		coldRunner := newRunner(bundle, state, r.emb, r.normW, r.lm, r.lmGPU, r.mtpHead)
		coldStart := time.Now()
		var coldNext int
		var coldLogit float32
		var coldH, coldPre []float32
		for _, id := range inputIDs {
			coldNext, coldLogit, coldH, coldPre, err = coldRunner.step(id, ropeFreqs)
			check("kv cold prefill", err)
		}
		_ = coldLogit
		_ = coldPre
		rep.KVColdDurationMS = time.Since(coldStart).Milliseconds()
		rep.KVColdNextID = coldNext
		rep.KVCachedNextID = prefillVerifierNext
		rep.KVColdMatchesCached = coldNext == prefillVerifierNext && len(coldH) == len(baselineH)
		if rep.KVGPUUsedForRestore {
			rep.KVGPURestoreMatchesCold = rep.KVColdMatchesCached
		}
		if rep.KVCachedDurationMS > 0 && rep.KVColdDurationMS > 0 {
			rep.KVSpeedupVsCold = float64(rep.KVColdDurationMS) / float64(rep.KVCachedDurationMS)
		}
	}
	if *compareSequential && *mtpGenerate {
		seqBaseLinear := mtpLinearStats
		seqBaseLMHead := mtpLMHeadStats
		seqRunner := newRunner(bundle, baselineState, r.emb, r.normW, r.lm, r.lmGPU, r.mtpHead)
		seqStart := time.Now()
		seqIDs, seqNext, seqLogit, seqH, seqPre, err := seqRunner.generateSequential(baselineNext, baselineH, baselinePreNorm, *steps, ropeFreqs)
		check("sequential compare", err)
		_ = seqNext
		_ = seqLogit
		_ = seqH
		_ = seqPre
		rep.SequentialDurationMS = time.Since(seqStart).Milliseconds()
		if rep.SequentialDurationMS > 0 {
			rep.SequentialDecodeTPS = float64(len(seqIDs)) / (float64(rep.SequentialDurationMS) / 1000.0)
		}
		if tok != nil {
			rep.SequentialDecoded = tok.Decode(seqIDs)
		}
		seqTotalLinear := qwen.Qwen35LinearStatsSnapshot()
		seqTotalLMHead := qwen36LMHeadStatsSnapshot()
		rep.SequentialLinearStats = diffQwen35LinearStats(seqTotalLinear, seqBaseLinear)
		rep.SequentialLMHeadStats = diffQwen36LMHeadStats(seqTotalLMHead, seqBaseLMHead)
		rep.Passed = rep.Passed && len(seqIDs) == len(generated) && rep.SequentialDecoded == rep.Decoded
	}
	if *topK > 0 {
		rep.BaseTop = topKMatVec(r.lm, h, *topK)
	}
	rep.TokensPerSecond = tokensPerSecond(rep.TokensProcessed, rep.DurationMS)
	rep.MmapEagerBytes = mmapEagerBytes
	rep.MmapEagerMS = mmapEagerMS
	rep.GPUPrewarm = prewarmStats
	rep.GPUPrewarmMS = prewarmMS
	rep.GPUCache = qwen.Qwen35GPUCacheStatsSnapshot()
	qwenPopulateTransientPerToken(&rep)
	rep.GPUVerify = qwen.Qwen35GPUVerifyStatsSnapshot()
	rep.LinearStats = mtpLinearStats
	rep.LMHeadStats = mtpLMHeadStats
	addThroughputBreakdown(&rep)
	if rep.DecodeTokensPerSecond > 0 && rep.SequentialDecodeTPS > 0 {
		rep.MTPSpeedupVsSequential = rep.DecodeTokensPerSecond / rep.SequentialDecodeTPS
	}
	rep.Summary = Qwen36Summary{KVHit: rep.KVCacheHit, KVReuseEfficiency: rep.KVReuseEfficiency, KVSpeedupVsCold: rep.KVSpeedupVsCold, MTPAcceptanceRate: rep.MTPGeneratedAcceptanceRate, MTPSpeedupVsSeq: rep.MTPSpeedupVsSequential, DecodeTPS: rep.DecodeTokensPerSecond}
	rep.GPULMHead = r.lmGPU != nil
	if *mtp && !*mtpGenerate {
		applyMTPDiagnostics(&rep, &r, h, prefillVerifierNext, prefillHidden, prefillToken, prefillPos, generated, preNormHidden, ropeFreqs, meta, *mtpSteps, *greedySeed)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
	if !rep.Passed {
		os.Exit(1)
	}
}

func (r *runner) buildMTPPromptKV(ropeFreqs []float32, meta loaderconfig.QwenNativeMTPMetadata) error {
	if r == nil || r.mtpHead == nil || len(r.mtpHead.Layers) == 0 {
		return fmt.Errorf("incomplete native MTP head")
	}
	r.mtpPastK, r.mtpPastV = nil, nil
	// Seed the draft self-attention cache with the full prompt. Generation first
	// commits the prompt verifier token as a bonus, so the first native-MTP seed
	// is that verifier token; all prompt-token MTP K/V belongs in its past.
	limit := len(r.promptTokens)
	for i := 0; i < limit; i++ {
		e := bf16Row(r.emb, r.promptTokens[i])
		pre, err := r.mtpHead.PreProject(e, r.promptHidden[i], 1e-6)
		if err != nil {
			return err
		}
		_, k, v, err := r.mtpHead.Layers[0].ForwardWithKV(pre, i, ropeFreqs, r.mtpPastK, r.mtpPastV, 1e-6, meta)
		if err != nil {
			return err
		}
		r.mtpPastK = append(r.mtpPastK, k...)
		r.mtpPastV = append(r.mtpPastV, v...)
	}
	return nil
}

func shouldStoreQwenPromptPrefix(prefixLen, totalLen, chunkSize, storeEvery int, finalOnly bool, minTokens int) bool {
	if prefixLen <= 0 || totalLen <= 0 || prefixLen > totalLen {
		return false
	}
	if minTokens > 0 && prefixLen < minTokens {
		return false
	}
	if finalOnly {
		return prefixLen == totalLen
	}
	if prefixLen == totalLen {
		return true
	}
	if chunkSize <= 0 || prefixLen%chunkSize != 0 {
		return false
	}
	if storeEvery <= 1 {
		return true
	}
	chunkIdx := prefixLen / chunkSize
	return chunkIdx%storeEvery == 0
}

func nativeMTPDraftStepsForRemaining(mtpSteps, remaining int) int {
	if mtpSteps <= 0 || remaining <= 1 {
		return 0
	}
	if mtpSteps > remaining-1 {
		return remaining - 1
	}
	return mtpSteps
}

func shouldFallbackNativeMTP(stats mtpGenerateStats, adaptive bool, minAcceptance float64, warmupRounds int) bool {
	if !adaptive || stats.Rounds < warmupRounds || stats.Drafted <= 0 {
		return false
	}
	return float64(stats.Accepted)/float64(stats.Drafted) < minAcceptance
}

type qwenVerifierChunk struct {
	IDs       []int
	Next      int
	Logit     float32
	Hidden    []float32
	PreNorm   []float32
	State     qwen.Qwen35BaseForwardState
	StepState []qwen.Qwen35BaseForwardState
	StepH     [][]float32
	StepPre   [][]float32
}

func (r *runner) verifyDraftChunkLayerStreamed(drafts []int, startNext int, ropeFreqs []float32) (qwenVerifierChunk, error) {
	if len(drafts) == 0 {
		return qwenVerifierChunk{Next: startNext, State: qwen.CloneQwen35BaseForwardState(r.state)}, nil
	}
	inputs := make([][]float32, len(drafts))
	for i, draft := range drafts {
		inputs[i] = bf16Row(r.emb, draft)
	}
	outs, stepStates, nextState, err := r.bundle.Base.ForwardChunkLayerStreamedDetailed(inputs, qwen.CloneQwen35BaseForwardState(r.state), ropeFreqs, 1e-6, r.bundle.Meta)
	if err != nil {
		return qwenVerifierChunk{}, err
	}
	out := qwenVerifierChunk{IDs: make([]int, 0, len(drafts)), StepState: stepStates, StepH: make([][]float32, 0, len(drafts)), StepPre: make([][]float32, 0, len(drafts)), State: nextState}
	next := startNext
	for _, hidden := range outs {
		out.IDs = append(out.IDs, next)
		pre := append([]float32(nil), hidden...)
		h := append([]float32(nil), pre...)
		rmsNorm(h, r.normW, 1e-6)
		next, out.Logit = argmaxLMHead(r.lm, r.lmGPU, h)
		out.StepH = append(out.StepH, h)
		out.StepPre = append(out.StepPre, pre)
	}
	out.Next = next
	if len(out.StepH) > 0 {
		out.Hidden = out.StepH[len(out.StepH)-1]
		out.PreNorm = out.StepPre[len(out.StepPre)-1]
	}
	return out, nil
}

func (r *runner) verifyDraftChunk(drafts []int, startNext int, ropeFreqs []float32) (qwenVerifierChunk, error) {
	if len(drafts) == 0 {
		return qwenVerifierChunk{Next: startNext, State: qwen.CloneQwen35BaseForwardState(r.state)}, nil
	}
	verifier := runner{bundle: r.bundle, state: qwen.CloneQwen35BaseForwardState(r.state), emb: r.emb, normW: r.normW, lm: r.lm, lmGPU: r.lmGPU, mtpHead: r.mtpHead}
	out := qwenVerifierChunk{IDs: make([]int, 0, len(drafts)), StepState: make([]qwen.Qwen35BaseForwardState, 0, len(drafts)), StepH: make([][]float32, 0, len(drafts)), StepPre: make([][]float32, 0, len(drafts))}
	next := startNext
	var logit float32
	var h, pre []float32
	var err error
	for _, draft := range drafts {
		out.IDs = append(out.IDs, next)
		next, logit, h, pre, err = verifier.step(draft, ropeFreqs)
		if err != nil {
			return qwenVerifierChunk{}, err
		}
		out.StepState = append(out.StepState, qwen.CloneQwen35BaseForwardState(verifier.state))
		out.StepH = append(out.StepH, append([]float32(nil), h...))
		out.StepPre = append(out.StepPre, append([]float32(nil), pre...))
	}
	out.Next, out.Logit, out.Hidden, out.PreNorm, out.State = next, logit, h, pre, qwen.CloneQwen35BaseForwardState(verifier.state)
	return out, nil
}

func (r *runner) generateWithNativeMTP(verifierNext int, hidden []float32, maxTokens, mtpSteps int, ropeFreqs []float32, meta loaderconfig.QwenNativeMTPMetadata, adaptive bool, minAcceptance float64, warmupRounds int, verifyChunk, verifyLayerChunk bool) ([]int, mtpGenerateStats, int, float32, []float32, []float32, error) {
	if r == nil || r.mtpHead == nil {
		return nil, mtpGenerateStats{}, verifierNext, 0, nil, nil, fmt.Errorf("native MTP head is not loaded")
	}
	if maxTokens <= 0 {
		return nil, mtpGenerateStats{}, verifierNext, 0, nil, hidden, nil
	}
	out := make([]int, 0, maxTokens)
	stats := mtpGenerateStats{}
	curVerifier := verifierNext
	curHidden := append([]float32(nil), hidden...)
	lastToken := -1
	var lastH []float32
	var logit float32
	var err error

	// The verifier token computed from the prompt is a LiteRT-style bonus token:
	// emit and commit it first, then use that committed token plus its pre-norm
	// hidden row as the MTP seed for subsequent drafts.
	out = append(out, curVerifier)
	stats.BonusTokens++
	lastToken = curVerifier
	curVerifier, logit, lastH, curHidden, err = r.step(lastToken, ropeFreqs)
	if err != nil || len(out) >= maxTokens {
		return out, stats, curVerifier, logit, lastH, curHidden, err
	}

	for len(out) < maxTokens {
		remaining := maxTokens - len(out)
		if remaining <= 1 {
			out = append(out, curVerifier)
			stats.BonusTokens++
			lastToken = curVerifier
			curVerifier, logit, lastH, curHidden, err = r.step(lastToken, ropeFreqs)
			if err != nil {
				return out, stats, curVerifier, logit, lastH, curHidden, err
			}
			break
		}
		draftSteps := nativeMTPDraftStepsForRemaining(mtpSteps, remaining)
		drafts, stepK, stepV, err := draftMTPIDsDetailedWithPast(r.mtpHead, r.emb, r.lm, r.lmGPU, lastToken, curHidden, r.state.Pos-1, ropeFreqs, meta, draftSteps, r.mtpPastK, r.mtpPastV)
		if err != nil {
			return out, stats, curVerifier, logit, lastH, curHidden, err
		}
		stats.Rounds++
		stats.Drafted += len(drafts)
		committedMTPSteps := 0
		if verifyChunk || verifyLayerChunk {
			var verify qwenVerifierChunk
			if verifyLayerChunk {
				verify, err = r.verifyDraftChunkLayerStreamed(drafts, curVerifier, ropeFreqs)
				stats.VerifierLayerChunks++
			} else {
				verify, err = r.verifyDraftChunk(drafts, curVerifier, ropeFreqs)
				stats.VerifierChunks++
			}
			if err != nil {
				return out, stats, curVerifier, logit, lastH, curHidden, err
			}
			acceptedThisRound := 0
			for i, draftID := range drafts {
				if len(out)+acceptedThisRound >= maxTokens || i >= len(verify.IDs) || draftID != verify.IDs[i] {
					break
				}
				acceptedThisRound++
			}
			if verifyLayerChunk && acceptedThisRound == len(drafts) && len(out)+acceptedThisRound <= maxTokens {
				for _, draftID := range drafts {
					out = append(out, draftID)
					stats.Accepted++
					committedMTPSteps++
					lastToken = draftID
				}
				r.state = qwen.CloneQwen35BaseForwardState(verify.State)
				lastH = append([]float32(nil), verify.Hidden...)
				curHidden = append([]float32(nil), verify.PreNorm...)
				curVerifier = verify.Next
				logit = verify.Logit
			} else {
				if verifyLayerChunk && acceptedThisRound > 0 && acceptedThisRound <= len(verify.StepState) {
					for i := 0; i < acceptedThisRound; i++ {
						draftID := drafts[i]
						out = append(out, draftID)
						stats.Accepted++
						committedMTPSteps++
						lastToken = draftID
					}
					idx := acceptedThisRound - 1
					r.state = qwen.CloneQwen35BaseForwardState(verify.StepState[idx])
					lastH = append([]float32(nil), verify.StepH[idx]...)
					curHidden = append([]float32(nil), verify.StepPre[idx]...)
					if acceptedThisRound < len(verify.IDs) {
						curVerifier = verify.IDs[acceptedThisRound]
					} else {
						curVerifier = verify.Next
						logit = verify.Logit
					}
				} else {
					for i := 0; i < acceptedThisRound; i++ {
						draftID := drafts[i]
						out = append(out, draftID)
						stats.Accepted++
						committedMTPSteps++
						lastToken = draftID
						if i < len(verify.StepState) {
							r.state = qwen.CloneQwen35BaseForwardState(verify.StepState[i])
							lastH = append([]float32(nil), verify.StepH[i]...)
							curHidden = append([]float32(nil), verify.StepPre[i]...)
						} else {
							curVerifier, logit, lastH, curHidden, err = r.step(draftID, ropeFreqs)
							if err != nil {
								return out, stats, curVerifier, logit, lastH, curHidden, err
							}
						}
					}
				}
				if committedMTPSteps > 0 {
					if committedMTPSteps < len(verify.IDs) {
						curVerifier = verify.IDs[committedMTPSteps]
					} else {
						curVerifier = verify.Next
						logit = verify.Logit
					}
				}
			}
		} else {
			for _, draftID := range drafts {
				if len(out) >= maxTokens || draftID != curVerifier {
					break
				}
				out = append(out, draftID)
				stats.Accepted++
				committedMTPSteps++
				lastToken = draftID
				curVerifier, logit, lastH, curHidden, err = r.step(draftID, ropeFreqs)
				if err != nil {
					return out, stats, curVerifier, logit, lastH, curHidden, err
				}
			}
		}
		if committedMTPSteps == 0 && len(stepK) > 0 {
			// Even if the first draft was rejected, the MTP pass for lastToken is
			// part of the committed context for the next draft round.
			committedMTPSteps = 1
		}
		for i := 0; i < committedMTPSteps && i < len(stepK); i++ {
			r.mtpPastK = append(r.mtpPastK, stepK[i]...)
			r.mtpPastV = append(r.mtpPastV, stepV[i]...)
		}
		if len(out) >= maxTokens {
			break
		}
		if shouldFallbackNativeMTP(stats, adaptive, minAcceptance, warmupRounds) {
			stats.AdaptiveFallback = true
			for len(out) < maxTokens {
				out = append(out, curVerifier)
				lastToken = curVerifier
				curVerifier, logit, lastH, curHidden, err = r.step(lastToken, ropeFreqs)
				if err != nil {
					return out, stats, curVerifier, logit, lastH, curHidden, err
				}
			}
			break
		}
		// Mismatch/all-accepted completion: emit verifier bonus token and commit it.
		bonus := curVerifier
		out = append(out, bonus)
		stats.BonusTokens++
		lastToken = bonus
		curVerifier, logit, lastH, curHidden, err = r.step(bonus, ropeFreqs)
		if err != nil {
			return out, stats, curVerifier, logit, lastH, curHidden, err
		}
	}
	return out, stats, curVerifier, logit, lastH, curHidden, nil
}

func applyMTPDiagnostics(rep *Report, r *runner, h []float32, prefillVerifierNext int, prefillHidden []float32, prefillToken, prefillPos int, generated []int, preNormHidden []float32, ropeFreqs []float32, meta loaderconfig.QwenNativeMTPMetadata, mtpSteps int, greedySeed bool) {
	mtpHead := r.mtpHead
	if mtpHead == nil {
		fmt.Fprintln(os.Stderr, "MTP diagnostics requested but MTP head is not loaded")
		os.Exit(2)
	}
	if mtpHead.Norm == nil {
		fmt.Fprintln(os.Stderr, "MTP logits: missing mtp.norm.weight")
		os.Exit(2)
	}
	prefillMTPEmbedding := bf16Row(r.emb, prefillToken)
	prefillMTPOut, err := mtpHead.ForwardOne(prefillMTPEmbedding, prefillHidden, prefillPos, ropeFreqs, 1e-6, meta)
	check("prefill MTP forward", err)
	prefillMTPLogitHidden := append([]float32(nil), prefillMTPOut...)
	rmsNorm(prefillMTPLogitHidden, mtpHead.Norm.Data(), 1e-6)
	rep.PrefillMTPNextID, rep.PrefillMTPLogit = argmaxLMHead(r.lm, r.lmGPU, prefillMTPLogitHidden)
	rep.PrefillMTPAccepted = rep.PrefillMTPNextID == prefillVerifierNext
	if greedySeed {
		prefillGreedySeedEmbedding := bf16Row(r.emb, prefillVerifierNext)
		prefillGreedySeedOut, err := mtpHead.ForwardOne(prefillGreedySeedEmbedding, prefillHidden, prefillPos, ropeFreqs, 1e-6, meta)
		check("prefill greedy-seed MTP forward", err)
		prefillGreedySeedLogitHidden := append([]float32(nil), prefillGreedySeedOut...)
		rmsNorm(prefillGreedySeedLogitHidden, mtpHead.Norm.Data(), 1e-6)
		rep.PrefillGreedySeedMTPNextID, _ = argmaxLMHead(r.lm, r.lmGPU, prefillGreedySeedLogitHidden)
		rep.PrefillGreedySeedAccepted = rep.PrefillGreedySeedMTPNextID == prefillVerifierNext
	}
	mtpEmbedding := bf16Row(r.emb, generated[len(generated)-1])
	mtpOut, err := mtpHead.ForwardOne(mtpEmbedding, preNormHidden, r.state.Pos-1, ropeFreqs, 1e-6, meta)
	check("MTP forward", err)
	rep.MTPOutputLen = len(mtpOut)
	for _, v := range mtpOut {
		if v < 0 {
			rep.MTPAbsSum -= v
		} else {
			rep.MTPAbsSum += v
		}
	}
	mtpLogitHidden := append([]float32(nil), mtpOut...)
	rmsNorm(mtpLogitHidden, mtpHead.Norm.Data(), 1e-6)
	rep.MTPNextID, rep.MTPLogit = argmaxLMHead(r.lm, r.lmGPU, mtpLogitHidden)
	if len(rep.BaseTop) > 0 {
		rep.MTPTop = topKMatVec(r.lm, mtpLogitHidden, len(rep.BaseTop))
	}
	rep.MTPVerifierNextID = rep.NextID
	rep.MTPAcceptedByGreedy = rep.MTPVerifierNextID == rep.MTPNextID
	rep.VerifierLogitForMTP = matVecRow(r.lm, h, rep.MTPNextID)
	rep.VerifierBestMinusMTP = rep.Logit - rep.VerifierLogitForMTP
	rep.MTPLogitForVerifier = matVecRow(r.lm, mtpLogitHidden, rep.MTPVerifierNextID)
	rep.MTPBestMinusVerifier = rep.MTPLogit - rep.MTPLogitForVerifier
	rep.MTPDraftIDs, err = draftMTPIDs(mtpHead, r.emb, r.lm, generated[len(generated)-1], preNormHidden, r.state.Pos-1, ropeFreqs, meta, mtpSteps)
	check("MTP draft steps", err)
	verifier := runner{bundle: r.bundle, state: qwen.CloneQwen35BaseForwardState(r.state), emb: r.emb, normW: r.normW, lm: r.lm, lmGPU: r.lmGPU, mtpHead: r.mtpHead}
	verifierNext := rep.NextID
	for _, draftID := range rep.MTPDraftIDs {
		rep.MTPVerifierIDs = append(rep.MTPVerifierIDs, verifierNext)
		if draftID != verifierNext {
			break
		}
		rep.MTPAcceptedPrefix++
		rep.MTPCommittedTokens = append(rep.MTPCommittedTokens, draftID)
		verifierNext, _, _, _, err = verifier.step(draftID, ropeFreqs)
		check("MTP verifier accepted step", err)
	}
	// Commit LiteRT-style bonus token as well: on mismatch this is the first
	// verifier token, and when all drafts match it is the verifier token after
	// the accepted prefix.
	rep.MTPCommittedTokens = append(rep.MTPCommittedTokens, verifierNext)
	_, _, _, _, err = verifier.step(verifierNext, ropeFreqs)
	check("MTP verifier bonus commit", err)
	r.state = qwen.CloneQwen35BaseForwardState(verifier.state)
	rep.MTPCommitStatePos = r.state.Pos
	rep.Passed = rep.Passed && rep.MTPOutputLen == meta.HiddenSize && rep.MTPNextID >= 0 && rep.MTPCommitStatePos > 0
}

func draftMTPIDs(head *qwen.QwenNativeMTPHead, emb, lm rawTensor, tokenID int, hidden []float32, pos int, ropeFreqs []float32, meta loaderconfig.QwenNativeMTPMetadata, steps int) ([]int, error) {
	ids, _, _, err := draftMTPIDsDetailedWithPast(head, emb, lm, nil, tokenID, hidden, pos, ropeFreqs, meta, steps, nil, nil)
	return ids, err
}

func draftMTPIDsWithPast(head *qwen.QwenNativeMTPHead, emb, lm rawTensor, tokenID int, hidden []float32, pos int, ropeFreqs []float32, meta loaderconfig.QwenNativeMTPMetadata, steps int, initialK, initialV []float32) ([]int, error) {
	ids, _, _, err := draftMTPIDsDetailedWithPast(head, emb, lm, nil, tokenID, hidden, pos, ropeFreqs, meta, steps, initialK, initialV)
	return ids, err
}

func draftMTPIDsDetailedWithPast(head *qwen.QwenNativeMTPHead, emb, lm rawTensor, lmGPU *nvidia.Buffer, tokenID int, hidden []float32, pos int, ropeFreqs []float32, meta loaderconfig.QwenNativeMTPMetadata, steps int, initialK, initialV []float32) ([]int, [][]float32, [][]float32, error) {
	if head == nil || len(head.Layers) == 0 || head.Norm == nil {
		return nil, nil, nil, fmt.Errorf("incomplete Qwen MTP head")
	}
	ids := make([]int, 0, steps)
	stepK := make([][]float32, 0, steps)
	stepV := make([][]float32, 0, steps)
	curToken := tokenID
	curHidden := append([]float32(nil), hidden...)
	pastK := append([]float32(nil), initialK...)
	pastV := append([]float32(nil), initialV...)
	for i := 0; i < steps; i++ {
		e := bf16Row(emb, curToken)
		pre, err := head.PreProject(e, curHidden, 1e-6)
		if err != nil {
			return nil, nil, nil, err
		}
		out, k, v, err := head.Layers[0].ForwardWithKV(pre, pos+i, ropeFreqs, pastK, pastV, 1e-6, meta)
		if err != nil {
			return nil, nil, nil, err
		}
		pastK = append(pastK, k...)
		pastV = append(pastV, v...)
		logitHidden := append([]float32(nil), out...)
		rmsNorm(logitHidden, head.Norm.Data(), 1e-6)
		next, _ := argmaxLMHead(lm, lmGPU, logitHidden)
		ids = append(ids, next)
		stepK = append(stepK, append([]float32(nil), k...))
		stepV = append(stepV, append([]float32(nil), v...))
		curToken = next
		curHidden = out
	}
	return ids, stepK, stepV, nil
}

func qwenStorePromptPrefix(modelID, layout string, tokens []int, chunkSize int, snap qwen.PromptSnapshot) bool {
	return qwenPromptStateCache.Store(modelID, layout, "mlx4", tokens, chunkSize, snap)
}

func qwenStorePromptPrefixGPU(cache *qwen.GPUPromptCache, modelID, layout string, tokens []int, chunkSize int, snap qwen.PromptSnapshot) bool {
	if cache == nil {
		return false
	}
	key := qwen.PromptPrefixKey(modelID, layout, "mlx4", tokens, chunkSize)
	return cache.Put(key, snap.State)
}

func qwenCloseEnough(a, b, tol float32) bool {
	if a > b {
		return a-b <= tol
	}
	return b-a <= tol
}

func qwenVerifyFloatSamples(got, want []float32, tol float32) bool {
	if len(got) != len(want) {
		return false
	}
	if len(got) == 0 {
		return true
	}
	idxs := []int{0, len(got) / 2, len(got) - 1}
	for _, idx := range idxs {
		if !qwenCloseEnough(got[idx], want[idx], tol) {
			return false
		}
	}
	return true
}

func qwenVerifyPromptPrefixGPU(cache *qwen.GPUPromptCache, modelID, layout string, tokens []int, chunkSize int, snap qwen.PromptSnapshot) bool {
	if cache == nil {
		return false
	}
	key := qwen.PromptPrefixKey(modelID, layout, "mlx4", tokens, chunkSize)
	got, ok, err := cache.Download(key)
	if err != nil || !ok {
		return false
	}
	if got.Pos != snap.State.Pos || len(got.FullK) != len(snap.State.FullK) || len(got.FullV) != len(snap.State.FullV) || len(got.Linear) != len(snap.State.Linear) {
		return false
	}
	tol := float32(0)
	if cache.Stats().Compressed {
		tol = 1.0 / 128.0
	}
	for i := range snap.State.FullK {
		if !qwenVerifyFloatSamples(got.FullK[i], snap.State.FullK[i], tol) {
			return false
		}
	}
	for i := range snap.State.FullV {
		if !qwenVerifyFloatSamples(got.FullV[i], snap.State.FullV[i], tol) {
			return false
		}
	}
	for i := range snap.State.Linear {
		if got.Linear[i].Pos != snap.State.Linear[i].Pos || !qwenVerifyFloatSamples(got.Linear[i].Conv, snap.State.Linear[i].Conv, tol) || !qwenVerifyFloatSamples(got.Linear[i].SSM, snap.State.Linear[i].SSM, tol) {
			return false
		}
	}
	return true
}

func qwenFindLongestPromptPrefix(modelID, layout string, tokens []int, chunkSize int) (qwen.PromptSnapshot, bool) {
	return qwenPromptStateCache.FindLongest(modelID, layout, "mlx4", tokens, chunkSize)
}

func qwenFindLongestPromptPrefixWithKey(modelID, layout string, tokens []int, chunkSize int) (qwen.PromptSnapshot, kv.ChunkKey, bool) {
	return qwenPromptStateCache.FindLongestWithKey(modelID, layout, "mlx4", tokens, chunkSize)
}

func newRunner(bundle *qwen.Qwen35NativeMTPBundle, state qwen.Qwen35BaseForwardState, emb rawTensor, normW []float32, lm rawTensor, lmGPU *nvidia.Buffer, mtpHead *qwen.QwenNativeMTPHead) runner {
	return runner{bundle: bundle, state: qwen.CloneQwen35BaseForwardState(state), emb: emb, normW: normW, lm: lm, lmGPU: lmGPU, mtpHead: mtpHead}
}

func qwen36LMHeadStatsSnapshot() Qwen36LMHeadStats {
	out := qwen36LMHeadStats
	if out.Calls > 0 {
		out.AvgMS = float64(out.GPUMillis+out.CPUMillis) / float64(out.Calls)
	}
	return out
}

func diffQwen35LinearStats(a, b qwen.Qwen35LinearStats) qwen.Qwen35LinearStats {
	return qwen.Qwen35LinearStats{Calls: a.Calls - b.Calls, GPUCalls: a.GPUCalls - b.GPUCalls, CPUCalls: a.CPUCalls - b.CPUCalls, GPUFailures: a.GPUFailures - b.GPUFailures, GPUMillis: a.GPUMillis - b.GPUMillis, GPUUploadMS: a.GPUUploadMS - b.GPUUploadMS, GPUKernelMS: a.GPUKernelMS - b.GPUKernelMS, CPUMillis: a.CPUMillis - b.CPUMillis, VerifyMillis: a.VerifyMillis - b.VerifyMillis}
}

func diffQwen36LMHeadStats(a, b Qwen36LMHeadStats) Qwen36LMHeadStats {
	out := Qwen36LMHeadStats{Calls: a.Calls - b.Calls, GPUMillis: a.GPUMillis - b.GPUMillis, CPUMillis: a.CPUMillis - b.CPUMillis}
	if a.DownloadRows != 0 || b.DownloadRows != 0 {
		out.DownloadRows = a.DownloadRows
	}
	if out.Calls > 0 {
		out.AvgMS = float64(out.GPUMillis+out.CPUMillis) / float64(out.Calls)
	}
	return out
}

func addThroughputBreakdown(rep *Report) {
	if rep == nil {
		return
	}
	if rep.TokensProcessed > len(rep.InputIDs) && rep.DurationMS > 0 {
		rep.DecodeTokensPerSecond = float64(rep.TokensProcessed-len(rep.InputIDs)) / (float64(rep.DurationMS) / 1000)
	}
	if rep.DurationMS+rep.GPUPrewarmMS > 0 {
		rep.PrewarmTokensPerSecond = float64(rep.TokensProcessed) / (float64(rep.DurationMS+rep.GPUPrewarmMS) / 1000)
	}
}

func tokensPerSecond(tokens int, durationMS int64) float64 {
	if tokens <= 0 || durationMS <= 0 {
		return 0
	}
	return float64(tokens) * 1000 / float64(durationMS)
}

func qwenPopulateTransientPerToken(rep *Report) {
	if rep == nil || len(rep.GeneratedIDs) == 0 {
		return
	}
	n := float64(len(rep.GeneratedIDs))
	rep.GPUTransientBytesPerToken = float64(rep.GPUCache.TransientBytes) / n
	rep.GPUTransientUploadsPerToken = float64(rep.GPUCache.Transient) / n
	if rep.GPUCache.TransientUniqueBytes <= 0 || rep.GPUCache.TransientUniqueWeights <= 0 {
		return
	}
	for _, window := range []int{2, 4, 8} {
		bytesPerToken := float64(rep.GPUCache.TransientUniqueBytes) / float64(window)
		uploadsPerToken := float64(rep.GPUCache.TransientUniqueWeights) / float64(window)
		est := QwenGPUWindowEstimate{Tokens: window, TransientBytesPerToken: bytesPerToken, TransientUploadsPerToken: uploadsPerToken}
		if rep.GPUTransientBytesPerToken > 0 {
			est.ByteReduction = rep.GPUTransientBytesPerToken / bytesPerToken
		}
		if rep.GPUTransientUploadsPerToken > 0 {
			est.UploadReduction = rep.GPUTransientUploadsPerToken / uploadsPerToken
		}
		rep.GPUWindowEstimates = append(rep.GPUWindowEstimates, est)
	}
}

func applySweepLimit(prompts []string, limit int) []string {
	if limit > 0 && limit < len(prompts) {
		return prompts[:limit]
	}
	return prompts
}

func loadSweepPrompts(path string) []string {
	data, err := os.ReadFile(path)
	check("sweep", err)
	lines := strings.Split(string(data), "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			out = append(out, line)
		}
	}
	return out
}

func runPrompt(r runner, tok *tokenizer.Tokenizer, prompt string, steps int, mtp bool, mtpSteps, topK int, greedySeed bool, ropeFreqs []float32, meta loaderconfig.QwenNativeMTPMetadata, dir string) (Report, error) {
	start := time.Now()
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		return Report{}, fmt.Errorf("prompt %q encoded to zero tokens", prompt)
	}
	var next int
	var logit float32
	var h, preNormHidden []float32
	var err error
	for _, id := range ids {
		next, logit, h, preNormHidden, err = r.step(id, ropeFreqs)
		if err != nil {
			return Report{}, err
		}
	}
	prefillVerifierNext := next
	prefillHidden := append([]float32(nil), preNormHidden...)
	prefillToken := ids[len(ids)-1]
	prefillPos := r.state.Pos - 1
	generated := make([]int, 0, steps)
	cur := next
	for i := 0; i < steps; i++ {
		generated = append(generated, cur)
		if i == steps-1 {
			break
		}
		next, logit, h, preNormHidden, err = r.step(cur, ropeFreqs)
		if err != nil {
			return Report{}, err
		}
		cur = next
	}
	var sum float32
	for _, v := range h {
		if v < 0 {
			sum -= v
		} else {
			sum += v
		}
	}
	rep := Report{ModelDir: dir, Prompt: prompt, InputIDs: ids, GeneratedIDs: generated, Decoded: tok.Decode(generated), TokenID: ids[len(ids)-1], NextID: next, Logit: logit, HiddenAbsSum: sum, DurationMS: time.Since(start).Milliseconds(), TokensProcessed: len(ids) + len(generated), Passed: next >= 0 && len(h) == meta.HiddenSize}
	rep.TokensPerSecond = tokensPerSecond(rep.TokensProcessed, rep.DurationMS)
	rep.GPUCache = qwen.Qwen35GPUCacheStatsSnapshot()
	qwenPopulateTransientPerToken(&rep)
	rep.GPUVerify = qwen.Qwen35GPUVerifyStatsSnapshot()
	rep.LinearStats = qwen.Qwen35LinearStatsSnapshot()
	rep.LMHeadStats = qwen36LMHeadStatsSnapshot()
	addThroughputBreakdown(&rep)
	rep.GPULMHead = r.lmGPU != nil
	if topK > 0 {
		rep.BaseTop = topKMatVec(r.lm, h, topK)
	}
	if mtp {
		applyMTPDiagnostics(&rep, &r, h, prefillVerifierNext, prefillHidden, prefillToken, prefillPos, generated, preNormHidden, ropeFreqs, meta, mtpSteps, greedySeed)
	}
	return rep, nil
}

func (r *runner) generateSequential(firstNext int, firstHidden, firstPreNorm []float32, steps int, ropeFreqs []float32) ([]int, int, float32, []float32, []float32, error) {
	if steps <= 0 {
		return nil, firstNext, 0, firstHidden, firstPreNorm, nil
	}
	generated := make([]int, 0, steps)
	cur := firstNext
	next := firstNext
	var logit float32
	h := append([]float32(nil), firstHidden...)
	pre := append([]float32(nil), firstPreNorm...)
	var err error
	for i := 0; i < steps; i++ {
		generated = append(generated, cur)
		if i == steps-1 {
			break
		}
		next, logit, h, pre, err = r.step(cur, ropeFreqs)
		if err != nil {
			return generated, next, logit, h, pre, err
		}
		cur = next
	}
	return generated, next, logit, h, pre, nil
}

func (r *runner) prefillLayerStreamed(tokenIDs []int, chunkSize int, ropeFreqs []float32) (int, float32, []float32, []float32, error) {
	if chunkSize <= 0 {
		chunkSize = len(tokenIDs)
	}
	var last []float32
	for start := 0; start < len(tokenIDs); start += chunkSize {
		end := start + chunkSize
		if end > len(tokenIDs) {
			end = len(tokenIDs)
		}
		inputs := make([][]float32, end-start)
		for i, id := range tokenIDs[start:end] {
			inputs[i] = bf16Row(r.emb, id)
		}
		outs, nextState, err := r.bundle.Base.ForwardChunkLayerStreamed(inputs, r.state, ropeFreqs, 1e-6, r.bundle.Meta)
		if err != nil {
			return 0, 0, nil, nil, err
		}
		r.state = nextState
		for i, out := range outs {
			r.promptTokens = append(r.promptTokens, tokenIDs[start+i])
			r.promptHidden = append(r.promptHidden, append([]float32(nil), out...))
		}
		if len(outs) > 0 {
			last = outs[len(outs)-1]
		}
	}
	if len(last) == 0 {
		return 0, 0, nil, nil, fmt.Errorf("empty streamed prefill")
	}
	preNorm := append([]float32(nil), last...)
	h := append([]float32(nil), preNorm...)
	rmsNorm(h, r.normW, 1e-6)
	id, val := argmaxLMHead(r.lm, r.lmGPU, h)
	return id, val, h, preNorm, nil
}

func (r *runner) step(tokenID int, ropeFreqs []float32) (int, float32, []float32, []float32, error) {
	hidden := bf16Row(r.emb, tokenID)
	outs, nextState, err := r.bundle.ForwardBaseSequence([][]float32{hidden}, r.state, ropeFreqs, 1e-6)
	if err != nil {
		return 0, 0, nil, nil, err
	}
	r.state = nextState
	preNorm := append([]float32(nil), outs[len(outs)-1]...)
	h := append([]float32(nil), preNorm...)
	rmsNorm(h, r.normW, 1e-6)
	id, val := argmaxLMHead(r.lm, r.lmGPU, h)
	return id, val, h, preNorm, nil
}

func check(what string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		os.Exit(2)
	}
}
func mustRaw(src interface {
	GetRaw(string) ([]byte, string, []int, error)
}, name string) rawTensor {
	r, d, s, e := src.GetRaw(name)
	check(name, e)
	return rawTensor{raw: r, dtype: d, shape: s}
}

func mustRawCandidate(src interface {
	GetRaw(string) ([]byte, string, []int, error)
}, names ...string) rawTensor {
	var last error
	for _, name := range names {
		r, d, s, e := src.GetRaw(name)
		if e == nil {
			return rawTensor{raw: r, dtype: d, shape: s}
		}
		last = e
	}
	check(strings.Join(names, "|"), last)
	return rawTensor{}
}

type qwen36MLXLoader struct {
	src interface {
		Get(string, []int) (*tensor.Tensor, error)
		GetRaw(string) ([]byte, string, []int, error)
	}
}

func (l qwen36MLXLoader) GetRaw(name string) ([]byte, string, []int, error) {
	return l.src.GetRaw(name)
}
func (l qwen36MLXLoader) GetFloat32(name string) ([]float32, []int, error) {
	t, err := l.src.Get(name, nil)
	if err != nil {
		return nil, nil, err
	}
	return t.Data(), t.Shape(), nil
}

func mustEmbedding(src interface {
	Get(string, []int) (*tensor.Tensor, error)
	GetRaw(string) ([]byte, string, []int, error)
}, meta loaderconfig.QwenNativeMTPMetadata) rawTensor {
	loader := qwen36MLXLoader{src: src}
	for _, prefix := range []string{"model.language_model.embed_tokens", "language_model.model.embed_tokens"} {
		qw, err := mlx.LoadWeight(loader, prefix, meta.VocabSize, meta.HiddenSize, qwen35QuantGroup(meta), qwen35QuantBits(meta))
		if err == nil {
			return rawTensor{mlx: qw, dtype: "MLX", shape: []int{qw.OutDim, qw.InDim}}
		}
	}
	return mustRawCandidate(src, "model.language_model.embed_tokens.weight", "language_model.model.embed_tokens.weight")
}

func mustLMHead(src interface {
	Get(string, []int) (*tensor.Tensor, error)
	GetRaw(string) ([]byte, string, []int, error)
}, meta loaderconfig.QwenNativeMTPMetadata) rawTensor {
	loader := qwen36MLXLoader{src: src}
	for _, prefix := range []string{"lm_head", "language_model.lm_head"} {
		qw, err := mlx.LoadWeight(loader, prefix, meta.VocabSize, meta.HiddenSize, qwen35QuantGroup(meta), qwen35QuantBits(meta))
		if err == nil {
			return rawTensor{mlx: qw, dtype: "MLX", shape: []int{qw.OutDim, qw.InDim}}
		}
	}
	return mustRawCandidate(src, "lm_head.weight", "language_model.lm_head.weight")
}

func qwen35QuantGroup(meta loaderconfig.QwenNativeMTPMetadata) int {
	if meta.QuantGroup > 0 {
		return meta.QuantGroup
	}
	return 64
}
func qwen35QuantBits(meta loaderconfig.QwenNativeMTPMetadata) int {
	if meta.QuantBits > 0 {
		return meta.QuantBits
	}
	return 4
}
func bf16(bits []byte, i int) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(bits[i*2:])) << 16)
}
func bf16Row(t rawTensor, row int) []float32 {
	if t.mlx != nil {
		out := make([]float32, t.mlx.InDim)
		if !mlx.DequantRowTo(out, t.mlx, row) {
			check("MLX row", fmt.Errorf("row=%d shape=%v", row, t.shape))
		}
		return out
	}
	if t.dtype != "BF16" || len(t.shape) != 2 {
		check("BF16 matrix", fmt.Errorf("dtype=%s shape=%v", t.dtype, t.shape))
	}
	cols := t.shape[1]
	out := make([]float32, cols)
	off := row * cols * 2
	for i := 0; i < cols; i++ {
		out[i] = bf16(t.raw[off:], i)
	}
	return out
}
func bf16All(t rawTensor) []float32 {
	if t.dtype != "BF16" {
		check("BF16 tensor", fmt.Errorf("dtype=%s", t.dtype))
	}
	n := len(t.raw) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = bf16(t.raw, i)
	}
	return out
}
func rmsNorm(x, w []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	scale := float32(1 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := range x {
		x[i] *= scale * w[i]
	}
}
func topKMatVec(t rawTensor, x []float32, k int) []TopLogit {
	if k <= 0 || len(t.shape) != 2 {
		return nil
	}
	rows := t.shape[0]
	if k > rows {
		k = rows
	}
	top := make([]TopLogit, 0, k)
	for row := 0; row < rows; row++ {
		v := matVecRow(t, x, row)
		inserted := false
		for i := range top {
			if v > top[i].Logit {
				top = append(top, TopLogit{})
				copy(top[i+1:], top[i:])
				top[i] = TopLogit{ID: row, Logit: v}
				inserted = true
				break
			}
		}
		if !inserted && len(top) < k {
			top = append(top, TopLogit{ID: row, Logit: v})
		}
		if len(top) > k {
			top = top[:k]
		}
	}
	return top
}

func argmaxLMHead(t rawTensor, lmGPU *nvidia.Buffer, x []float32) (int, float32) {
	if t.mlx != nil {
		if qwen36UseGPULMHead {
			if idx, val, ok := qwen35ArgmaxMLXGPUIndex(t.mlx, x); ok {
				return idx, val
			}
		}
		logits := make([]float32, t.mlx.OutDim)
		if qwen36UseGPULMHead && qwen35ArgmaxMLXGPU(logits, t.mlx, x) {
			qwen36LMHeadStats.Calls++
			qwen36LMHeadStats.DownloadRows = t.mlx.OutDim
		} else {
			mlx.Gemv(logits, x, t.mlx)
			qwen36LMHeadStats.Calls++
		}
		best := -1
		bestv := float32(math.Inf(-1))
		for i, v := range logits {
			if v > bestv {
				bestv = v
				best = i
			}
		}
		return best, bestv
	}
	if qwen36UseGPULMHead && t.dtype == "BF16" && len(t.shape) == 2 && lmGPU != nil {
		if cap(qwen36LMHeadLogitsScratch) < t.shape[0] {
			qwen36LMHeadLogitsScratch = make([]float32, t.shape[0])
		}
		logits := qwen36LMHeadLogitsScratch[:t.shape[0]]
		start := time.Now()
		if err := nvidia.BF16LMHeadWithBuffer(logits, lmGPU, x, t.shape[0], t.shape[1]); err == nil {
			qwen36LMHeadStats.Calls++
			qwen36LMHeadStats.GPUMillis += time.Since(start).Milliseconds()
			qwen36LMHeadStats.DownloadRows = t.shape[0]
			best := -1
			bestv := float32(math.Inf(-1))
			for i, v := range logits {
				if v > bestv {
					bestv = v
					best = i
				}
			}
			if best >= 0 {
				return best, bestv
			}
		}
	}
	start := time.Now()
	best, bestv := argmaxBF16MatVec(t, x)
	qwen36LMHeadStats.Calls++
	qwen36LMHeadStats.CPUMillis += time.Since(start).Milliseconds()
	return best, bestv
}

func argmaxBF16MatVec(t rawTensor, x []float32) (int, float32) {
	rows := t.shape[0]
	best := -1
	bestv := float32(math.Inf(-1))
	for r := 0; r < rows; r++ {
		s := matVecRow(t, x, r)
		if s > bestv {
			bestv = s
			best = r
		}
	}
	return best, bestv
}

func matVecRow(t rawTensor, x []float32, row int) float32 {
	if t.mlx != nil {
		w := make([]float32, t.mlx.InDim)
		if !mlx.DequantRowTo(w, t.mlx, row) {
			return float32(math.Inf(-1))
		}
		var sum float32
		for i := range w {
			sum += w[i] * x[i]
		}
		return sum
	}
	if len(t.shape) != 2 || row < 0 || row >= t.shape[0] {
		return float32(math.Inf(-1))
	}
	cols := t.shape[1]
	off := row * cols * 2
	var s float32
	for c := 0; c < cols; c++ {
		s += bf16(t.raw[off:], c) * x[c]
	}
	return s
}
