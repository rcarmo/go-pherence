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

	"github.com/rcarmo/go-pherence/gpu"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
)

var qwen36UseGPULMHead bool
var qwen36LMHeadLogitsScratch []float32
var qwen36LMHeadStats Qwen36LMHeadStats

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

type Report struct {
	ModelDir                   string                      `json:"model_dir"`
	Prompt                     string                      `json:"prompt,omitempty"`
	InputIDs                   []int                       `json:"input_ids"`
	GeneratedIDs               []int                       `json:"generated_ids,omitempty"`
	Decoded                    string                      `json:"decoded,omitempty"`
	TokenID                    int                         `json:"token_id,omitempty"`
	NextID                     int                         `json:"next_id"`
	Logit                      float32                     `json:"logit"`
	HiddenAbsSum               float32                     `json:"hidden_abs_sum"`
	MTPOutputLen               int                         `json:"mtp_output_len,omitempty"`
	MTPAbsSum                  float32                     `json:"mtp_abs_sum,omitempty"`
	MTPNextID                  int                         `json:"mtp_next_id,omitempty"`
	MTPLogit                   float32                     `json:"mtp_logit,omitempty"`
	MTPVerifierNextID          int                         `json:"mtp_verifier_next_id,omitempty"`
	MTPAcceptedByGreedy        bool                        `json:"mtp_accepted_by_greedy"`
	PrefillMTPNextID           int                         `json:"prefill_mtp_next_id,omitempty"`
	PrefillMTPLogit            float32                     `json:"prefill_mtp_logit,omitempty"`
	PrefillMTPAccepted         bool                        `json:"prefill_mtp_accepted"`
	PrefillGreedySeedMTPNextID int                         `json:"prefill_greedy_seed_mtp_next_id,omitempty"`
	PrefillGreedySeedAccepted  bool                        `json:"prefill_greedy_seed_accepted"`
	VerifierLogitForMTP        float32                     `json:"verifier_logit_for_mtp,omitempty"`
	VerifierBestMinusMTP       float32                     `json:"verifier_best_minus_mtp,omitempty"`
	MTPLogitForVerifier        float32                     `json:"mtp_logit_for_verifier,omitempty"`
	MTPBestMinusVerifier       float32                     `json:"mtp_best_minus_verifier,omitempty"`
	MTPDraftIDs                []int                       `json:"mtp_draft_ids,omitempty"`
	MTPVerifierIDs             []int                       `json:"mtp_verifier_ids,omitempty"`
	MTPAcceptedPrefix          int                         `json:"mtp_accepted_prefix,omitempty"`
	MmapEagerBytes             int64                       `json:"mmap_eager_bytes,omitempty"`
	MmapEagerMS                int64                       `json:"mmap_eager_ms,omitempty"`
	GPUPrewarm                 model.Qwen35GPUPrewarmStats `json:"gpu_prewarm,omitempty"`
	GPUPrewarmMS               int64                       `json:"gpu_prewarm_ms,omitempty"`
	GPUCache                   model.Qwen35GPUCacheStats   `json:"gpu_cache,omitempty"`
	GPUVerify                  model.Qwen35GPUVerifyStats  `json:"gpu_verify,omitempty"`
	LinearStats                model.Qwen35LinearStats     `json:"linear_stats,omitempty"`
	LMHeadStats                Qwen36LMHeadStats           `json:"lm_head_stats,omitempty"`
	PrewarmTokensPerSecond     float64                     `json:"prewarm_tokens_per_second,omitempty"`
	DecodeTokensPerSecond      float64                     `json:"decode_tokens_per_second,omitempty"`
	GPULMHead                  bool                        `json:"gpu_lm_head,omitempty"`
	DurationMS                 int64                       `json:"duration_ms"`
	TokensProcessed            int                         `json:"tokens_processed"`
	TokensPerSecond            float64                     `json:"tokens_per_second"`
	BaseTop                    []TopLogit                  `json:"base_top,omitempty"`
	MTPTop                     []TopLogit                  `json:"mtp_top,omitempty"`
	Passed                     bool                        `json:"passed"`
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
}

type runner struct {
	bundle  *model.Qwen35NativeMTPBundle
	state   model.Qwen35BaseForwardState
	emb     rawTensor
	normW   []float32
	lm      rawTensor
	lmGPU   *gpu.Buffer
	mtpHead *model.QwenNativeMTPHead
}

func main() {
	dir := flag.String("model", "", "Qwen3.6 model directory")
	token := flag.Int("token", 0, "single token id to run when -prompt is empty")
	prompt := flag.String("prompt", "", "text prompt to encode and run")
	steps := flag.Int("steps", 1, "greedy decode steps after prompt/token")
	mtp := flag.Bool("mtp", false, "also run native MTP head from last base hidden state and generated token")
	mtpSteps := flag.Int("mtp-steps", 1, "native MTP draft steps for diagnostics")
	topK := flag.Int("topk", 0, "include top-K base/MTP logits in reports; 0 disables")
	greedySeed := flag.Bool("greedy-seed", false, "also run the more expensive prefill MTP diagnostic seeded with the base greedy token")
	useGPU := flag.Bool("gpu", false, "use CUDA for Qwen3.6 NVFP4 GEMV when available")
	gpuCacheMB := flag.Int("gpu-cache-mb", 12288, "GPU cache budget for packed Qwen3.6 NVFP4 weights; 0 disables eviction; auto-clamped to free VRAM")
	eagerMmap := flag.Bool("eager-mmap", false, "prefault safetensors mmap before timed generation")
	gpuPrewarm := flag.Bool("gpu-prewarm", true, "pre-upload GPU cache before timed generation")
	gpuTransientDetail := flag.Bool("gpu-transient-detail", false, "include top transient NVFP4 upload tensor names in GPU cache stats")
	gpuVerify := flag.Int("gpu-verify", 0, "verify first N GPU NVFP4 GEMVs against CPU reference")
	gpuVerifyTol := flag.Float64("gpu-verify-tol", 1e-4, "GPU NVFP4 verification max-diff tolerance")
	gpuLMHead := flag.Bool("gpu-lm-head", true, "run BF16 LM head on GPU when -gpu is enabled; set -gpu-lm-head=false to disable")
	sweep := flag.String("sweep", "", "newline-separated prompt file for MTP acceptance sweep")
	sweepLimit := flag.Int("sweep-limit", 0, "maximum prompts to run from -sweep; 0 means all")
	flag.Parse()
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
	model.SetQwen35GPUEnabled(*useGPU)
	model.SetQwen35GPUTransientDetail(*gpuTransientDetail)
	model.SetQwen35GPUVerify(*gpuVerify, float32(*gpuVerifyTol))
	model.ResetQwen35LinearStats()
	qwen36LMHeadStats = Qwen36LMHeadStats{}
	defer model.ResetQwen35GPUCache()
	data, err := os.ReadFile(filepath.Join(*dir, "config.json"))
	check("config", err)
	meta, err := loaderconfig.ParseQwenNativeMTPMetadata(data)
	check("parse config", err)
	bundle, err := model.LoadQwen35NativeMTPBundleFromDir(*dir)
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
	src, err := model.OpenQwenNativeMTPSafetensorsSource(*dir)
	check("open tensors", err)
	defer src.Close()
	r := runner{bundle: bundle, state: state, emb: mustRaw(src, "model.language_model.embed_tokens.weight"), normW: bf16All(mustRaw(src, "model.language_model.norm.weight")), lm: mustRaw(src, "lm_head.weight")}
	if *gpuLMHead && *useGPU {
		r.lmGPU, err = gpu.UploadBF16LMHead(r.lm.raw, r.lm.shape[0], r.lm.shape[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "upload GPU LM head failed, falling back to CPU LM head: %v\n", err)
			r.lmGPU = nil
		}
		if r.lmGPU != nil {
			defer r.lmGPU.Free()
		}
	}
	model.ConfigureQwen35GPUCache(int64(*gpuCacheMB) * 1024 * 1024)
	var prewarmStats model.Qwen35GPUPrewarmStats
	var prewarmMS int64
	if *useGPU && *gpuPrewarm {
		prewarmStart := time.Now()
		prewarmStats = model.PrewarmQwen35GPUCache(bundle.Base)
		prewarmMS = time.Since(prewarmStart).Milliseconds()
	}
	if *mtp {
		r.mtpHead, err = model.LoadQwenNativeMTPHeadFromSafetensorsDir(*dir, meta)
		check("load MTP head", err)
	}
	ropeMax := meta.MaxPositionEmbeddings
	if ropeMax <= 0 || ropeMax > 4096 {
		ropeMax = 4096
	}
	ropeFreqs := model.NewQwen35RoPEFreqs(meta, ropeMax)
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
	var next int
	var logit float32
	var h []float32
	var preNormHidden []float32
	for _, id := range inputIDs {
		next, logit, h, preNormHidden, err = r.step(id, ropeFreqs)
		check("prefill", err)
	}
	prefillVerifierNext := next
	prefillHidden := append([]float32(nil), preNormHidden...)
	prefillToken := inputIDs[len(inputIDs)-1]
	prefillPos := r.state.Pos - 1
	generated := make([]int, 0, *steps)
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
	rep := Report{ModelDir: *dir, Prompt: *prompt, InputIDs: inputIDs, GeneratedIDs: generated, Decoded: decoded, TokenID: inputIDs[len(inputIDs)-1], NextID: next, Logit: logit, HiddenAbsSum: sum, DurationMS: time.Since(runStart).Milliseconds(), TokensProcessed: len(inputIDs) + len(generated), Passed: next >= 0 && len(h) == meta.HiddenSize}
	if *topK > 0 {
		rep.BaseTop = topKBF16MatVec(r.lm, h, *topK)
	}
	rep.TokensPerSecond = tokensPerSecond(rep.TokensProcessed, rep.DurationMS)
	rep.MmapEagerBytes = mmapEagerBytes
	rep.MmapEagerMS = mmapEagerMS
	rep.GPUPrewarm = prewarmStats
	rep.GPUPrewarmMS = prewarmMS
	rep.GPUCache = model.Qwen35GPUCacheStatsSnapshot()
	rep.GPUVerify = model.Qwen35GPUVerifyStatsSnapshot()
	rep.LinearStats = model.Qwen35LinearStatsSnapshot()
	rep.LMHeadStats = qwen36LMHeadStatsSnapshot()
	addThroughputBreakdown(&rep)
	rep.GPULMHead = r.lmGPU != nil
	if *mtp {
		applyMTPDiagnostics(&rep, &r, h, prefillVerifierNext, prefillHidden, prefillToken, prefillPos, generated, preNormHidden, ropeFreqs, meta, *mtpSteps, *greedySeed)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
	if !rep.Passed {
		os.Exit(1)
	}
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
	rep.PrefillMTPNextID, rep.PrefillMTPLogit = argmaxBF16MatVec(r.lm, prefillMTPLogitHidden)
	rep.PrefillMTPAccepted = rep.PrefillMTPNextID == prefillVerifierNext
	if greedySeed {
		prefillGreedySeedEmbedding := bf16Row(r.emb, prefillVerifierNext)
		prefillGreedySeedOut, err := mtpHead.ForwardOne(prefillGreedySeedEmbedding, prefillHidden, prefillPos, ropeFreqs, 1e-6, meta)
		check("prefill greedy-seed MTP forward", err)
		prefillGreedySeedLogitHidden := append([]float32(nil), prefillGreedySeedOut...)
		rmsNorm(prefillGreedySeedLogitHidden, mtpHead.Norm.Data(), 1e-6)
		rep.PrefillGreedySeedMTPNextID, _ = argmaxBF16MatVec(r.lm, prefillGreedySeedLogitHidden)
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
	rep.MTPNextID, rep.MTPLogit = argmaxBF16MatVec(r.lm, mtpLogitHidden)
	if len(rep.BaseTop) > 0 {
		rep.MTPTop = topKBF16MatVec(r.lm, mtpLogitHidden, len(rep.BaseTop))
	}
	rep.MTPVerifierNextID = rep.NextID
	rep.MTPAcceptedByGreedy = rep.MTPVerifierNextID == rep.MTPNextID
	rep.VerifierLogitForMTP = bf16MatVecRow(r.lm, h, rep.MTPNextID)
	rep.VerifierBestMinusMTP = rep.Logit - rep.VerifierLogitForMTP
	rep.MTPLogitForVerifier = bf16MatVecRow(r.lm, mtpLogitHidden, rep.MTPVerifierNextID)
	rep.MTPBestMinusVerifier = rep.MTPLogit - rep.MTPLogitForVerifier
	rep.MTPDraftIDs, err = draftMTPIDs(mtpHead, r.emb, r.lm, generated[len(generated)-1], preNormHidden, r.state.Pos-1, ropeFreqs, meta, mtpSteps)
	check("MTP draft steps", err)
	verifier := runner{bundle: r.bundle, state: model.CloneQwen35BaseForwardState(r.state), emb: r.emb, normW: r.normW, lm: r.lm}
	verifierNext := rep.NextID
	for _, draftID := range rep.MTPDraftIDs {
		rep.MTPVerifierIDs = append(rep.MTPVerifierIDs, verifierNext)
		if draftID != verifierNext {
			break
		}
		rep.MTPAcceptedPrefix++
		verifierNext, _, _, _, err = verifier.step(draftID, ropeFreqs)
		check("MTP verifier accepted step", err)
	}
	rep.Passed = rep.Passed && rep.MTPOutputLen == meta.HiddenSize && rep.MTPNextID >= 0
}

func draftMTPIDs(head *model.QwenNativeMTPHead, emb, lm rawTensor, tokenID int, hidden []float32, pos int, ropeFreqs []float32, meta loaderconfig.QwenNativeMTPMetadata, steps int) ([]int, error) {
	if head == nil || len(head.Layers) == 0 || head.Norm == nil {
		return nil, fmt.Errorf("incomplete Qwen MTP head")
	}
	ids := make([]int, 0, steps)
	curToken := tokenID
	curHidden := append([]float32(nil), hidden...)
	var pastK, pastV []float32
	for i := 0; i < steps; i++ {
		e := bf16Row(emb, curToken)
		pre, err := head.PreProject(e, curHidden, 1e-6)
		if err != nil {
			return nil, err
		}
		out, k, v, err := head.Layers[0].ForwardWithKV(pre, pos+i, ropeFreqs, pastK, pastV, 1e-6, meta)
		if err != nil {
			return nil, err
		}
		pastK = append(pastK, k...)
		pastV = append(pastV, v...)
		logitHidden := append([]float32(nil), out...)
		rmsNorm(logitHidden, head.Norm.Data(), 1e-6)
		next, _ := argmaxBF16MatVec(lm, logitHidden)
		ids = append(ids, next)
		curToken = next
		curHidden = out
	}
	return ids, nil
}

func newRunner(bundle *model.Qwen35NativeMTPBundle, state model.Qwen35BaseForwardState, emb rawTensor, normW []float32, lm rawTensor, lmGPU *gpu.Buffer, mtpHead *model.QwenNativeMTPHead) runner {
	return runner{bundle: bundle, state: model.CloneQwen35BaseForwardState(state), emb: emb, normW: normW, lm: lm, lmGPU: lmGPU, mtpHead: mtpHead}
}

func qwen36LMHeadStatsSnapshot() Qwen36LMHeadStats {
	out := qwen36LMHeadStats
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
	rep.GPUCache = model.Qwen35GPUCacheStatsSnapshot()
	rep.GPUVerify = model.Qwen35GPUVerifyStatsSnapshot()
	rep.LinearStats = model.Qwen35LinearStatsSnapshot()
	rep.LMHeadStats = qwen36LMHeadStatsSnapshot()
	addThroughputBreakdown(&rep)
	rep.GPULMHead = r.lmGPU != nil
	if topK > 0 {
		rep.BaseTop = topKBF16MatVec(r.lm, h, topK)
	}
	if mtp {
		applyMTPDiagnostics(&rep, &r, h, prefillVerifierNext, prefillHidden, prefillToken, prefillPos, generated, preNormHidden, ropeFreqs, meta, mtpSteps, greedySeed)
	}
	return rep, nil
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
	return rawTensor{r, d, s}
}
func bf16(bits []byte, i int) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(bits[i*2:])) << 16)
}
func bf16Row(t rawTensor, row int) []float32 {
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
func topKBF16MatVec(t rawTensor, x []float32, k int) []TopLogit {
	if k <= 0 || len(t.shape) != 2 {
		return nil
	}
	rows := t.shape[0]
	if k > rows {
		k = rows
	}
	top := make([]TopLogit, 0, k)
	for row := 0; row < rows; row++ {
		v := bf16MatVecRow(t, x, row)
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

func argmaxLMHead(t rawTensor, lmGPU *gpu.Buffer, x []float32) (int, float32) {
	if qwen36UseGPULMHead && t.dtype == "BF16" && len(t.shape) == 2 && lmGPU != nil {
		if cap(qwen36LMHeadLogitsScratch) < t.shape[0] {
			qwen36LMHeadLogitsScratch = make([]float32, t.shape[0])
		}
		logits := qwen36LMHeadLogitsScratch[:t.shape[0]]
		start := time.Now()
		if err := gpu.BF16LMHeadWithBuffer(logits, lmGPU, x, t.shape[0], t.shape[1]); err == nil {
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
		s := bf16MatVecRow(t, x, r)
		if s > bestv {
			bestv = s
			best = r
		}
	}
	return best, bestv
}

func bf16MatVecRow(t rawTensor, x []float32, row int) float32 {
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
