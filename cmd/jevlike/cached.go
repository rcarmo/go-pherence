package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	backbone "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/jevlike"
)

func featureRequestExample(r directBatchInput) (jevlike.ChoiceExample, error) {
	ex := jevlike.ChoiceExample{Context: "Evidence:\n" + r.Request.Evidence + "\nQuestion:\n" + r.Request.Question, Label: -1}
	for i, c := range r.Request.Candidates {
		ex.Options = append(ex.Options, c.Text)
		if c.ID == r.GoldID {
			ex.Label = i
		}
	}
	return jevlike.ValidateChoiceExample(ex)
}
func orderedFeatureRequests(path string, expected map[string]directBatchInput) ([]directBatchInput, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 65536), 1<<20)
	var rows []directBatchInput
	for scan.Scan() {
		var r directBatchInput
		if e = json.Unmarshal(scan.Bytes(), &r); e != nil {
			return nil, e
		}
		if _, ok := expected[directRowKey(r.Task, r.ID, r.Variant)]; !ok {
			return nil, fmt.Errorf("unexpected request")
		}
		rows = append(rows, r)
	}
	return rows, scan.Err()
}
func featureSelection(dataset, selection, requests string) (directSelection, []directBatchInput, []jevlike.ChoiceExample, error) {
	s, expected, e := readDirectSelection(selection, requests, dataset)
	if e != nil {
		return s, nil, nil, e
	}
	if s.Partition == "test" {
		return s, nil, nil, fmt.Errorf("final test is closed for this pilot")
	}
	rows, e := orderedFeatureRequests(requests, expected)
	if e != nil {
		return s, nil, nil, e
	}
	examples := make([]jevlike.ChoiceExample, len(rows))
	for i, r := range rows {
		examples[i], e = featureRequestExample(r)
		if e != nil {
			return s, nil, nil, e
		}
	}
	return s, rows, examples, nil
}
func frozenFeatureContract(modelID, datasetHash, dtype string) jevlike.FeatureContract {
	return jevlike.FeatureContract{Version: 1, ModelID: modelID, DatasetSHA256: datasetHash, Backend: "cuda-bf16-resident-compensated-f32/qwen-f32-rope", TokenPolicy: "plain/no-bos/no-eos/reject-overlength", Representation: "causal/final-rmsnorm/all-token-rows", Pooling: "option-mean-f32-before-storage", DType: dtype, Width: 2560, ContextTokens: 512, OptionTokens: 128}
}

func runCacheExtract(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cache-extract", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("encoder-model", "", "Pinned Qwen3 model")
	assets := fs.String("verified-assets", "", "Full verified asset manifest")
	dataset := fs.String("dataset", "", "Prepared dataset")
	selection := fs.String("selection", "", "Partition selection")
	requests := fs.String("requests", "", "Selected requests")
	root := fs.String("cache", "", "Immutable feature root (12 GiB maximum)")
	dtype := fs.String("dtype", "f32", "f32 baseline or f16 ablation")
	plan := fs.Bool("plan", false, "Tokenise and account without GPU extraction")
	maxNew := fs.Int("max-new", 0, "Stop after this many requests with new features; zero is unlimited")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if *model == "" || *assets == "" || *dataset == "" || *selection == "" || *requests == "" || *root == "" || *maxNew < 0 {
		return fmt.Errorf("model, assets, dataset, selection, requests and cache required")
	}
	s, rows, examples, e := featureSelection(*dataset, *selection, *requests)
	if e != nil {
		return e
	}
	id, e := verifyScoreAssets(*model, *assets)
	if e != nil {
		return e
	}
	tok, e := tokenizer.LoadWithConfig(*model)
	if e != nil {
		return e
	}
	contract := frozenFeatureContract(id, s.SourceManifestSHA256, *dtype)
	if e = contract.Validate(); e != nil {
		return e
	}
	tokenize := func(text string) ([]int, error) { return tok.Encode(text), nil }
	unique := map[string]int64{}
	tokenTotal := 0
	for _, ex := range examples {
		for i, text := range append([]string{ex.Context}, ex.Options...) {
			limit := contract.OptionTokens
			if i == 0 {
				limit = contract.ContextTokens
			}
			ids := tok.Encode(text)
			if len(ids) == 0 || len(ids) > limit {
				return fmt.Errorf("admission: tokens=%d limit=%d; no truncation", len(ids), limit)
			}
			key := fmt.Sprintf("%t:%s", i == 0, text)
			if _, ok := unique[key]; ok {
				continue
			}
			n := 1
			if i == 0 {
				n = len(ids)
			}
			size := 4
			if *dtype == "f16" {
				size = 2
			}
			unique[key] = int64(n*contract.Width*size + len(text) + len(ids)*12 + 2048)
			tokenTotal += len(ids)
		}
	}
	var estimated int64
	for _, n := range unique {
		estimated += n
	}
	if estimated > 12<<30 {
		return fmt.Errorf("planned features exceed 12 GiB")
	}
	if *plan {
		return writeJSON(stdout, map[string]any{"contract": contract, "contract_id": contract.ID(), "requests": len(rows), "unique_features": len(unique), "encoder_tokens": tokenTotal, "estimated_bytes_upper": estimated})
	}
	if e = ensureFeatureDisk(*root, estimated); e != nil {
		return e
	}
	encoder, e := backbone.NewFrozenGPUEncoder(*model, backbone.FrozenGPUOptions{MaxTokens: 512, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
	if e != nil {
		return e
	}
	defer nvidia.Shutdown()
	defer encoder.Close()
	cache, e := jevlike.OpenFeatureCache(*root, contract, 12<<30, tokenize, encoder.EncodeTokenHiddenStates)
	if e != nil {
		return e
	}
	defer cache.Close()
	started := time.Now()
	newRows := 0
	completed := 0
	for i, ex := range examples {
		before := cache.Misses
		if _, e = cache.Encode(ex.Context, contract.ContextTokens); e != nil {
			return fmt.Errorf("%s context: %w", rows[i].ID, e)
		}
		for _, text := range ex.Options {
			if _, e = cache.EncodeOption(text, contract.OptionTokens); e != nil {
				return fmt.Errorf("%s option: %w", rows[i].ID, e)
			}
		}
		completed++
		if cache.Misses > before {
			newRows++
		}
		fmt.Fprintf(stderr, "row=%d/%d id=%s variant=%s new_features=%d bytes=%d seconds=%.3f\n", completed, len(rows), rows[i].ID, rows[i].Variant, cache.Misses, cache.Bytes(), time.Since(started).Seconds())
		if *maxNew > 0 && newRows >= *maxNew {
			break
		}
	}
	encoder.Close()
	if after, e := verifyScoreAssets(*model, *assets); e != nil || after != id {
		return fmt.Errorf("backbone changed during extraction: %v", e)
	}
	return writeJSON(stdout, map[string]any{"contract_id": contract.ID(), "partition": s.Partition, "request_sha256": s.RequestSHA256, "requests_completed": completed, "requests": len(rows), "complete": completed == len(rows), "hits": cache.Hits, "misses": cache.Misses, "bytes": cache.Bytes(), "elapsed_seconds": time.Since(started).Seconds(), "weights_unchanged": true})
}

func runCachedTrain(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cached-train", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("cache", "", "Feature cache (offline only)")
	dataset := fs.String("dataset", "", "Prepared dataset")
	train := fs.String("train-study", "", "Training selection directory")
	validation := fs.String("validation-study", "", "Validation selection directory")
	state := fs.String("state", "", "Resumable optimiser JSON")
	checkpoint := fs.String("checkpoint", "", "Best completed head checkpoint")
	code := fs.String("code-revision", "", "Pinned code revision")
	seed := fs.Int64("seed", 7, "Initialisation and per-epoch sampler seed")
	epochs := fs.Int("epochs", 5, "Fixed epoch count")
	batch := fs.Int("batch-size", 4, "CPU head batch size")
	rank := fs.Int("rank", 64, "Head rank")
	lr := fs.Float64("learning-rate", 0.002, "Learning rate")
	steps := fs.Int("max-steps", 0, "Intentional interruption after additional steps")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *root == "" || *dataset == "" || *train == "" || *validation == "" || *state == "" || *checkpoint == "" || *code == "" {
		return fmt.Errorf("cache/dataset/studies/state/checkpoint/code-revision required")
	}
	ts, tr, tx, e := featureSelection(*dataset, filepath.Join(*train, "selection.json"), filepath.Join(*train, "requests.jsonl"))
	if e != nil {
		return e
	}
	vs, vr, vx, e := featureSelection(*dataset, filepath.Join(*validation, "selection.json"), filepath.Join(*validation, "requests.jsonl"))
	if e != nil {
		return e
	}
	if ts.Partition != "train" || vs.Partition != "validation" || ts.SourceManifestSHA256 != vs.SourceManifestSHA256 {
		return fmt.Errorf("separate train/validation from same dataset required")
	}
	normal := func(rows []directBatchInput, x []jevlike.ChoiceExample) ([]jevlike.ChoiceExample, error) {
		var out []jevlike.ChoiceExample
		counts := map[string]int{}
		for i, r := range rows {
			if r.Variant == "normal" {
				out = append(out, x[i])
				counts[r.Task]++
			}
		}
		n := -1
		for _, v := range counts {
			if n != -1 && v != n {
				return nil, fmt.Errorf("pilot training selection must be task-balanced")
			}
			n = v
		}
		return out, nil
	}
	tx, e = normal(tr, tx)
	if e != nil {
		return e
	}
	vx, e = normal(vr, vx)
	if e != nil {
		return e
	}
	c, e := jevlike.LoadFeatureContract(*root)
	if e != nil {
		return e
	}
	if c.DatasetSHA256 != ts.SourceManifestSHA256 {
		return fmt.Errorf("cache dataset identity mismatch")
	}
	cache, e := jevlike.OpenFeatureCache(*root, c, 12<<30, nil, nil)
	if e != nil {
		return e
	}
	defer cache.Close()
	head, e := jevlike.NewAttentionHead(c.Width, *rank)
	if e != nil {
		return e
	}
	scorer := &jevlike.FrozenScorer{Config: jevlike.Config{Width: c.Width, Rank: *rank, ContextTokens: c.ContextTokens, OptionTokens: c.OptionTokens}, Reference: cache.FeatureReference(), Head: *head, Encoder: cache}
	th, _ := jevlike.ChoiceExamplesSHA256(tx)
	vh, _ := jevlike.ChoiceExamplesSHA256(vx)
	result, done, e := jevlike.TrainFrozenResumable(scorer, tx, vx, jevlike.TrainConfig{Epochs: *epochs, BatchSize: *batch, LearningRate: float32(*lr), Seed: *seed, MaxGradNorm: 1}, jevlike.FrozenRunIdentity{CacheID: c.ID(), TrainSHA256: th, ValidationSHA256: vh, CodeRevision: *code}, *state, *steps)
	if e != nil {
		return e
	}
	if done {
		ck, e := jevlike.FrozenCheckpoint(scorer)
		if e != nil {
			return e
		}
		if e = jevlike.SaveCheckpoint(*checkpoint, ck); e != nil {
			return e
		}
	}
	// First interrupted batch has no validation NLL yet; omit rather than encode Inf.
	best := any(nil)
	if len(result.History) > 0 {
		best = result.BestValidationNLL
	}
	return writeJSON(stdout, map[string]any{"complete": done, "history": result.History, "best_validation_nll": best, "cache_hits": cache.Hits, "cache_misses": cache.Misses, "train_examples": len(tx), "validation_examples": len(vx), "seed": *seed})
}
func runCachedScore(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cached-score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("cache", "", "Feature root")
	checkpoint := fs.String("checkpoint", "", "Frozen head")
	dataset := fs.String("dataset", "", "Dataset")
	study := fs.String("study", "", "Selection directory")
	output := fs.String("output", "", "New results JSONL")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *root == "" || *checkpoint == "" || *dataset == "" || *study == "" || *output == "" {
		return fmt.Errorf("cache/checkpoint/dataset/study/output required")
	}
	s, rows, examples, e := featureSelection(*dataset, filepath.Join(*study, "selection.json"), filepath.Join(*study, "requests.jsonl"))
	if e != nil {
		return e
	}
	c, e := jevlike.LoadFeatureContract(*root)
	if e != nil {
		return e
	}
	if c.DatasetSHA256 != s.SourceManifestSHA256 {
		return fmt.Errorf("cache dataset mismatch")
	}
	cache, e := jevlike.OpenFeatureCache(*root, c, 12<<30, nil, nil)
	if e != nil {
		return e
	}
	defer cache.Close()
	ck, e := jevlike.LoadCheckpoint(*checkpoint)
	if e != nil {
		return e
	}
	scorer, e := ck.Frozen(cache)
	if e != nil {
		return e
	}
	b, e := os.ReadFile(*checkpoint)
	if e != nil {
		return e
	}
	id := "jevlike-head@" + directHash(b) + "/" + cache.FeatureReference()
	f, e := os.OpenFile(*output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for i, ex := range examples {
		started := time.Now()
		logits, e := scorer.Forward([]jevlike.ChoiceExample{ex}, false)
		if e != nil {
			return e
		}
		r := jevlike.DirectChoiceResult{Logits: logits[0], ProjectionBackend: "cpu-frozen-attention-head"}
		best := 0
		for j, opt := range rows[i].Request.Candidates {
			r.IDs = append(r.IDs, opt.ID)
			if r.Logits[j] > r.Logits[best] {
				best = j
			}
		}
		r.SelectedID = r.IDs[best]
		entry := directBatchOutput{ID: rows[i].ID, Task: rows[i].Task, Variant: rows[i].Variant, GoldID: rows[i].GoldID, ModelID: id, Elapsed: time.Since(started).Seconds(), Result: &r}
		if e = enc.Encode(entry); e != nil {
			return e
		}
	}
	if e = f.Sync(); e != nil {
		return e
	}
	return writeJSON(stdout, map[string]any{"rows": len(rows), "model_id": id, "cache_hits": cache.Hits, "cache_misses": cache.Misses})
}
