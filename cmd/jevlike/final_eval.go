package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	backbone "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/jevlike"
)

type finalFreeze struct {
	Version           int    `json:"version"`
	Scope             string `json:"scope"`
	Dataset           string `json:"dataset"`
	DatasetSHA256     string `json:"dataset_sha256"`
	PartitionSHA256   string `json:"partition_sha256"`
	Expected          int    `json:"expected_examples"`
	ModelID           string `json:"model_id"`
	Assets            string `json:"verified_assets"`
	Contract          string `json:"feature_contract"`
	DirectCalibration string `json:"direct_calibration"`
	Heads             []struct {
		Seed        int    `json:"seed"`
		Checkpoint  string `json:"checkpoint"`
		Calibration string `json:"calibration"`
	} `json:"heads"`
	Files map[string]string `json:"files"`
}
type finalEntry struct {
	Version        int                          `json:"version"`
	FreezeSHA256   string                       `json:"freeze_sha256"`
	RequestSHA256  string                       `json:"request_sha256"`
	RequestKey     string                       `json:"request_key"`
	Outcomes       map[string]directBatchOutput `json:"outcomes"`
	FeatureSeconds float64                      `json:"feature_seconds"`
}

func verifyFinalFreeze(path string) (finalFreeze, string, error) {
	var f finalFreeze
	b, e := os.ReadFile(path)
	if e != nil {
		return f, "", e
	}
	if e = json.Unmarshal(b, &f); e != nil {
		return f, "", e
	}
	if f.Version != 1 || f.Scope != "final-once-normal-v1" || f.Expected < 1 || len(f.Heads) != 3 || len(f.Files) < 1 || f.ModelID == "" {
		return f, "", fmt.Errorf("explicit complete final freeze required")
	}
	for p, h := range f.Files {
		b, e := os.ReadFile(p)
		if e != nil {
			return f, "", e
		}
		if directHash(b) != h {
			return f, "", fmt.Errorf("frozen file changed: %s", p)
		}
	}
	for _, p := range []string{f.Assets, f.Contract, f.DirectCalibration, filepath.Join(f.Dataset, "manifest.json")} {
		if len(f.Files[p]) != 64 {
			return f, "", fmt.Errorf("required file absent from freeze: %s", p)
		}
	}
	for i, h := range f.Heads {
		if h.Seed != []int{7, 17, 27}[i] || len(f.Files[h.Checkpoint]) != 64 || len(f.Files[h.Calibration]) != 64 {
			return f, "", fmt.Errorf("invalid frozen heads")
		}
	}
	return f, directHash(b), nil
}
func writeFinalEntry(path string, entry finalEntry) error {
	b, e := json.Marshal(entry)
	if e != nil {
		return e
	}
	b = append(b, '\n')
	f, e := os.CreateTemp(filepath.Dir(path), ".final-*.tmp")
	if e != nil {
		return e
	}
	defer os.Remove(f.Name())
	if _, e = f.Write(b); e != nil {
		f.Close()
		return e
	}
	if e = f.Sync(); e != nil {
		f.Close()
		return e
	}
	if e = f.Close(); e != nil {
		return e
	}
	if e = os.Link(f.Name(), path); e != nil {
		return e
	}
	d, e := os.Open(filepath.Dir(path))
	if e != nil {
		return e
	}
	defer d.Close()
	return d.Sync()
}
func validateFinalEntry(entry finalEntry, freeze, requestHash string, row directBatchInput, models map[string]string) error {
	if entry.Version != 1 || entry.FreezeSHA256 != freeze || entry.RequestSHA256 != requestHash || entry.RequestKey != directRowKey(row.Task, row.ID, row.Variant) || len(entry.Outcomes) != len(models) || math.IsNaN(entry.FeatureSeconds) || math.IsInf(entry.FeatureSeconds, 0) || entry.FeatureSeconds < 0 {
		return fmt.Errorf("final record identity/shape mismatch")
	}
	for arm, id := range models {
		r, ok := entry.Outcomes[arm]
		if !ok || r.ModelID != id || r.ID != row.ID || r.Task != row.Task || r.Variant != row.Variant || r.GoldID != row.GoldID || (r.Error != "") == (r.Result != nil) || math.IsNaN(r.Elapsed) || math.IsInf(r.Elapsed, 0) || r.Elapsed < 0 {
			return fmt.Errorf("final outcome mismatch: %s", arm)
		}
		if r.Result == nil {
			continue
		}
		x := r.Result
		if len(x.IDs) != len(row.Request.Candidates) || len(x.Logits) != len(x.IDs) {
			return fmt.Errorf("incomplete final logits")
		}
		best := 0
		for i, c := range row.Request.Candidates {
			if x.IDs[i] != c.ID || math.IsNaN(float64(x.Logits[i])) || math.IsInf(float64(x.Logits[i]), 0) {
				return fmt.Errorf("invalid final candidate/logit")
			}
			if x.Logits[i] > x.Logits[best] {
				best = i
			}
		}
		if x.SelectedID != x.IDs[best] {
			return fmt.Errorf("final selected ID mismatch")
		}
	}
	return nil
}
func runFinalEval(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("final-eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	freezePath := fs.String("freeze", "", "Committed pre-test freeze")
	study := fs.String("study", "", "Full final requests/selection")
	dir := fs.String("encoder-model", "", "Pinned instruction model")
	cacheRoot := fs.String("cache", "", "Existing frozen F32 feature directory")
	otherCache := fs.String("other-cache", "", "Existing F16 cache for combined accounting")
	output := fs.String("output", "", "Directory of immutable per-original records")
	planOnly := fs.Bool("plan", false, "Verify and account without GPU work")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *freezePath == "" || *study == "" || *dir == "" || *cacheRoot == "" || *otherCache == "" || *output == "" {
		return fmt.Errorf("freeze/study/model/cache/other-cache/output required")
	}
	f, freezeHash, e := verifyFinalFreeze(*freezePath)
	if e != nil {
		return e
	}
	selectionPath := filepath.Join(*study, "selection.json")
	b, e := os.ReadFile(selectionPath)
	if e != nil {
		return e
	}
	var gate struct {
		Freeze string `json:"final_freeze_sha256"`
	}
	if e = json.Unmarshal(b, &gate); e != nil {
		return e
	}
	if gate.Freeze != freezeHash {
		return fmt.Errorf("selection not bound to final freeze")
	}
	s, expected, e := readDirectSelection(selectionPath, filepath.Join(*study, "requests.jsonl"), f.Dataset)
	if e != nil {
		return e
	}
	if s.Partition != "test" || s.Requests != f.Expected || s.PartitionSHA256 != f.PartitionSHA256 || s.SourceManifestSHA256 != f.DatasetSHA256 {
		return fmt.Errorf("wrong final selection")
	}
	rows, e := orderedFeatureRequests(filepath.Join(*study, "requests.jsonl"), expected)
	if e != nil {
		return e
	}
	if len(rows) != f.Expected {
		return fmt.Errorf("incomplete final requests")
	}
	for _, r := range rows {
		if r.Variant != "normal" {
			return fmt.Errorf("only frozen normal variant permitted")
		}
	}
	modelID, e := verifyScoreAssets(*dir, f.Assets)
	if e != nil {
		return e
	}
	if modelID != f.ModelID {
		return fmt.Errorf("wrong model bytes")
	}
	contract, e := jevlike.LoadFeatureContract(*cacheRoot)
	if e != nil {
		return e
	}
	cb, e := os.ReadFile(filepath.Join(*cacheRoot, "contract.json"))
	if e != nil || directHash(cb) != f.Files[f.Contract] {
		return fmt.Errorf("feature contract mismatch")
	}
	tok, e := tokenizer.LoadWithConfig(*dir)
	if e != nil {
		return e
	}
	prompt, e := jevlike.LoadQwen3ChoicePrompt(*dir, 512)
	if e != nil {
		return e
	}
	modelIDs := map[string]string{"instruction": f.ModelID}
	heads := map[string]*jevlike.FrozenScorer{}
	checkpoints := map[string]jevlike.Checkpoint{}
	for _, h := range f.Heads {
		ck, e := jevlike.LoadCheckpoint(h.Checkpoint)
		if e != nil {
			return e
		}
		arm := fmt.Sprintf("seed-%d", h.Seed)
		checkpoints[arm] = ck
		modelIDs[arm] = "jevlike-head@" + f.Files[h.Checkpoint] + "/feature:" + contract.ID()
	}
	examples := make([]jevlike.ChoiceExample, len(rows))
	headReject := make([]string, len(rows))
	directRejected := 0
	unique := map[string]int64{}
	for i, r := range rows {
		if _, _, _, e := prompt.Prepare(r.Request); e != nil {
			directRejected++
		}
		ex, e := featureRequestExample(r)
		if e != nil {
			return e
		}
		examples[i] = ex
		for j, text := range append([]string{ex.Context}, ex.Options...) {
			limit := contract.OptionTokens
			if j == 0 {
				limit = contract.ContextTokens
			}
			n := len(tok.Encode(text))
			if n < 1 || n > limit {
				headReject[i] = fmt.Sprintf("feature tokens=%d limit=%d; no truncation", n, limit)
			}
			count := 1
			if j == 0 {
				count = n
			}
			key := fmt.Sprintf("%t:%s", j == 0, text)
			unique[key] = int64(count*contract.Width*4 + len(text) + n*12 + 4096)
		}
	}
	var estimated int64
	for _, n := range unique {
		estimated += n
	}
	existing, e := finalDirBytes(*cacheRoot)
	if e != nil {
		return e
	}
	other, e := finalDirBytes(*otherCache)
	if e != nil {
		return e
	}
	if estimated+existing+other > 12<<30 {
		return fmt.Errorf("combined feature upper bound exceeds 12 GiB")
	}
	if e = ensureFeatureDisk(*cacheRoot, estimated); e != nil {
		return e
	}
	rejectedHeads := 0
	for _, s := range headReject {
		if s != "" {
			rejectedHeads++
		}
	}
	if *planOnly {
		return writeJSON(stdout, map[string]any{"examples": len(rows), "freeze_sha256": freezeHash, "request_sha256": s.RequestSHA256, "estimated_additional_bytes_upper": estimated, "combined_upper_bytes": estimated + existing + other, "direct_admission_rejections": directRejected, "head_admission_rejections": rejectedHeads, "model_ids": modelIDs})
	}
	if e = os.MkdirAll(*output, 0o755); e != nil {
		return e
	}
	lock := filepath.Join(*output, ".run.lock")
	if e = os.Mkdir(lock, 0o700); e != nil {
		return fmt.Errorf("final run lock: %w", e)
	}
	defer os.Remove(lock)
	encoder, e := backbone.NewFrozenGPUEncoder(*dir, backbone.FrozenGPUOptions{MaxTokens: 512, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
	if e != nil {
		return e
	}
	defer nvidia.Shutdown()
	defer encoder.Close()
	cache, e := jevlike.OpenFeatureCache(*cacheRoot, contract, (12<<30)-other, func(text string) ([]int, error) { return tok.Encode(text), nil }, encoder.EncodeTokenHiddenStates)
	if e != nil {
		return e
	}
	defer cache.Close()
	for arm, ck := range checkpoints {
		m, e := ck.Frozen(cache)
		if e != nil {
			return e
		}
		heads[arm] = m
	}
	start := time.Now()
	resumed := 0
	for i, row := range rows {
		key := directRowKey(row.Task, row.ID, row.Variant)
		path := filepath.Join(*output, fmt.Sprintf("%04d-%s.json", i+1, directHash([]byte(key))[:16]))
		if bytes, e := os.ReadFile(path); e == nil {
			var entry finalEntry
			if e = json.Unmarshal(bytes, &entry); e != nil {
				return e
			}
			if e = validateFinalEntry(entry, freezeHash, s.RequestSHA256, row, modelIDs); e != nil {
				return e
			}
			resumed++
			continue
		} else if !os.IsNotExist(e) {
			return e
		}
		entry := finalEntry{Version: 1, FreezeSHA256: freezeHash, RequestSHA256: s.RequestSHA256, RequestKey: key, Outcomes: map[string]directBatchOutput{}}
		base := func(arm string) directBatchOutput {
			return directBatchOutput{ID: row.ID, Task: row.Task, Variant: row.Variant, GoldID: row.GoldID, ModelID: modelIDs[arm]}
		}
		r := base("instruction")
		begun := time.Now()
		if _, _, _, e := prompt.Prepare(row.Request); e != nil {
			r.Error = e.Error()
		} else {
			result, e := jevlike.ScoreChoices(encoder, prompt, row.Request)
			if e != nil {
				return fmt.Errorf("direct row %d: %w", i+1, e)
			}
			result.ProjectionBackend = "cpu-f64-selected-bf16-rows"
			r.Result = &result
		}
		r.Elapsed = time.Since(begun).Seconds()
		entry.Outcomes["instruction"] = r
		begun = time.Now()
		if headReject[i] == "" {
			if _, e = cache.Encode(examples[i].Context, contract.ContextTokens); e != nil {
				return e
			}
			for _, text := range examples[i].Options {
				if _, e = cache.EncodeOption(text, contract.OptionTokens); e != nil {
					return e
				}
			}
		}
		entry.FeatureSeconds = time.Since(begun).Seconds()
		for _, arm := range []string{"seed-7", "seed-17", "seed-27"} {
			r := base(arm)
			begun = time.Now()
			if headReject[i] != "" {
				r.Error = headReject[i]
			} else {
				logits, e := heads[arm].Forward([]jevlike.ChoiceExample{examples[i]}, false)
				if e != nil {
					return e
				}
				result := jevlike.DirectChoiceResult{Logits: logits[0], ProjectionBackend: "cpu-frozen-attention-head"}
				best := 0
				for j, c := range row.Request.Candidates {
					result.IDs = append(result.IDs, c.ID)
					if result.Logits[j] > result.Logits[best] {
						best = j
					}
				}
				result.SelectedID = result.IDs[best]
				r.Result = &result
			}
			r.Elapsed = time.Since(begun).Seconds()
			entry.Outcomes[arm] = r
		}
		if e = validateFinalEntry(entry, freezeHash, s.RequestSHA256, row, modelIDs); e != nil {
			return e
		}
		if e = writeFinalEntry(path, entry); e != nil {
			return e
		}
		fmt.Fprintf(stderr, "final row=%d/%d seconds=%.3f cache_bytes=%d direct_rejected=%v head_rejected=%v\n", i+1, len(rows), time.Since(start).Seconds(), cache.Bytes(), entry.Outcomes["instruction"].Error != "", headReject[i] != "")
	}
	encoder.Close()
	if id, e := verifyScoreAssets(*dir, f.Assets); e != nil || id != f.ModelID {
		return fmt.Errorf("weights changed after run: %v", e)
	}
	if _, after, e := verifyFinalFreeze(*freezePath); e != nil || after != freezeHash {
		return fmt.Errorf("freeze changed after run: %v", e)
	}
	return writeJSON(stdout, map[string]any{"complete": true, "examples": len(rows), "resumed": resumed, "freeze_sha256": freezeHash, "request_sha256": s.RequestSHA256, "cache_bytes": cache.Bytes(), "other_cache_bytes": other, "elapsed_seconds": time.Since(start).Seconds(), "weights_unchanged": true})
}
func finalDirBytes(root string) (int64, error) {
	var total int64
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("symlink in cache budget")
		}
		if !d.IsDir() {
			s, e := d.Info()
			if e != nil {
				return e
			}
			total += s.Size()
		}
		return nil
	})
	return total, err
}
