package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

func runCacheCompare(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cache-compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	source := fs.String("source", "", "F32 cache")
	target := fs.String("target", "", "New F16 cache")
	checkpoint := fs.String("checkpoint", "", "F32-trained head")
	dataset := fs.String("dataset", "", "Dataset")
	study := fs.String("study", "", "Validation study")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *source == "" || *target == "" || *checkpoint == "" || *dataset == "" || *study == "" {
		return fmt.Errorf("source/target/checkpoint/dataset/study required")
	}
	// Combined F32+F16 storage must stay below the same pilot-cache budget.
	c, e := jevlike.LoadFeatureContract(*source)
	if e != nil {
		return e
	}
	src, e := jevlike.OpenFeatureCache(*source, c, 12<<30, nil, nil)
	if e != nil {
		return e
	}
	defer src.Close()
	remaining := int64(12<<30) - src.Bytes()
	if remaining <= 0 {
		return fmt.Errorf("no combined cache budget")
	}
	if e = ensureFeatureDisk(filepath.Dir(*target), src.Bytes()); e != nil {
		return e
	}
	count, e := jevlike.ConvertFeatureCache(*source, *target, remaining)
	if e != nil {
		return e
	}
	tc := c
	tc.DType = "f16"
	dst, e := jevlike.OpenFeatureCache(*target, tc, remaining, nil, nil)
	if e != nil {
		return e
	}
	defer dst.Close()
	s, _, examples, e := featureSelection(*dataset, filepath.Join(*study, "selection.json"), filepath.Join(*study, "requests.jsonl"))
	if e != nil {
		return e
	}
	if s.Partition != "validation" || s.SourceManifestSHA256 != c.DatasetSHA256 {
		return fmt.Errorf("validation identity required")
	}
	ck, e := jevlike.LoadCheckpoint(*checkpoint)
	if e != nil {
		return e
	}
	a, e := ck.Frozen(src)
	if e != nil {
		return e
	}
	// This is an explicit representation ablation, never a compatible load. Bind
	// the copied head to the new contract and record both IDs in the comparison.
	copied := ck
	copied.EncoderReference = dst.FeatureReference()
	b, e := copied.Frozen(dst)
	if e != nil {
		return e
	}
	var maxFeature, maxLogit float64
	changed := 0
	var raw, rounded [][]float32
	var labels []int
	compare := func(x, y [][]float32) {
		for i := range x {
			for j, v := range x[i] {
				maxFeature = math.Max(maxFeature, math.Abs(float64(v-y[i][j])))
			}
		}
	}
	for _, ex := range examples {
		ca, e := src.Encode(ex.Context, c.ContextTokens)
		if e != nil {
			return e
		}
		cb, e := dst.Encode(ex.Context, c.ContextTokens)
		if e != nil {
			return e
		}
		compare(ca, cb)
		for _, text := range ex.Options {
			oa, e := src.EncodeOption(text, c.OptionTokens)
			if e != nil {
				return e
			}
			ob, e := dst.EncodeOption(text, c.OptionTokens)
			if e != nil {
				return e
			}
			compare([][]float32{oa}, [][]float32{ob})
		}
		x, e := a.Forward([]jevlike.ChoiceExample{ex}, false)
		if e != nil {
			return e
		}
		y, e := b.Forward([]jevlike.ChoiceExample{ex}, false)
		if e != nil {
			return e
		}
		bestA, bestB := 0, 0
		for j, v := range x[0] {
			maxLogit = math.Max(maxLogit, math.Abs(float64(v-y[0][j])))
			if v > x[0][bestA] {
				bestA = j
			}
			if y[0][j] > y[0][bestB] {
				bestB = j
			}
		}
		if bestA != bestB {
			changed++
		}
		raw = append(raw, x[0])
		rounded = append(rounded, y[0])
		labels = append(labels, ex.Label)
	}
	m1, e := jevlike.DecisionMetricsAtTemperature(raw, labels, 1)
	if e != nil {
		return e
	}
	m2, e := jevlike.DecisionMetricsAtTemperature(rounded, labels, 1)
	if e != nil {
		return e
	}
	return writeJSON(stdout, map[string]any{"source_contract": c.ID(), "target_contract": tc.ID(), "entries": count, "combined_bytes": src.Bytes() + dst.Bytes(), "examples_including_controls": len(examples), "max_feature_delta": maxFeature, "max_logit_delta": maxLogit, "changed_decisions": changed, "f32": m1, "f16": m2, "adopted": false})
}

func runCachedSimilarity(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cached-similarity", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("cache", "", "Cache")
	dataset := fs.String("dataset", "", "Dataset")
	study := fs.String("study", "", "Study directory")
	output := fs.String("output", "", "New JSONL results")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *root == "" || *dataset == "" || *study == "" || *output == "" {
		return fmt.Errorf("paths required")
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
	f, e := os.OpenFile(*output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for i, ex := range examples {
		ctx, e := cache.Encode(ex.Context, c.ContextTokens)
		if e != nil {
			return e
		}
		mean := make([]float64, c.Width)
		for _, v := range ctx {
			for j, x := range v {
				mean[j] += float64(x) / float64(len(ctx))
			}
		}
		result := jevlike.DirectChoiceResult{ProjectionBackend: "cpu-mean-feature-cosine"}
		best := 0
		for j, text := range ex.Options {
			v, e := cache.EncodeOption(text, c.OptionTokens)
			if e != nil {
				return e
			}
			var dot, n1, n2 float64
			for d, x := range v {
				dot += mean[d] * float64(x)
				n1 += mean[d] * mean[d]
				n2 += float64(x) * float64(x)
			}
			if n1 == 0 || n2 == 0 {
				return fmt.Errorf("zero feature norm")
			}
			result.Logits = append(result.Logits, float32(dot/math.Sqrt(n1*n2)))
			result.IDs = append(result.IDs, rows[i].Request.Candidates[j].ID)
			if result.Logits[j] > result.Logits[best] {
				best = j
			}
		}
		result.SelectedID = result.IDs[best]
		if e = enc.Encode(directBatchOutput{ID: rows[i].ID, Task: rows[i].Task, Variant: rows[i].Variant, GoldID: rows[i].GoldID, ModelID: "mean-feature-cosine@" + c.ID(), Result: &result}); e != nil {
			return e
		}
	}
	if e = f.Sync(); e != nil {
		return e
	}
	return writeJSON(stdout, map[string]any{"rows": len(rows), "warning": "cosine is a diagnostic, not a calibrated probability model"})
}
