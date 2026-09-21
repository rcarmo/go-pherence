package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

func TestCachedCLIRequiresPaths(t *testing.T) {
	for _, command := range []string{"cache-extract", "cached-train", "cached-score"} {
		var out, stderr bytes.Buffer
		if e := run([]string{command, "-h"}, &out, &stderr); e != nil {
			t.Fatal(e)
		}
		if e := run([]string{command}, &out, &stderr); e == nil {
			t.Fatal("missing paths accepted", command)
		}
	}
}
func TestCachedScoreIsOfflineAndBoundToFeatureCheckpoint(t *testing.T) {
	args, row := reportFixture(t)
	dir := args[7]
	selection, _, examples, e := featureSelection(dir, args[3], args[5])
	if e != nil {
		t.Fatal(e)
	}
	contract := frozenFeatureContract("synthetic#sha256="+strings.Repeat("a", 64), selection.SourceManifestSHA256, "f32")
	contract.Width = 4
	cacheDir := filepath.Join(dir, "cache")
	cache, e := jevlike.OpenFeatureCache(cacheDir, contract, 1<<20, func(s string) ([]int, error) { return []int{1}, nil }, func(ids []int) ([][]float32, error) { return [][]float32{{1, 2, 3, 4}}, nil })
	if e != nil {
		t.Fatal(e)
	}
	for _, ex := range examples {
		if _, e = cache.Encode(ex.Context, 512); e != nil {
			t.Fatal(e)
		}
		for _, s := range ex.Options {
			if _, e = cache.EncodeOption(s, 128); e != nil {
				t.Fatal(e)
			}
		}
	}
	head, _ := jevlike.NewAttentionHead(4, 2)
	model := &jevlike.FrozenScorer{Config: jevlike.Config{Width: 4, Rank: 2, ContextTokens: 512, OptionTokens: 128}, Reference: cache.FeatureReference(), Head: *head, Encoder: cache}
	jevlike.InitializeFrozenScorer(model, 7)
	ck, e := jevlike.FrozenCheckpoint(model)
	if e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(dir, "head.json")
	if e = jevlike.SaveCheckpoint(path, ck); e != nil {
		t.Fatal(e)
	}
	cache.Close()
	output := filepath.Join(dir, "scored.jsonl")
	var out, stderr bytes.Buffer
	scoreArgs := []string{"-cache", cacheDir, "-checkpoint", path, "-dataset", dir, "-study", dir, "-output", output}
	if e = runCachedScore(scoreArgs, &out, &stderr); e != nil {
		t.Fatal(e)
	}
	data, e := os.ReadFile(output)
	if e != nil {
		t.Fatal(e)
	}
	var scored directBatchOutput
	if e = json.Unmarshal(data, &scored); e != nil {
		t.Fatal(e)
	}
	if scored.ID != row.ID || scored.GoldID != row.GoldID || scored.Result == nil {
		t.Fatal(scored)
	}
	if e = runCachedScore(scoreArgs, &out, &stderr); e == nil {
		t.Fatal("results overwritten")
	}
	reportArgs := append([]string{}, args...)
	reportArgs[1] = output
	if e = runDirectReport(reportArgs, &out, &stderr); e != nil {
		t.Fatal(e)
	}
}
