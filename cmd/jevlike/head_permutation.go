package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"path/filepath"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

func runHeadPermutation(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("head-permutation", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("cache", "", "Offline cache")
	checkpoint := fs.String("checkpoint", "", "Head checkpoint")
	dataset := fs.String("dataset", "", "Dataset")
	study := fs.String("study", "", "Validation selection")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *root == "" || *checkpoint == "" || *dataset == "" || *study == "" {
		return fmt.Errorf("cache/checkpoint/dataset/study required")
	}
	s, rows, examples, e := featureSelection(*dataset, filepath.Join(*study, "selection.json"), filepath.Join(*study, "requests.jsonl"))
	if e != nil {
		return e
	}
	if s.Partition != "validation" {
		return fmt.Errorf("validation only")
	}
	c, e := jevlike.LoadFeatureContract(*root)
	if e != nil {
		return e
	}
	if c.DatasetSHA256 != s.SourceManifestSHA256 {
		return fmt.Errorf("dataset mismatch")
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
	m, e := ck.Frozen(cache)
	if e != nil {
		return e
	}
	comparisons, originals, changedUnique, ties := 0, 0, 0, 0
	var maxDelta float64
	for k, row := range rows {
		if row.Variant != "normal" {
			continue
		}
		ex := examples[k]
		baseline, e := m.Forward([]jevlike.ChoiceExample{ex}, false)
		if e != nil {
			return e
		}
		base := baseline[0]
		best := 0
		for i := range base {
			if base[i] > base[best] {
				best = i
			}
		}
		tie := false
		for i := range base {
			if i != best && base[i] == base[best] {
				tie = true
			}
		}
		if tie {
			ties++
		}
		originals++
		n := len(ex.Options)
		perms := make([][]int, 0, n+1)
		reverse := make([]int, n)
		for i := range n {
			reverse[i] = n - 1 - i
		}
		perms = append(perms, reverse)
		for shift := 1; shift < n; shift++ {
			p := make([]int, n)
			for i := range n {
				p[i] = (i + shift) % n
			}
			perms = append(perms, p)
		}
		for _, p := range perms {
			candidate := jevlike.ChoiceExample{Context: ex.Context, Options: make([]string, n)}
			for j, i := range p {
				candidate.Options[j] = ex.Options[i]
				if i == ex.Label {
					candidate.Label = j
				}
			}
			got, e := m.Forward([]jevlike.ChoiceExample{candidate}, false)
			if e != nil {
				return e
			}
			selected := 0
			for j, i := range p {
				maxDelta = math.Max(maxDelta, math.Abs(float64(got[0][j]-base[i])))
				if got[0][j] > got[0][selected] {
					selected = j
				}
			}
			if !tie && row.Request.Candidates[p[selected]].ID != row.Request.Candidates[best].ID {
				changedUnique++
			}
			comparisons++
		}
	}
	report := map[string]any{"feature_contract": c.ID(), "request_sha256": s.RequestSHA256, "originals": originals, "permutations": comparisons, "policy": "reverse plus every nonzero cyclic rotation; compare logits after stable-ID mapping", "max_mapped_logit_delta": maxDelta, "unique_argmax_changes": changedUnique, "strict_tie_originals": ties, "tie_policy": "first supplied option; a strict-tie winner need not be permutation-invariant", "pass": maxDelta == 0 && changedUnique == 0}
	if err := writeJSON(stdout, report); err != nil {
		return err
	}
	if maxDelta != 0 || changedUnique != 0 {
		return fmt.Errorf("head permutation equivariance failed")
	}
	return nil
}
