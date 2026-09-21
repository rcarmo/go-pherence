package main

import (
	"flag"
	"fmt"
	"io"
	"reflect"
	"sort"
	"sync"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	backbone "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/jevlike"
)

type directBenchCase struct {
	ID                  string                          `json:"id"`
	Tokens              int                             `json:"tokens"`
	Choices             int                             `json:"choices"`
	TokenisationSeconds float64                         `json:"tokenisation_seconds"`
	DecisionSeconds     float64                         `json:"decision_seconds"`
	Phases              backbone.FrozenGPUDecisionStats `json:"phases"`
}

// Direct-bench measures independent requests; it never fits parameters or reads
// final-test data. Two concurrent callers share the encoder's serialised buffer.
func runDirectBench(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("direct-bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("encoder-model", "", "Local pinned checkpoint")
	assets := fs.String("verified-assets", "", "Verified asset manifest")
	spec := fs.String("selection", "", "Validation selection manifest")
	requests := fs.String("requests", "", "Exact study requests")
	dataset := fs.String("dataset", "", "Prepared dataset")
	repeats := fs.Int("repeats", 3, "Warm repeats per selected token length (1..20)")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if *dir == "" || *assets == "" || *spec == "" || *requests == "" || *dataset == "" || *repeats < 1 || *repeats > 20 {
		return fmt.Errorf("paths and bounded repeat count required")
	}
	selection, rows, err := readDirectSelection(*spec, *requests, *dataset)
	if err != nil {
		return err
	}
	if selection.Partition != "validation" {
		return fmt.Errorf("benchmark uses validation only")
	}
	started := time.Now()
	id, err := verifyScoreAssets(*dir, *assets)
	if err != nil {
		return err
	}
	hashSeconds := time.Since(started).Seconds()
	prompt, err := jevlike.LoadQwen3ChoicePrompt(*dir, 512)
	if err != nil {
		return err
	}
	type item struct {
		row    directBatchInput
		tokens int
	}
	var admitted []item
	for _, r := range rows {
		if r.Variant != "normal" {
			continue
		}
		_, ids, _, err := prompt.Prepare(r.Request)
		if err == nil {
			admitted = append(admitted, item{r, len(ids)})
		}
	}
	if len(admitted) < 3 {
		return fmt.Errorf("need three admitted normal requests")
	}
	sort.Slice(admitted, func(i, j int) bool {
		if admitted[i].tokens != admitted[j].tokens {
			return admitted[i].tokens < admitted[j].tokens
		}
		return admitted[i].row.ID < admitted[j].row.ID
	})
	samples := []item{admitted[0], admitted[len(admitted)/2], admitted[len(admitted)-1]}
	started = time.Now()
	e, err := backbone.NewFrozenGPUEncoder(*dir, backbone.FrozenGPUOptions{MaxTokens: 512, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
	if err != nil {
		return err
	}
	defer nvidia.Shutdown()
	defer e.Close()
	loadSeconds := time.Since(started).Seconds()
	free, total := nvidia.MemInfo()
	run := func(x item) (directBenchCase, []float32, error) {
		begun := time.Now()
		_, ids, codes, err := prompt.Prepare(x.row.Request)
		tokenSeconds := time.Since(begun).Seconds()
		if err != nil {
			return directBenchCase{}, nil, err
		}
		logits, phases, err := e.ProfileSelectedLogits(ids, codes)
		return directBenchCase{x.row.ID, len(ids), len(codes), tokenSeconds, time.Since(begun).Seconds(), phases}, logits, err
	}
	cold, _, err := run(samples[1])
	if err != nil {
		return err
	}
	var warm []directBenchCase
	for range *repeats {
		for _, x := range samples {
			entry, _, err := run(x)
			if err != nil {
				return err
			}
			warm = append(warm, entry)
		}
	}
	var reference [2][]float32
	for i := range 2 {
		_, reference[i], err = run(samples[i*2])
		if err != nil {
			return err
		}
	}
	var concurrent [2]directBenchCase
	var errors [2]error
	var logits [2][]float32
	var wg sync.WaitGroup
	barrier := make(chan struct{})
	begun := time.Now()
	for i := range 2 {
		wg.Add(1)
		go func(i int) { defer wg.Done(); <-barrier; concurrent[i], logits[i], errors[i] = run(samples[i*2]) }(i)
	}
	close(barrier)
	wg.Wait()
	concurrentSeconds := time.Since(begun).Seconds()
	for i := range 2 {
		if errors[i] != nil {
			return errors[i]
		}
		if !reflect.DeepEqual(reference[i], logits[i]) {
			return fmt.Errorf("concurrent request interference")
		}
	}
	stats := e.Stats()
	e.Close()
	freeAfter, _ := nvidia.MemInfo()
	if freeAfter+32<<20 < stats.FreeBytesAtLoad {
		return fmt.Errorf("GPU memory not released")
	}
	return writeJSON(stdout, map[string]any{"version": 1, "model_id": id, "request_sha256": selection.RequestSHA256, "hash_seconds": hashSeconds, "load_seconds": loadSeconds, "gpu": stats, "free_after_load": free, "total_gpu_bytes": total, "free_after_close": freeAfter, "cold_first_request": cold, "warm": warm, "concurrent": concurrent, "concurrent_wall_seconds": concurrentSeconds, "concurrent_matches_independent": true, "policy": "fresh process/load, OS file cache not dropped; wall-clock synchronised phases; encoder serialises two callers; no prefix reuse"})
}
