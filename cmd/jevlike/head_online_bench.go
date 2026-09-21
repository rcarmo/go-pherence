package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"sort"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	backbone "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/jevlike"
)

type onlineFeatureEncoder struct {
	live                          *jevlike.FeatureEncoder
	context                       [][]float32
	options                       map[string][]float32
	mode                          int
	contexts, optionCalls         int
	contextSeconds, optionSeconds float64
}

func (e *onlineFeatureEncoder) FeatureReference() string { return e.live.FeatureReference() }
func (e *onlineFeatureEncoder) Encode(text string, limit int) ([][]float32, error) {
	if e.mode == 2 {
		return e.context, nil
	}
	start := time.Now()
	rows, err := e.live.Encode(text, limit)
	e.contextSeconds += time.Since(start).Seconds()
	e.contexts++
	return rows, err
}
func (e *onlineFeatureEncoder) EncodeOption(text string, limit int) ([]float32, error) {
	if e.mode > 0 {
		v, ok := e.options[text]
		if !ok {
			return nil, fmt.Errorf("candidate cache miss")
		}
		return v, nil
	}
	start := time.Now()
	rows, err := e.live.EncodeOption(text, limit)
	e.optionSeconds += time.Since(start).Seconds()
	e.optionCalls++
	return rows, err
}

type onlineLatencyObservation struct {
	Seconds                float64 `json:"seconds"`
	ContextSeconds         float64 `json:"context_extraction_seconds"`
	OptionSeconds          float64 `json:"candidate_extraction_seconds"`
	HeadAndOverheadSeconds float64 `json:"head_and_overhead_seconds"`
	ContextForwards        int     `json:"context_forwards"`
	OptionForwards         int     `json:"candidate_forwards"`
}

func runHeadOnlineBench(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("head-online-bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("encoder-model", "", "Model")
	assets := fs.String("verified-assets", "", "Verified file set")
	root := fs.String("contract", "", "Contract directory")
	checkpoint := fs.String("checkpoint", "", "Head")
	request := fs.String("request", "", "Self-contained candidate request")
	repeats := fs.Int("repeats", 5, "1..20 repeats per scenario")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *model == "" || *assets == "" || *root == "" || *checkpoint == "" || *request == "" || *repeats < 1 || *repeats > 20 {
		return fmt.Errorf("all paths and 1..20 repeats required")
	}
	started := time.Now()
	id, e := verifyScoreAssets(*model, *assets)
	if e != nil {
		return e
	}
	hashSeconds := time.Since(started).Seconds()
	contract, e := jevlike.LoadFeatureContract(*root)
	if e != nil {
		return e
	}
	if contract.ModelID != id {
		return fmt.Errorf("model identity mismatch")
	}
	bytes, e := os.ReadFile(*request)
	if e != nil {
		return e
	}
	var r jevlike.DirectChoiceRequest
	if e = json.Unmarshal(bytes, &r); e != nil {
		return e
	}
	if e = jevlike.ValidateCandidateContract(r); e != nil {
		return e
	}
	ex, e := featureRequestExample(directBatchInput{Request: r, GoldID: r.Candidates[0].ID})
	if e != nil {
		return e
	}
	tok, e := tokenizer.LoadWithConfig(*model)
	if e != nil {
		return e
	}
	tokenize := func(s string) ([]int, error) { return tok.Encode(s), nil }
	started = time.Now()
	encoder, e := backbone.NewFrozenGPUEncoder(*model, backbone.FrozenGPUOptions{MaxTokens: 512, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
	if e != nil {
		return e
	}
	defer nvidia.Shutdown()
	defer encoder.Close()
	loadSeconds := time.Since(started).Seconds()
	live := &jevlike.FeatureEncoder{Contract: contract, Tokenize: tokenize, EncodeTokens: encoder.EncodeTokenHiddenStates}
	// Record setup costs separately. Cached scenarios reuse immutable features of
	// exactly this request and this model/representation, never trained projections.
	started = time.Now()
	optionCache := map[string][]float32{}
	var optionTokens []int
	for _, text := range ex.Options {
		optionTokens = append(optionTokens, len(tok.Encode(text)))
		v, e := live.EncodeOption(text, contract.OptionTokens)
		if e != nil {
			return e
		}
		optionCache[text] = v
	}
	candidateSetup := time.Since(started).Seconds()
	started = time.Now()
	context, e := live.Encode(ex.Context, contract.ContextTokens)
	if e != nil {
		return e
	}
	contextSetup := time.Since(started).Seconds()
	ck, e := jevlike.LoadCheckpoint(*checkpoint)
	if e != nil {
		return e
	}
	var reference []float32
	scenarios := []any{}
	for mode, name := range []string{"fresh-context-fresh-candidates", "fresh-context-reused-candidates", "cached-context-cached-candidates"} {
		source := &onlineFeatureEncoder{live: live, context: context, options: optionCache, mode: mode}
		scorer, e := ck.Frozen(source)
		if e != nil {
			return e
		}
		var observations []onlineLatencyObservation
		var timings []float64
		for range *repeats {
			source.contexts = 0
			source.optionCalls = 0
			source.contextSeconds = 0
			source.optionSeconds = 0
			start := time.Now()
			got, e := scorer.Forward([]jevlike.ChoiceExample{ex}, false)
			if e != nil {
				return e
			}
			seconds := time.Since(start).Seconds()
			if reference == nil {
				reference = append([]float32(nil), got[0]...)
			} else if !reflect.DeepEqual(reference, got[0]) {
				return fmt.Errorf("online scenario feature reuse changed logits")
			}
			wantContexts, wantOptions := 1, len(ex.Options)
			if mode > 0 {
				wantOptions = 0
			}
			if mode == 2 {
				wantContexts = 0
			}
			if source.contexts != wantContexts || source.optionCalls != wantOptions {
				return fmt.Errorf("incorrect scenario encoder calls")
			}
			observations = append(observations, onlineLatencyObservation{seconds, source.contextSeconds, source.optionSeconds, seconds - source.contextSeconds - source.optionSeconds, source.contexts, source.optionCalls})
			timings = append(timings, seconds)
		}
		sort.Float64s(timings)
		scenarios = append(scenarios, map[string]any{"name": name, "samples": observations, "p50_seconds": timings[int(math.Ceil(float64(len(timings))*.5))-1], "p95_seconds": timings[int(math.Ceil(float64(len(timings))*.95))-1]})
	}
	stats := encoder.Stats()
	encoder.Close()
	free, _ := nvidia.MemInfo()
	if free+(32<<20) < stats.FreeBytesAtLoad {
		return fmt.Errorf("GPU cleanup failed")
	}
	return writeJSON(stdout, map[string]any{"version": 1, "model_id": id, "feature_contract": contract.ID(), "request_sha256": directHash(bytes), "candidate_contract": r.CandidateContract, "semantic_review_verified": false, "hash_seconds": hashSeconds, "model_load_seconds": loadSeconds, "candidate_cache_setup_seconds": candidateSetup, "context_cache_setup_seconds": contextSetup, "context_tokens": len(tok.Encode(ex.Context)), "candidate_tokens": optionTokens, "scenarios": scenarios, "exact_logits_match": true, "logits": reference, "free_after_close": free, "exclusions": "warm resident model; no network/queue time; hashing/loading and cache setup reported separately; full-cache uses RAM, not disk; no transformer-prefix reuse"})
}
