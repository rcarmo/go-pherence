package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	backbone "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/jevlike"
)

// Fresh text uses the exact saved feature contract, not the cache-only path.
func runFeatureScore(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("feature-score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("encoder-model", "", "Pinned Qwen3")
	assets := fs.String("verified-assets", "", "Verified files")
	root := fs.String("contract", "", "Directory with feature contract")
	checkpoint := fs.String("checkpoint", "", "Trained frozen head")
	request := fs.String("request", "", "Fresh DirectChoiceRequest JSON")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *model == "" || *assets == "" || *root == "" || *checkpoint == "" || *request == "" {
		return fmt.Errorf("model/assets/contract/checkpoint/request required")
	}
	id, e := verifyScoreAssets(*model, *assets)
	if e != nil {
		return e
	}
	contract, e := jevlike.LoadFeatureContract(*root)
	if e != nil {
		return e
	}
	if contract.ModelID != id {
		return fmt.Errorf("backbone differs from feature contract")
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
	started := time.Now()
	encoder, e := backbone.NewFrozenGPUEncoder(*model, backbone.FrozenGPUOptions{MaxTokens: 512, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
	if e != nil {
		return e
	}
	defer nvidia.Shutdown()
	defer encoder.Close()
	load := time.Since(started).Seconds()
	live := &jevlike.FeatureEncoder{Contract: contract, Tokenize: func(s string) ([]int, error) { return tok.Encode(s), nil }, EncodeTokens: encoder.EncodeTokenHiddenStates}
	ck, e := jevlike.LoadCheckpoint(*checkpoint)
	if e != nil {
		return e
	}
	scorer, e := ck.Frozen(live)
	if e != nil {
		return e
	}
	started = time.Now()
	logits, e := scorer.Forward([]jevlike.ChoiceExample{ex}, false)
	if e != nil {
		return e
	}
	elapsed := time.Since(started).Seconds()
	result := jevlike.DirectChoiceResult{Logits: logits[0], ProjectionBackend: "cpu-frozen-attention-head"}
	best := 0
	for i, c := range r.Candidates {
		result.IDs = append(result.IDs, c.ID)
		if result.Logits[i] > result.Logits[best] {
			best = i
		}
	}
	result.SelectedID = result.IDs[best]
	return writeJSON(stdout, map[string]any{"model_id": id, "feature_contract": contract.ID(), "load_seconds": load, "fresh_decision_seconds": elapsed, "result": result, "cache_reads": 0, "candidate_contract": r.CandidateContract, "semantic_review_verified": false, "gpu": encoder.Stats()})
}
