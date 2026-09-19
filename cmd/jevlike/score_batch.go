package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	backbone "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/jevlike"
)

type directBatchInput struct {
	ID      string                      `json:"id"`
	Task    string                      `json:"task"`
	Variant string                      `json:"variant"`
	GoldID  string                      `json:"gold_id"`
	Request jevlike.DirectChoiceRequest `json:"request"`
}
type directBatchOutput struct {
	ID      string                      `json:"id"`
	Task    string                      `json:"task"`
	Variant string                      `json:"variant"`
	GoldID  string                      `json:"gold_id"`
	ModelID string                      `json:"model_id"`
	Elapsed float64                     `json:"decision_seconds"`
	Error   string                      `json:"error,omitempty"`
	Result  *jevlike.DirectChoiceResult `json:"result,omitempty"`
}

// score-batch keeps one encoder resident but every row gets independent prefill.
// Per-row admission errors are recorded, never silently discarded/truncated.
func runScoreBatch(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("score-batch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("encoder-model", "", "Local pinned BF16 Qwen3")
	identity := fs.String("verified-assets", "", "Full model-verified.json")
	input := fs.String("requests", "", "JSONL requests with IDs/task/variant/gold sidecar fields")
	output := fs.String("output", "", "New JSONL results file (never overwrites)")
	maxTokens := fs.Int("max-tokens", 512, "Prompt bound")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if *dir == "" || *identity == "" || *input == "" || *output == "" {
		return fmt.Errorf("model, verified assets, requests and output required")
	}
	modelID, err := verifyScoreAssets(*dir, *identity)
	if err != nil {
		return err
	}
	prompt, err := jevlike.LoadQwen3ChoicePrompt(*dir, *maxTokens)
	if err != nil {
		return err
	}
	f, err := os.Open(*input)
	if err != nil {
		return err
	}
	defer f.Close()
	out, err := os.OpenFile(*output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	defer out.Close()
	e, err := backbone.NewFrozenGPUEncoder(*dir, backbone.FrozenGPUOptions{MaxTokens: *maxTokens, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
	if err != nil {
		return err
	}
	defer nvidia.Shutdown()
	defer e.Close()
	fmt.Fprintf(stderr, "model=%s gpu=%+v\n", modelID, e.Stats())
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 64*1024), 1<<20)
	enc := json.NewEncoder(out)
	count := 0
	seen := map[string]bool{}
	for scan.Scan() {
		var row directBatchInput
		dec := json.NewDecoder(bytes.NewReader(scan.Bytes()))
		dec.DisallowUnknownFields()
		if err = dec.Decode(&row); err != nil {
			return err
		}
		if dec.Decode(new(any)) != io.EOF {
			return fmt.Errorf("trailing request data")
		}
		if row.ID == "" || row.Task == "" || row.Variant == "" {
			return fmt.Errorf("row identity required")
		}
		key := directRowKey(row.Task, row.ID, row.Variant)
		if seen[key] {
			return fmt.Errorf("duplicate request %s", row.ID)
		}
		seen[key] = true
		entry := directBatchOutput{ID: row.ID, Task: row.Task, Variant: row.Variant, GoldID: row.GoldID, ModelID: modelID}
		start := time.Now()
		_, _, _, admission := prompt.Prepare(row.Request)
		if row.Request.Temperature <= 0 || math.IsNaN(row.Request.Temperature) || math.IsInf(row.Request.Temperature, 0) {
			admission = fmt.Errorf("positive finite temperature required")
		}
		if admission != nil {
			entry.Error = admission.Error()
		} else {
			result, scoreErr := jevlike.ScoreChoices(e, prompt, row.Request)
			if scoreErr != nil {
				return fmt.Errorf("row %s GPU execution: %w", row.ID, scoreErr)
			}
			result.ProjectionBackend = "cpu-f64-selected-bf16-rows"
			entry.Result = &result
		}
		entry.Elapsed = time.Since(start).Seconds()
		if err = enc.Encode(entry); err != nil {
			return err
		}
		if err = out.Sync(); err != nil {
			return err
		}
		count++
		fmt.Fprintf(stderr, "row=%d id=%s variant=%s seconds=%.3f rejected=%v\n", count, row.ID, row.Variant, entry.Elapsed, entry.Error != "")
	}
	if err = scan.Err(); err != nil {
		return err
	}
	return writeJSON(stdout, map[string]any{"rows": count, "output": *output, "model_id": modelID})
}
