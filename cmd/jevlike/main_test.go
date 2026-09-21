package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

type syntheticReport struct {
	OutputDir string             `json:"output_dir"`
	Seed      int64              `json:"seed"`
	Sizes     jevlike.SplitSizes `json:"sizes"`
}

type trainReport struct {
	Train             string                 `json:"train"`
	Validation        string                 `json:"validation"`
	Checkpoint        string                 `json:"checkpoint"`
	Config            jevlike.Config         `json:"config"`
	History           []jevlike.EpochMetrics `json:"history"`
	BestValidationNLL float64                `json:"best_validation_nll"`
}

type predictReport struct {
	Options       []string  `json:"options"`
	Logits        []float32 `json:"logits"`
	Probabilities []float32 `json:"probabilities"`
	BestIndex     int       `json:"best_index"`
	BestOption    string    `json:"best_option"`
}

func TestRunSyntheticTrainPredictEval(t *testing.T) {
	root := t.TempDir()
	dataDir := filepath.Join(root, "data")
	checkpointPath := filepath.Join(root, "model.json")

	var stdout, stderr bytes.Buffer
	if err := run([]string{
		"synthetic",
		"-output-dir", dataDir,
		"-train", "8",
		"-validation", "4",
		"-test", "2",
		"-seed", "11",
	}, &stdout, &stderr); err != nil {
		t.Fatalf("synthetic error = %v\nstderr=%s", err, stderr.String())
	}
	var synthetic syntheticReport
	if err := json.Unmarshal(stdout.Bytes(), &synthetic); err != nil {
		t.Fatalf("synthetic json error = %v\nstdout=%s", err, stdout.String())
	}
	if synthetic.OutputDir != dataDir || synthetic.Seed != 11 {
		t.Fatalf("synthetic report = %+v", synthetic)
	}
	if synthetic.Sizes != (jevlike.SplitSizes{Train: 8, Validation: 4, Test: 2}) {
		t.Fatalf("synthetic sizes = %+v", synthetic.Sizes)
	}

	stdout.Reset()
	stderr.Reset()
	if err := run([]string{
		"train",
		"-train", filepath.Join(dataDir, "train.jsonl"),
		"-validation", filepath.Join(dataDir, "validation.jsonl"),
		"-checkpoint", checkpointPath,
		"-epochs", "1",
		"-batch-size", "4",
	}, &stdout, &stderr); err != nil {
		t.Fatalf("train error = %v\nstderr=%s", err, stderr.String())
	}
	var trained trainReport
	if err := json.Unmarshal(stdout.Bytes(), &trained); err != nil {
		t.Fatalf("train json error = %v\nstdout=%s", err, stdout.String())
	}
	if trained.Checkpoint != checkpointPath {
		t.Fatalf("checkpoint path = %q want %q", trained.Checkpoint, checkpointPath)
	}
	if len(trained.History) != 1 {
		t.Fatalf("history len = %d want 1", len(trained.History))
	}
	if trained.Config != (jevlike.Config{Width: defaultTinyWidth, Rank: defaultTinyRank, ContextTokens: defaultTinyContextTokens, OptionTokens: defaultTinyOptionTokens}) {
		t.Fatalf("train config = %+v", trained.Config)
	}
	if _, err := os.Stat(checkpointPath); err != nil {
		t.Fatalf("checkpoint stat error = %v", err)
	}
	checkpoint, err := jevlike.LoadCheckpoint(checkpointPath)
	if err != nil {
		t.Fatalf("LoadCheckpoint() error = %v", err)
	}
	if checkpoint.Encoder != "tiny" {
		t.Fatalf("checkpoint encoder = %q", checkpoint.Encoder)
	}

	testExamples, err := jevlike.LoadJSONL(filepath.Join(dataDir, "test.jsonl"))
	if err != nil {
		t.Fatalf("LoadJSONL(test) error = %v", err)
	}
	if len(testExamples) == 0 {
		t.Fatal("expected synthetic test examples")
	}
	example := testExamples[0]

	stdout.Reset()
	stderr.Reset()
	predictArgs := []string{"predict", "-checkpoint", checkpointPath, "-context", example.Context}
	for _, option := range example.Options {
		predictArgs = append(predictArgs, "-option", option)
	}
	if err := run(predictArgs, &stdout, &stderr); err != nil {
		t.Fatalf("predict error = %v\nstderr=%s", err, stderr.String())
	}
	var predicted predictReport
	if err := json.Unmarshal(stdout.Bytes(), &predicted); err != nil {
		t.Fatalf("predict json error = %v\nstdout=%s", err, stdout.String())
	}
	if len(predicted.Options) != len(example.Options) || len(predicted.Probabilities) != len(example.Options) || len(predicted.Logits) != len(example.Options) {
		t.Fatalf("predict shape mismatch = %+v want %d options", predicted, len(example.Options))
	}
	if predicted.BestIndex < 0 || predicted.BestIndex >= len(predicted.Options) {
		t.Fatalf("best index = %d for %d options", predicted.BestIndex, len(predicted.Options))
	}
	if predicted.BestOption != predicted.Options[predicted.BestIndex] {
		t.Fatalf("best option mismatch: %q vs %q", predicted.BestOption, predicted.Options[predicted.BestIndex])
	}
	var sum float64
	for _, probability := range predicted.Probabilities {
		sum += float64(probability)
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Fatalf("probability sum = %g", sum)
	}

	stdout.Reset()
	stderr.Reset()
	if err := run([]string{
		"eval",
		"-checkpoint", checkpointPath,
		"-data", filepath.Join(dataDir, "test.jsonl"),
		"-batch-size", "2",
	}, &stdout, &stderr); err != nil {
		t.Fatalf("eval error = %v\nstderr=%s", err, stderr.String())
	}
	var evaluation jevlike.Evaluation
	if err := json.Unmarshal(stdout.Bytes(), &evaluation); err != nil {
		t.Fatalf("eval json error = %v\nstdout=%s", err, stdout.String())
	}
	if evaluation.Model.Examples != len(testExamples) || evaluation.ShuffledContext.Examples != len(testExamples) {
		t.Fatalf("evaluation examples = %+v want %d", evaluation, len(testExamples))
	}
}

func TestRunWikispeedia(t *testing.T) {
	root := t.TempDir()
	graphDir := filepath.Join(root, "wikispeedia_paths-and-graph")
	articlesDir := filepath.Join(root, "plaintext_articles")
	if err := os.MkdirAll(graphDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(articlesDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(graphDir, "links.tsv"), []byte(strings.Join([]string{
		"# source\ttarget",
		"Start\tTarget_Article",
		"Start\tOther_Article",
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(graphDir, "paths_finished.tsv"), []byte(strings.Join([]string{
		"# header",
		"user\tsession\t0\tStart;Target_Article",
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(articlesDir, "Start.txt"), []byte("alpha beta"), 0o644); err != nil {
		t.Fatal(err)
	}

	outputDir := filepath.Join(root, "out")
	var stdout, stderr bytes.Buffer
	if err := run([]string{
		"wikispeedia",
		"-root-dir", root,
		"-output-dir", outputDir,
		"-max-options", "2",
	}, &stdout, &stderr); err != nil {
		t.Fatalf("wikispeedia error = %v\nstderr=%s", err, stderr.String())
	}
	var report struct {
		Sizes jevlike.SplitSizes `json:"sizes"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("wikispeedia json error = %v\nstdout=%s", err, stdout.String())
	}
	if report.Sizes.Train+report.Sizes.Validation+report.Sizes.Test != 1 {
		t.Fatalf("unexpected counts = %+v", report.Sizes)
	}
	for _, name := range []string{"train.jsonl", "validation.jsonl", "test.jsonl"} {
		if _, err := os.Stat(filepath.Join(outputDir, name)); err != nil {
			t.Fatalf("missing %s: %v", name, err)
		}
	}
}

func TestRunErrors(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if err := run(nil, &stdout, &stderr); err == nil || !strings.Contains(err.Error(), "expected subcommand") {
		t.Fatalf("no-subcommand error = %v", err)
	}
	stdout.Reset()
	stderr.Reset()
	if err := run([]string{"unknown"}, &stdout, &stderr); err == nil || !strings.Contains(err.Error(), "unknown subcommand") {
		t.Fatalf("unknown-subcommand error = %v", err)
	}
	stdout.Reset()
	stderr.Reset()
	if err := run([]string{"train", "-train", "a.jsonl", "-validation", "b.jsonl"}, &stdout, &stderr); err == nil || !strings.Contains(err.Error(), "-train, -validation and -checkpoint are required") {
		t.Fatalf("train missing flags error = %v", err)
	}
	stdout.Reset()
	stderr.Reset()
	if err := run([]string{"predict", "-checkpoint", "x.json", "-context", "ctx", "-option", "only-one"}, &stdout, &stderr); err == nil || !strings.Contains(err.Error(), "at least two -option flags are required") {
		t.Fatalf("predict option error = %v", err)
	}
}
