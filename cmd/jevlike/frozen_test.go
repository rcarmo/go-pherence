package main

import (
	"bytes"
	"encoding/json"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

type frozenAliasEncoder struct {
	rows map[byte][]float32
}

func newFrozenAliasEncoder(alphabet string) *frozenAliasEncoder {
	rows := make(map[byte][]float32, len(alphabet)+1)
	for i := 0; i < len(alphabet); i++ {
		b := alphabet[i]
		row := make([]float32, len(alphabet))
		row[i] = 1
		rows[b] = row
	}
	rows['_'] = []float32{1, 0, 0, 0}
	return &frozenAliasEncoder{rows: rows}
}

func (e *frozenAliasEncoder) Encode(text string, maxTokens int) ([][]float32, error) {
	if maxTokens <= 0 {
		return nil, nil
	}
	ids := []byte(text)
	if len(ids) > maxTokens {
		ids = ids[:maxTokens]
	}
	if len(ids) == 0 {
		return [][]float32{append([]float32(nil), e.rows['_']...)}, nil
	}
	out := make([][]float32, len(ids))
	for i, b := range ids {
		row, ok := e.rows[b]
		if !ok {
			return nil, nil
		}
		out[i] = append([]float32(nil), row...)
	}
	return out, nil
}

func TestRunFrozenTrainPredictEval(t *testing.T) {
	oldLoader := frozenEncoderLoader
	defer func() { frozenEncoderLoader = oldLoader }()

	var gotDirs []string
	var gotBOS []int
	frozenEncoderLoader = func(dir string, bos int) (jevlike.TokenEncoder, int, string, error) {
		gotDirs = append(gotDirs, dir)
		gotBOS = append(gotBOS, bos)
		return newFrozenAliasEncoder("abcd"), 4, filepath.Clean(dir), nil
	}

	root := t.TempDir()
	trainPath := filepath.Join(root, "train.jsonl")
	validationPath := filepath.Join(root, "validation.jsonl")
	testPath := filepath.Join(root, "test.jsonl")
	checkpointPath := filepath.Join(root, "frozen.json")
	if err := jevlike.WriteJSONL(trainPath, frozenCLIExamples(16, 0)); err != nil {
		t.Fatal(err)
	}
	if err := jevlike.WriteJSONL(validationPath, frozenCLIExamples(8, 1)); err != nil {
		t.Fatal(err)
	}
	testExamples := frozenCLIExamples(4, 2)
	if err := jevlike.WriteJSONL(testPath, testExamples); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if err := runFrozenTrain([]string{
		"--encoder-model", filepath.Join(root, "encoder"),
		"--encoder-bos", "99",
		"--train", trainPath,
		"--validation", validationPath,
		"--checkpoint", checkpointPath,
		"--epochs", "1",
		"--batch-size", "4",
	}, &stdout, &stderr); err != nil {
		t.Fatalf("runFrozenTrain error = %v\nstderr=%s", err, stderr.String())
	}
	var trained struct {
		EncoderModel      string                 `json:"encoder_model"`
		Train             string                 `json:"train"`
		Validation        string                 `json:"validation"`
		Checkpoint        string                 `json:"checkpoint"`
		Config            jevlike.Config         `json:"config"`
		History           []jevlike.EpochMetrics `json:"history"`
		BestValidationNLL float64                `json:"best_validation_nll"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &trained); err != nil {
		t.Fatalf("train json error = %v\nstdout=%s", err, stdout.String())
	}
	if trained.Checkpoint != checkpointPath {
		t.Fatalf("checkpoint = %q want %q", trained.Checkpoint, checkpointPath)
	}
	if trained.Config != (jevlike.Config{Width: 4, Rank: defaultFrozenRank, ContextTokens: defaultFrozenContextTokens, OptionTokens: defaultFrozenOptionTokens}) {
		t.Fatalf("config = %+v", trained.Config)
	}
	if len(trained.History) != 1 {
		t.Fatalf("history len = %d want 1", len(trained.History))
	}
	checkpoint, err := jevlike.LoadCheckpoint(checkpointPath)
	if err != nil {
		t.Fatal(err)
	}
	if checkpoint.Encoder != "frozen" {
		t.Fatalf("checkpoint encoder = %q want frozen", checkpoint.Encoder)
	}
	if checkpoint.EncoderReference != filepath.Join(root, "encoder") {
		t.Fatalf("encoder reference = %q want %q", checkpoint.EncoderReference, filepath.Join(root, "encoder"))
	}

	stdout.Reset()
	stderr.Reset()
	predictArgs := []string{
		"--encoder-model", filepath.Join(root, "encoder"),
		"--checkpoint", checkpointPath,
		"--context", testExamples[0].Context,
	}
	for _, option := range testExamples[0].Options {
		predictArgs = append(predictArgs, "--option", option)
	}
	if err := runFrozenPredict(predictArgs, &stdout, &stderr); err != nil {
		t.Fatalf("runFrozenPredict error = %v\nstderr=%s", err, stderr.String())
	}
	var predicted struct {
		Options       []string  `json:"options"`
		Logits        []float32 `json:"logits"`
		Probabilities []float32 `json:"probabilities"`
		BestIndex     int       `json:"best_index"`
		BestOption    string    `json:"best_option"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &predicted); err != nil {
		t.Fatalf("predict json error = %v\nstdout=%s", err, stdout.String())
	}
	if len(predicted.Options) != len(testExamples[0].Options) || len(predicted.Logits) != len(testExamples[0].Options) || len(predicted.Probabilities) != len(testExamples[0].Options) {
		t.Fatalf("predict shape mismatch: %+v", predicted)
	}
	if predicted.BestIndex < 0 || predicted.BestIndex >= len(predicted.Options) {
		t.Fatalf("best index = %d", predicted.BestIndex)
	}
	if predicted.BestOption != predicted.Options[predicted.BestIndex] {
		t.Fatalf("best option = %q want %q", predicted.BestOption, predicted.Options[predicted.BestIndex])
	}
	var probabilitySum float64
	for _, p := range predicted.Probabilities {
		probabilitySum += float64(p)
	}
	if math.Abs(probabilitySum-1) > 1e-4 {
		t.Fatalf("probability sum = %g", probabilitySum)
	}

	stdout.Reset()
	stderr.Reset()
	if err := runFrozenEval([]string{
		"--encoder-model", filepath.Join(root, "encoder"),
		"--checkpoint", checkpointPath,
		"--data", testPath,
		"--batch-size", "2",
	}, &stdout, &stderr); err != nil {
		t.Fatalf("runFrozenEval error = %v\nstderr=%s", err, stderr.String())
	}
	var evaluation jevlike.Evaluation
	if err := json.Unmarshal(stdout.Bytes(), &evaluation); err != nil {
		t.Fatalf("eval json error = %v\nstdout=%s", err, stdout.String())
	}
	if evaluation.Model.Examples != len(testExamples) || evaluation.ShuffledContext.Examples != len(testExamples) {
		t.Fatalf("evaluation examples = %+v want %d", evaluation, len(testExamples))
	}

	wantDirs := []string{filepath.Join(root, "encoder"), filepath.Join(root, "encoder"), filepath.Join(root, "encoder")}
	if len(gotDirs) != len(wantDirs) {
		t.Fatalf("loader calls = %d want %d", len(gotDirs), len(wantDirs))
	}
	for i := range wantDirs {
		if gotDirs[i] != wantDirs[i] {
			t.Fatalf("loader dir[%d] = %q want %q", i, gotDirs[i], wantDirs[i])
		}
	}
	if len(gotBOS) != 3 || gotBOS[0] != 99 || gotBOS[1] != -1 || gotBOS[2] != -1 {
		t.Fatalf("loader bos calls = %v want [99 -1 -1]", gotBOS)
	}
}

func TestRunFrozenPredictRejectsWidthMismatch(t *testing.T) {
	oldLoader := frozenEncoderLoader
	defer func() { frozenEncoderLoader = oldLoader }()
	frozenEncoderLoader = func(dir string, bos int) (jevlike.TokenEncoder, int, string, error) {
		return newFrozenAliasEncoder("abcd"), 4, filepath.Clean(dir), nil
	}

	root := t.TempDir()
	checkpointPath := filepath.Join(root, "bad-width.json")
	head, err := jevlike.NewAttentionHead(5, defaultFrozenRank)
	if err != nil {
		t.Fatal(err)
	}
	model := &jevlike.FrozenScorer{
		Config:    jevlike.Config{Width: 5, Rank: defaultFrozenRank, ContextTokens: defaultFrozenContextTokens, OptionTokens: defaultFrozenOptionTokens},
		Reference: filepath.Join(root, "encoder"),
		Head:      *head,
	}
	if err := jevlike.InitializeFrozenScorer(model, 7); err != nil {
		t.Fatal(err)
	}
	checkpoint, err := jevlike.FrozenCheckpoint(model)
	if err != nil {
		t.Fatal(err)
	}
	if err := jevlike.SaveCheckpoint(checkpointPath, checkpoint); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	err = runFrozenPredict([]string{
		"--encoder-model", filepath.Join(root, "encoder"),
		"--checkpoint", checkpointPath,
		"--context", "a",
		"--option", "a",
		"--option", "b",
	}, &stdout, &stderr)
	if err == nil || !strings.Contains(err.Error(), "encoder width 4 does not match checkpoint width 5") {
		t.Fatalf("width mismatch error = %v", err)
	}
}

func frozenCLIExamples(count, offset int) []jevlike.ChoiceExample {
	alphabet := []string{"a", "b", "c", "d"}
	examples := make([]jevlike.ChoiceExample, count)
	for i := range examples {
		correct := alphabet[(i+offset)%len(alphabet)]
		wrong := alphabet[(i+offset+1)%len(alphabet)]
		options := []string{correct, wrong}
		label := 0
		if i%2 == 1 {
			options[0], options[1] = options[1], options[0]
			label = 1
		}
		examples[i] = jevlike.ChoiceExample{Context: correct, Options: options, Label: label}
	}
	return examples
}
