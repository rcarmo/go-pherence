package jevlike

import (
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

type aliasEncoder struct {
	rows map[byte][]float32
}

func newAliasEncoder(alphabet string) *aliasEncoder {
	rows := make(map[byte][]float32, len(alphabet)+1)
	for i := 0; i < len(alphabet); i++ {
		b := alphabet[i]
		row := make([]float32, len(alphabet))
		row[i] = 1
		rows[b] = row
	}
	rows['_'] = []float32{1, 0, 0, 0}
	return &aliasEncoder{rows: rows}
}

func (e *aliasEncoder) Encode(text string, maxTokens int) ([][]float32, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("max tokens must be positive")
	}
	ids := []byte(text)
	if len(ids) > maxTokens {
		ids = ids[:maxTokens]
	}
	if len(ids) == 0 {
		return [][]float32{e.rows['_']}, nil
	}
	out := make([][]float32, len(ids))
	for i, b := range ids {
		row, ok := e.rows[b]
		if !ok {
			return nil, fmt.Errorf("unknown token %q at %d", b, i)
		}
		out[i] = row
	}
	return out, nil
}

func TestTrainFrozenScorerReducesLossAndRestoresBestState(t *testing.T) {
	train := frozenToyExamples(128, 0)
	validation := frozenToyExamples(64, 1)
	model := newFrozenTestScorer(t, newAliasEncoder("abcd"))

	initialLoss, err := meanCrossEntropyLossFrozen(model, validation, 16)
	if err != nil {
		t.Fatal(err)
	}
	result, err := TrainFrozenScorer(model, train, validation, TrainConfig{
		Epochs:       24,
		BatchSize:    16,
		LearningRate: 1e-2,
		Seed:         7,
		MaxGradNorm:  1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.History) != 24 {
		t.Fatalf("history len=%d want 24", len(result.History))
	}
	if result.BestState == nil {
		t.Fatal("expected best state")
	}
	finalLoss, err := meanCrossEntropyLossFrozen(model, validation, 16)
	if err != nil {
		t.Fatal(err)
	}
	if finalLoss >= initialLoss-0.2 {
		t.Fatalf("validation loss did not improve enough: initial=%g final=%g", initialLoss, finalLoss)
	}
	if math.Abs(finalLoss-result.BestValidationNLL) > 1e-6 {
		t.Fatalf("model was not restored to best state: final=%g best=%g", finalLoss, result.BestValidationNLL)
	}
	current := model.Head.NamedParameterMap(defaultHeadPrefix)
	if len(current) != len(result.BestState) {
		t.Fatalf("state size=%d want %d", len(current), len(result.BestState))
	}
	for name, want := range result.BestState {
		got, ok := current[name]
		if !ok {
			t.Fatalf("missing parameter %q in current state", name)
		}
		if len(got) != len(want) {
			t.Fatalf("parameter %q len=%d want=%d", name, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("parameter %q[%d]=%g want %g", name, i, got[i], want[i])
			}
		}
	}
}

func TestTrainFrozenScorerDoesNotMutateEncoderState(t *testing.T) {
	encoder := newAliasEncoder("abcd")
	snapshot := cloneAliasRows(encoder.rows)
	model := newFrozenTestScorer(t, encoder)
	before := model.Head.NamedParameterMap(defaultHeadPrefix)

	if _, err := TrainFrozenScorer(model, frozenToyExamples(64, 0), frozenToyExamples(32, 2), TrainConfig{
		Epochs:       12,
		BatchSize:    8,
		LearningRate: 1e-2,
		Seed:         7,
		MaxGradNorm:  1,
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(encoder.rows, snapshot) {
		t.Fatalf("encoder state mutated: got=%v want=%v", encoder.rows, snapshot)
	}
	after := model.Head.NamedParameterMap(defaultHeadPrefix)
	if reflect.DeepEqual(before, after) {
		t.Fatal("head parameters did not change")
	}
}

func TestFrozenCheckpointRoundTripAndEvaluate(t *testing.T) {
	train := frozenToyExamples(128, 0)
	validation := frozenToyExamples(32, 1)
	model := newFrozenTestScorer(t, newAliasEncoder("abcd"))
	if _, err := TrainFrozenScorer(model, train, validation, TrainConfig{
		Epochs:       24,
		BatchSize:    16,
		LearningRate: 1e-2,
		Seed:         7,
		MaxGradNorm:  1,
	}); err != nil {
		t.Fatal(err)
	}

	eval, err := EvaluateFrozen(model, validation[:8], 2)
	if err != nil {
		t.Fatal(err)
	}
	if eval.Model.Examples != 8 || eval.ShuffledContext.Examples != 8 {
		t.Fatalf("evaluation example counts %+v", eval)
	}
	if eval.Model.Top1 < 0.99 {
		t.Fatalf("expected high frozen accuracy, got %+v", eval.Model)
	}
	if eval.ShuffledContext.Top1 >= eval.Model.Top1 {
		t.Fatalf("expected shuffled control to be worse, got %+v", eval)
	}

	checkpoint, err := FrozenCheckpoint(model)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "nested", "frozen.json")
	if err := SaveCheckpoint(path, checkpoint); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := loaded.Frozen(newAliasEncoder("abcd"))
	if err != nil {
		t.Fatal(err)
	}
	wantLogits, err := model.Forward(validation[:8], false)
	if err != nil {
		t.Fatal(err)
	}
	gotLogits, err := restored.Forward(validation[:8], false)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotLogits, wantLogits) {
		t.Fatalf("restored logits mismatch: got=%v want=%v", gotLogits, wantLogits)
	}
	restoredEval, err := EvaluateFrozen(restored, validation[:8], 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restoredEval, eval) {
		t.Fatalf("restored evaluation mismatch: got=%+v want=%+v", restoredEval, eval)
	}
}

func newFrozenTestScorer(t *testing.T, encoder TokenEncoder) *FrozenScorer {
	t.Helper()
	head, err := NewAttentionHead(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	return &FrozenScorer{
		Config:    Config{Width: 4, Rank: 4, ContextTokens: 1, OptionTokens: 1},
		Reference: "test",
		Head:      *head,
		Encoder:   encoder,
	}
}

func frozenToyExamples(count, offset int) []ChoiceExample {
	alphabet := []string{"a", "b", "c", "d"}
	examples := make([]ChoiceExample, count)
	for i := range examples {
		correct := alphabet[(i+offset)%len(alphabet)]
		wrong := alphabet[(i+offset+1)%len(alphabet)]
		options := []string{correct, wrong}
		label := 0
		if i%2 == 1 {
			options[0], options[1] = options[1], options[0]
			label = 1
		}
		examples[i] = ChoiceExample{Context: correct, Options: options, Label: label}
	}
	return examples
}

func cloneAliasRows(rows map[byte][]float32) map[byte][]float32 {
	out := make(map[byte][]float32, len(rows))
	for k, v := range rows {
		out[k] = append([]float32(nil), v...)
	}
	return out
}
