package jevlike

import (
	"math"
	"testing"
)

func TestTinyScorerBackwardFiniteDifferenceEmbeddingsAndPositions(t *testing.T) {
	model, err := NewTinyScorer(Config{Width: 3, Rank: 2, ContextTokens: 4, OptionTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	for i := range model.EmbeddingWeight {
		model.EmbeddingWeight[i] = float32((i%17)-8) * 0.03
	}
	for i := 0; i < model.Config.Width; i++ {
		model.EmbeddingWeight[i] = 0
	}
	for i := range model.PositionWeight {
		model.PositionWeight[i] = float32((i%11)-5) * 0.02
	}
	copy(model.Head.ContextNormWeight, []float32{1.1, -0.9, 0.7})
	copy(model.Head.ContextNormBias, []float32{0.05, -0.03, 0.02})
	copy(model.Head.OptionNormWeight, []float32{0.8, 1.2, -1.1})
	copy(model.Head.OptionNormBias, []float32{0.04, -0.02, 0.01})
	copy(model.Head.QueryWeight, []float32{0.3, -0.2, 0.5, -0.4, 0.6, 0.1})
	copy(model.Head.KeyWeight, []float32{-0.5, 0.7, 0.2, 0.3, -0.1, 0.4})
	copy(model.Head.ValueWeight, []float32{0.6, -0.3, 0.2, -0.2, 0.5, -0.4})

	batch, err := BuildByteBatch([]ChoiceExample{{
		Context: "wxyz",
		Options: []string{"wx", "xz"},
		Label:   0,
	}}, model.Config.ContextTokens, model.Config.OptionTokens)
	if err != nil {
		t.Fatal(err)
	}
	dLogits := [][]float32{{0.6, -0.4}}
	grads, err := model.Backward(batch, dLogits)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := grads[embeddingParamName]; !ok {
		t.Fatalf("missing parameter gradient %q", embeddingParamName)
	}
	if _, ok := grads[positionParamName]; !ok {
		t.Fatalf("missing parameter gradient %q", positionParamName)
	}
	for dim, value := range grads[embeddingParamName][:model.Config.Width] {
		if value != 0 {
			t.Fatalf("padding embedding gradient dim=%d got=%g want 0", dim, value)
		}
	}

	eval := func() (float64, error) {
		return tinyObjective(model, batch, dLogits)
	}
	step := float32(1e-3)
	usedIDs := usedTokenIDs(batch)
	for _, id := range usedIDs {
		for dim := 0; dim < model.Config.Width; dim++ {
			index := id*model.Config.Width + dim
			numeric := centralDifference(t, &model.EmbeddingWeight[index], step, eval)
			requireGradientClose(t, embeddingParamName+indexLabel(id)+indexLabel(dim), grads[embeddingParamName][index], numeric)
		}
	}
	for position := 0; position < len(batch.ContextIDs[0]); position++ {
		for dim := 0; dim < model.Config.Width; dim++ {
			index := position*model.Config.Width + dim
			numeric := centralDifference(t, &model.PositionWeight[index], step, eval)
			requireGradientClose(t, positionParamName+indexLabel(position)+indexLabel(dim), grads[positionParamName][index], numeric)
		}
	}
	if math.Abs(float64(grads[positionParamName][0])) < 1e-5 {
		t.Fatal("position gradients are unexpectedly all zero")
	}
}

func TestTrainTinyScorerReducesSyntheticLossAndRestoresBestState(t *testing.T) {
	train := syntheticExamples(128, 11)
	validation := syntheticExamples(64, 11+int64(len(train))*syntheticStride)
	model, err := NewTinyScorer(Config{Width: 16, Rank: 16, ContextTokens: 128, OptionTokens: 32})
	if err != nil {
		t.Fatal(err)
	}
	initialLoss, err := meanCrossEntropyLoss(model, validation, 16)
	if err != nil {
		t.Fatal(err)
	}
	result, err := TrainTinyScorer(model, train, validation, TrainConfig{
		Epochs:       20,
		BatchSize:    16,
		LearningRate: 2e-3,
		Seed:         7,
		MaxGradNorm:  1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.History) != 20 {
		t.Fatalf("history len=%d want 20", len(result.History))
	}
	if result.BestState == nil {
		t.Fatal("expected best state")
	}
	finalLoss, err := meanCrossEntropyLoss(model, validation, 16)
	if err != nil {
		t.Fatal(err)
	}
	if finalLoss >= initialLoss-0.2 {
		t.Fatalf("validation loss did not improve enough: initial=%g final=%g", initialLoss, finalLoss)
	}
	if math.Abs(finalLoss-result.BestValidationNLL) > 1e-6 {
		t.Fatalf("model was not restored to best state: final=%g best=%g", finalLoss, result.BestValidationNLL)
	}
	current := model.NamedParameterMap()
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

func tinyObjective(model *TinyScorer, batch ByteBatch, dLogits [][]float32) (float64, error) {
	logits, err := model.Forward(batch)
	if err != nil {
		return 0, err
	}
	var total float64
	for row := range logits {
		for option := range logits[row] {
			if !batch.OptionMask[row][option] {
				continue
			}
			total += float64(logits[row][option]) * float64(dLogits[row][option])
		}
	}
	return total, nil
}

func syntheticExamples(count int, seed int64) []ChoiceExample {
	examples := make([]ChoiceExample, count)
	for i := range examples {
		examples[i] = SyntheticExample(seed + int64(i)*syntheticStride)
	}
	return examples
}

func usedTokenIDs(batch ByteBatch) []int {
	seen := make(map[int]struct{})
	var ids []int
	for row := range batch.ContextIDs {
		for _, id := range batch.ContextIDs[row] {
			value := int(id)
			if value == 0 {
				continue
			}
			if _, ok := seen[value]; ok {
				continue
			}
			seen[value] = struct{}{}
			ids = append(ids, value)
		}
		for option := range batch.OptionIDs[row] {
			for token := range batch.OptionIDs[row][option] {
				if !batch.OptionTokenMask[row][option][token] {
					continue
				}
				value := int(batch.OptionIDs[row][option][token])
				if value == 0 {
					continue
				}
				if _, ok := seen[value]; ok {
					continue
				}
				seen[value] = struct{}{}
				ids = append(ids, value)
			}
		}
	}
	return ids
}

func TestEmbeddingInitialisationMatchesUpstreamDistribution(t *testing.T) {
	cfg := Config{Width: 64, Rank: 64, ContextTokens: 192, OptionTokens: 32}
	m, err := NewInitializedTinyScorer(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range m.EmbeddingWeight[:cfg.Width] {
		if v != 0 {
			t.Fatal("padding embedding must be zero")
		}
	}
	for name, x := range map[string][]float32{"embedding": m.EmbeddingWeight[cfg.Width:], "position": m.PositionWeight} {
		var sum, sum2 float64
		for _, v := range x {
			sum += float64(v)
			sum2 += float64(v) * float64(v)
		}
		mean := sum / float64(len(x))
		variance := sum2/float64(len(x)) - mean*mean
		if math.Abs(mean) > .05 || variance < .9 || variance > 1.1 {
			t.Fatalf("%s mean=%g variance=%g want normal(0,1)", name, mean, variance)
		}
	}
}
