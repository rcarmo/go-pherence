package jevlike

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type countingFrozenEncoder struct {
	inner TokenEncoder
	calls int
}

func (e *countingFrozenEncoder) Encode(text string, maxTokens int) ([][]float32, error) {
	e.calls++
	return e.inner.Encode(text, maxTokens)
}

func TestTrainFrozenResumableMatchesUninterrupted(t *testing.T) {
	train := frozenToyExamples(23, 0)
	validation := frozenToyExamples(11, 1)
	cfg := TrainConfig{Epochs: 5, BatchSize: 4, LearningRate: 1e-2, Seed: 7, MaxGradNorm: 1}
	identity := frozenResumeTestIdentity(t, train, validation)

	baselineModel := newFrozenTestScorer(t, newAliasEncoder("abcd"))
	baselineResult, finished, err := TrainFrozenResumable(baselineModel, train, validation, cfg, identity, filepath.Join(t.TempDir(), "baseline.json"), 0)
	if err != nil {
		t.Fatal(err)
	}
	if !finished {
		t.Fatal("uninterrupted training did not finish")
	}

	statePath := filepath.Join(t.TempDir(), "resume.json")
	var resumedModel *FrozenScorer
	var resumedResult TrainResult
	sawInterrupt := false
	for attempt := 0; attempt < 20; attempt++ {
		resumedModel = newFrozenTestScorer(t, newAliasEncoder("abcd"))
		resumedResult, finished, err = TrainFrozenResumable(resumedModel, train, validation, cfg, identity, statePath, 2)
		if err != nil {
			t.Fatal(err)
		}
		if finished {
			break
		}
		sawInterrupt = true
	}
	if !sawInterrupt {
		t.Fatal("expected at least one interruption")
	}
	if !finished {
		t.Fatal("resumed training did not finish")
	}
	if !reflect.DeepEqual(resumedResult, baselineResult) {
		t.Fatalf("resumed result mismatch:\n got=%+v\nwant=%+v", resumedResult, baselineResult)
	}
	if !reflect.DeepEqual(resumedModel.Head.NamedParameterMap(defaultHeadPrefix), baselineModel.Head.NamedParameterMap(defaultHeadPrefix)) {
		t.Fatal("final parameters differ after resume")
	}
}

func TestTrainFrozenResumableRejectsChangedResumeInputs(t *testing.T) {
	train := frozenToyExamples(17, 0)
	validation := frozenToyExamples(9, 1)
	cfg := TrainConfig{Epochs: 4, BatchSize: 3, LearningRate: 1e-2, Seed: 7, MaxGradNorm: 1}
	identity := frozenResumeTestIdentity(t, train, validation)
	statePath := filepath.Join(t.TempDir(), "resume.json")

	if _, finished, err := TrainFrozenResumable(newFrozenTestScorer(t, newAliasEncoder("abcd")), train, validation, cfg, identity, statePath, 1); err != nil {
		t.Fatal(err)
	} else if finished {
		t.Fatal("expected interrupted run")
	}

	t.Run("reference", func(t *testing.T) {
		model := newFrozenTestScorer(t, newAliasEncoder("abcd"))
		model.Reference = "other"
		if _, _, err := TrainFrozenResumable(model, train, validation, cfg, identity, statePath, 1); err == nil {
			t.Fatal("expected reference mismatch")
		}
	})

	t.Run("cache", func(t *testing.T) {
		changed := identity
		changed.CacheID = "cache-v2"
		if _, _, err := TrainFrozenResumable(newFrozenTestScorer(t, newAliasEncoder("abcd")), train, validation, cfg, changed, statePath, 1); err == nil {
			t.Fatal("expected cache mismatch")
		}
	})

	t.Run("data", func(t *testing.T) {
		changedValidation := frozenResumeCloneExamples(validation)
		if changedValidation[0].Context == "a" {
			changedValidation[0].Context = "b"
		} else {
			changedValidation[0].Context = "a"
		}
		changed := frozenResumeTestIdentity(t, train, changedValidation)
		if _, _, err := TrainFrozenResumable(newFrozenTestScorer(t, newAliasEncoder("abcd")), train, changedValidation, cfg, changed, statePath, 1); err == nil {
			t.Fatal("expected data mismatch")
		}
	})

	t.Run("hparams", func(t *testing.T) {
		changed := cfg
		changed.LearningRate *= 0.5
		if _, _, err := TrainFrozenResumable(newFrozenTestScorer(t, newAliasEncoder("abcd")), train, validation, changed, identity, statePath, 1); err == nil {
			t.Fatal("expected hyperparameter mismatch")
		}
	})
}

func TestTrainFrozenResumableUsesSuppliedEncoderOnResume(t *testing.T) {
	train := frozenToyExamples(17, 0)
	validation := frozenToyExamples(9, 1)
	cfg := TrainConfig{Epochs: 4, BatchSize: 3, LearningRate: 1e-2, Seed: 7, MaxGradNorm: 1}
	identity := frozenResumeTestIdentity(t, train, validation)
	statePath := filepath.Join(t.TempDir(), "resume.json")

	if _, finished, err := TrainFrozenResumable(newFrozenTestScorer(t, newAliasEncoder("abcd")), train, validation, cfg, identity, statePath, 1); err != nil {
		t.Fatal(err)
	} else if finished {
		t.Fatal("expected interrupted run")
	}

	encoder := &countingFrozenEncoder{inner: newAliasEncoder("abcd")}
	if _, finished, err := TrainFrozenResumable(newFrozenTestScorer(t, encoder), train, validation, cfg, identity, statePath, 1); err != nil {
		t.Fatal(err)
	} else if finished {
		t.Fatal("unexpected completion")
	}
	if encoder.calls == 0 {
		t.Fatal("supplied resume encoder was not used")
	}
}

func frozenResumeTestIdentity(t *testing.T, train, validation []ChoiceExample) FrozenRunIdentity {
	t.Helper()
	trainHash, err := ChoiceExamplesSHA256(train)
	if err != nil {
		t.Fatal(err)
	}
	validationHash, err := ChoiceExamplesSHA256(validation)
	if err != nil {
		t.Fatal(err)
	}
	return FrozenRunIdentity{
		CacheID:          "cache-v1",
		TrainSHA256:      trainHash,
		ValidationSHA256: validationHash,
		CodeRevision:     "rev-a",
	}
}

func frozenResumeCloneExamples(examples []ChoiceExample) []ChoiceExample {
	out := make([]ChoiceExample, len(examples))
	for i, example := range examples {
		out[i] = ChoiceExample{Context: example.Context, Options: append([]string(nil), example.Options...), Label: example.Label}
	}
	return out
}

func TestFrozenResumeRejectsCorruptOptimizerAndCursor(t *testing.T) {
	// Use the existing fixture and alter one field in a valid interrupted file.
	for _, kind := range []string{"negative-variance", "cursor", "unknown", "trailing"} {
		t.Run(kind, func(t *testing.T) {
			train := frozenToyExamples(9, 0)
			validation := frozenToyExamples(4, 1)
			cfg := TrainConfig{Epochs: 2, BatchSize: 2, LearningRate: 0.001, Seed: 7, MaxGradNorm: 1}
			model := newFrozenTestScorer(t, newAliasEncoder("abcd"))
			identity := frozenResumeTestIdentity(t, train, validation)
			path := filepath.Join(t.TempDir(), "state.json")
			if _, _, err := TrainFrozenResumable(model, train, validation, cfg, identity, path, 1); err != nil {
				t.Fatal(err)
			}
			state, err := frozenResumeLoadState(path)
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "negative-variance":
				state.AdamState[0].V[0] = -1
			case "cursor":
				state.BatchOffset = 1
			}
			if err = frozenResumeSaveState(path, state); err != nil {
				t.Fatal(err)
			}
			if kind == "unknown" || kind == "trailing" {
				b, _ := os.ReadFile(path)
				if kind == "trailing" {
					b = append(b, []byte("{}")...)
				} else {
					b = append([]byte(`{"unknown":true,`), b[1:]...)
				}
				os.WriteFile(path, b, 0o600)
			}
			if _, _, err = TrainFrozenResumable(model, train, validation, cfg, identity, path, 0); err == nil {
				t.Fatal("corrupt state accepted")
			}
		})
	}
}
