package jevlike

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestSplitDatasetExamplesDeterministicOrderIndependentAndGroupPreserved(t *testing.T) {
	const source = "synthetic"
	const seed int64 = 17
	gTrain := findGroupForDatasetSplit(t, source, seed, DatasetSplitTrain)
	gValidation := findGroupForDatasetSplit(t, source, seed, DatasetSplitValidation)
	gCalibration := findGroupForDatasetSplit(t, source, seed, DatasetSplitCalibration)

	examples := []DatasetExample{
		mustDatasetExample(t, ChoiceExample{Context: "ctx train one", Options: []string{"yes", "no"}, Label: 0}, "row-1", gTrain, "qa", "", "train"),
		mustDatasetExample(t, ChoiceExample{Context: "ctx train two", Options: []string{"left", "right"}, Label: 1}, "row-2", gTrain, "qa", "", "train"),
		mustDatasetExample(t, ChoiceExample{Context: "ctx validation", Options: []string{"red", "blue"}, Label: 0}, "row-3", gValidation, "qa", "", "train"),
		mustDatasetExample(t, ChoiceExample{Context: "ctx calibration", Options: []string{"up", "down"}, Label: 1}, "row-4", gCalibration, "qa", "", "train"),
		mustDatasetExample(t, ChoiceExample{Context: "ctx heldout", Options: []string{"cat", "dog"}, Label: 1}, "row-5", "heldout-group", "qa", "", "validation"),
		mustDatasetExample(t, ChoiceExample{Context: "ctx train one", Options: []string{"no", "yes"}, Label: 1}, "row-1-dup", gTrain, "qa", "", "train"),
	}

	first, err := SplitDatasetExamples(source, examples, seed)
	if err != nil {
		t.Fatalf("SplitDatasetExamples() error = %v", err)
	}
	reversed := append([]DatasetExample(nil), examples...)
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}
	second, err := SplitDatasetExamples(source, reversed, seed)
	if err != nil {
		t.Fatalf("SplitDatasetExamples(reversed) error = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("SplitDatasetExamples() not deterministic/order-independent:\nfirst=%#v\nsecond=%#v", first, second)
	}
	if len(first.Train) != 2 || len(first.Validation) != 1 || len(first.Calibration) != 1 || len(first.Test) != 1 {
		t.Fatalf("split sizes = %d/%d/%d/%d", len(first.Train), len(first.Validation), len(first.Calibration), len(first.Test))
	}
	if first.Train[0].GroupID != gTrain || first.Train[1].GroupID != gTrain {
		t.Fatalf("train groups = %#v", first.Train)
	}
	for _, example := range first.Test {
		if normalizeSplitName(example.OriginalSplit) == DatasetSplitTrain {
			t.Fatalf("train example leaked into test: %#v", example)
		}
	}
	for _, example := range first.Train {
		if example.SourceID == "row-1-dup" {
			t.Fatalf("duplicate representative retained unexpectedly: %#v", example)
		}
	}
}

func TestSplitDatasetExamplesRejectsHeldoutLeak(t *testing.T) {
	const source = "synthetic"
	examples := []DatasetExample{
		mustDatasetExample(t, ChoiceExample{Context: "ctx", Options: []string{"a", "b"}, Label: 0}, "train-row", "group-a", "qa", "", "train"),
		mustDatasetExample(t, ChoiceExample{Context: "ctx", Options: []string{"b", "a"}, Label: 1}, "heldout-row", "group-b", "qa", "", "validation"),
	}
	_, err := SplitDatasetExamples(source, examples, 1)
	if err == nil || !strings.Contains(err.Error(), "official heldout and train") {
		t.Fatalf("SplitDatasetExamples() error = %v", err)
	}
}

func TestSplitDatasetExamplesRejectsConflictingLabels(t *testing.T) {
	const source = "synthetic"
	examples := []DatasetExample{
		mustDatasetExample(t, ChoiceExample{Context: "ctx", Options: []string{"a", "b"}, Label: 0}, "row-1", "group-a", "qa", "", "train"),
		mustDatasetExample(t, ChoiceExample{Context: "ctx", Options: []string{"b", "a"}, Label: 0}, "row-2", "group-b", "qa", "", "train"),
	}
	_, err := SplitDatasetExamples(source, examples, 1)
	if err == nil || !strings.Contains(err.Error(), "conflicting labels") {
		t.Fatalf("SplitDatasetExamples() error = %v", err)
	}
}

func TestPermuteDatasetExampleDeterministicAndOrderIndependent(t *testing.T) {
	const source = "synthetic"
	first := mustDatasetExample(t, ChoiceExample{Context: "ctx", Options: []string{"beta", "alpha", "gamma"}, Label: 1}, "row-1", "group-a", "qa", "", "train")
	second := mustDatasetExample(t, ChoiceExample{Context: "ctx", Options: []string{"gamma", "beta", "alpha"}, Label: 2}, "row-1", "group-a", "qa", "", "train")

	a, err := PermuteDatasetExample(source, first, 99)
	if err != nil {
		t.Fatalf("PermuteDatasetExample(first) error = %v", err)
	}
	b, err := PermuteDatasetExample(source, first, 99)
	if err != nil {
		t.Fatalf("PermuteDatasetExample(first repeat) error = %v", err)
	}
	c, err := PermuteDatasetExample(source, second, 99)
	if err != nil {
		t.Fatalf("PermuteDatasetExample(second) error = %v", err)
	}
	if !reflect.DeepEqual(a, b) || !reflect.DeepEqual(a, c) {
		t.Fatalf("permutations differ:\na=%#v\nb=%#v\nc=%#v", a, b, c)
	}
	if a.Choice.Options[a.Choice.Label] != "alpha" {
		t.Fatalf("gold option moved incorrectly: %#v", a.Choice)
	}
}

func mustDatasetExample(t *testing.T, choice ChoiceExample, sourceID, groupID, task, genre, split string) DatasetExample {
	t.Helper()
	example, err := ValidateDatasetExample(DatasetExample{
		Choice:        choice,
		SourceID:      sourceID,
		GroupID:       groupID,
		Task:          task,
		Genre:         genre,
		OriginalSplit: split,
	})
	if err != nil {
		t.Fatalf("ValidateDatasetExample() error = %v", err)
	}
	return example
}

func findGroupForDatasetSplit(t *testing.T, source string, seed int64, want string) string {
	t.Helper()
	for i := 0; i < 10000; i++ {
		group := fmt.Sprintf("group-%d", i)
		if trainSplitName(source, group, seed) == want {
			return group
		}
	}
	t.Fatalf("no group for split %q", want)
	return ""
}

func TestDatasetSplitRejectsUnknownAndGroupBoundaryLeak(t *testing.T) {
	for _, split := range []string{"", "trian", "calibration"} {
		ex := datasetExample("x", "g", split)
		if _, err := SplitDatasetExamples("source", []DatasetExample{ex}, 7); err == nil {
			t.Fatalf("accepted %q", split)
		}
	}
	a, b := datasetExample("a", "same-group", "train"), datasetExample("b", "same-group", "validation")
	if _, err := SplitDatasetExamples("source", []DatasetExample{a, b}, 7); err == nil {
		t.Fatal("group leaked into heldout")
	}
}

func TestDatasetSplitDuplicateConnectsWholeGroups(t *testing.T) {
	a := datasetExample("a", "g1", "train")
	b := datasetExample("b", "g2", "train")
	b.Choice = a.Choice
	c := datasetExample("c", "g1", "train")
	d := datasetExample("d", "g2", "train")
	for seed := int64(0); seed < 100; seed++ {
		s, err := SplitDatasetExamples("source", []DatasetExample{a, b, c, d}, seed)
		if err != nil {
			t.Fatal(err)
		}
		nonempty := 0
		for _, set := range [][]DatasetExample{s.Train, s.Validation, s.Calibration, s.Test} {
			if len(set) > 0 {
				nonempty++
				if len(set) != 3 {
					t.Fatal("connected siblings split", s)
				}
			}
		}
		if nonempty != 1 {
			t.Fatal("duplicate connected groups must share partition", s)
		}
	}
}

func datasetExample(id, group, split string) DatasetExample {
	return DatasetExample{Choice: ChoiceExample{Context: "question " + id, Options: []string{"yes", "no"}, Label: 0}, SourceID: id, GroupID: group, Task: "test", OriginalSplit: split}
}
