// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT

package jevlike

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestParseChoiceExampleJSONValidation(t *testing.T) {
	tests := []struct {
		name    string
		line    string
		wantErr string
	}{
		{
			name:    "missing context",
			line:    `{"options":["a","b"],"label":0}`,
			wantErr: "each row needs string context and list options",
		},
		{
			name:    "not enough options",
			line:    `{"context":"ctx","options":["a"],"label":0}`,
			wantErr: "options must contain at least two non-empty strings",
		},
		{
			name:    "empty option",
			line:    `{"context":"ctx","options":["a",""],"label":0}`,
			wantErr: "options must contain at least two non-empty strings",
		},
		{
			name:    "label out of range",
			line:    `{"context":"ctx","options":["a","b"],"label":2}`,
			wantErr: "label must be an option index",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseChoiceExampleJSON([]byte(tt.line))
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ParseChoiceExampleJSON() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}

	example, err := ParseChoiceExampleJSON([]byte(`{"context":"","options":["yes","no"],"label":1}`))
	if err != nil {
		t.Fatalf("ParseChoiceExampleJSON() unexpected error: %v", err)
	}
	if example.Context != "" || example.Label != 1 || !reflect.DeepEqual(example.Options, []string{"yes", "no"}) {
		t.Fatalf("ParseChoiceExampleJSON() = %#v", example)
	}
}

func TestLoadJSONLValidationAndBlankLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "examples.jsonl")
	content := strings.Join([]string{
		`{"context":"ctx-1","options":["a","b"],"label":0}`,
		"",
		`{"context":"ctx-2","options":["c","d"],"label":1}`,
		"",
	}, "\n")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	examples, err := LoadJSONL(path)
	if err != nil {
		t.Fatalf("LoadJSONL() error = %v", err)
	}
	want := []ChoiceExample{
		{Context: "ctx-1", Options: []string{"a", "b"}, Label: 0},
		{Context: "ctx-2", Options: []string{"c", "d"}, Label: 1},
	}
	if !reflect.DeepEqual(examples, want) {
		t.Fatalf("LoadJSONL() = %#v, want %#v", examples, want)
	}

	badPath := filepath.Join(dir, "bad.jsonl")
	if err := os.WriteFile(badPath, []byte(`{"context":"ctx","options":["a"],"label":0}`+"\n"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	if _, err := LoadJSONL(badPath); err == nil || !strings.Contains(err.Error(), "options must contain at least two non-empty strings") {
		t.Fatalf("LoadJSONL() error = %v, want validation error", err)
	}
}

func TestEncodeBytesAndBuildByteBatchPadding(t *testing.T) {
	ids, err := EncodeBytes("é🙂", 3)
	if err != nil {
		t.Fatalf("EncodeBytes() error = %v", err)
	}
	wantIDs := []uint16{0xC3 + 1, 0xA9 + 1, 0xF0 + 1}
	if !reflect.DeepEqual(ids, wantIDs) {
		t.Fatalf("EncodeBytes() = %v, want %v", ids, wantIDs)
	}

	batch, err := BuildByteBatch([]ChoiceExample{
		{Context: "é🙂", Options: []string{"a", "ñ"}, Label: 1},
		{Context: "", Options: []string{"猫", "do", "x"}, Label: 0},
	}, 3, 2)
	if err != nil {
		t.Fatalf("BuildByteBatch() error = %v", err)
	}

	wantBatch := ByteBatch{
		ContextIDs:  [][]uint16{{0xC3 + 1, 0xA9 + 1, 0xF0 + 1}, {0, 0, 0}},
		ContextMask: [][]bool{{true, true, true}, {false, false, false}},
		OptionIDs: [][][]uint16{
			{{'a' + 1, 0}, {0xC3 + 1, 0xB1 + 1}, {0, 0}},
			{{0xE7 + 1, 0x8C + 1}, {'d' + 1, 'o' + 1}, {'x' + 1, 0}},
		},
		OptionTokenMask: [][][]bool{
			{{true, false}, {true, true}, {false, false}},
			{{true, true}, {true, true}, {true, false}},
		},
		OptionMask: [][]bool{{true, true, false}, {true, true, true}},
		Labels:     []int{1, 0},
	}
	if !reflect.DeepEqual(batch, wantBatch) {
		encoded, _ := json.MarshalIndent(batch, "", "  ")
		want, _ := json.MarshalIndent(wantBatch, "", "  ")
		t.Fatalf("BuildByteBatch() =\n%s\nwant\n%s", encoded, want)
	}
}

func TestBuildByteBatchZeroTokenBudgetKeepsOptionMask(t *testing.T) {
	batch, err := BuildByteBatch([]ChoiceExample{{Context: "ctx", Options: []string{"猫", "dog"}, Label: 0}}, 0, 0)
	if err != nil {
		t.Fatalf("BuildByteBatch() error = %v", err)
	}
	if !reflect.DeepEqual(batch.OptionMask, [][]bool{{true, true}}) {
		t.Fatalf("OptionMask = %v", batch.OptionMask)
	}
	if len(batch.OptionTokenMask) != 1 || len(batch.OptionTokenMask[0]) != 2 {
		t.Fatalf("OptionTokenMask shape = %#v", batch.OptionTokenMask)
	}
	if len(batch.OptionTokenMask[0][0]) != 0 || len(batch.OptionTokenMask[0][1]) != 0 {
		t.Fatalf("OptionTokenMask token lengths = %#v", batch.OptionTokenMask)
	}
}

func TestSyntheticExampleAndWriteSyntheticDataset(t *testing.T) {
	first := SyntheticExample(17)
	second := SyntheticExample(17)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("SyntheticExample() not deterministic: %#v != %#v", first, second)
	}
	if len(first.Options) < 2 || len(first.Options) > 8 {
		t.Fatalf("SyntheticExample() options = %d, want 2..8", len(first.Options))
	}
	if first.Label < 0 || first.Label >= len(first.Options) {
		t.Fatalf("SyntheticExample() label = %d for %d options", first.Label, len(first.Options))
	}
	if !strings.Contains(first.Context, first.Options[first.Label]) {
		t.Fatalf("SyntheticExample() context %q does not mention target %q", first.Context, first.Options[first.Label])
	}

	dir := t.TempDir()
	sizes := SplitSizes{Train: 2, Validation: 1, Test: 1}
	if err := WriteSyntheticDataset(dir, sizes, 17); err != nil {
		t.Fatalf("WriteSyntheticDataset() error = %v", err)
	}

	train, err := LoadJSONL(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadJSONL(train) error = %v", err)
	}
	validation, err := LoadJSONL(filepath.Join(dir, "validation.jsonl"))
	if err != nil {
		t.Fatalf("LoadJSONL(validation) error = %v", err)
	}
	testSplit, err := LoadJSONL(filepath.Join(dir, "test.jsonl"))
	if err != nil {
		t.Fatalf("LoadJSONL(test) error = %v", err)
	}
	if len(train) != 2 || len(validation) != 1 || len(testSplit) != 1 {
		t.Fatalf("split sizes = %d/%d/%d", len(train), len(validation), len(testSplit))
	}
	if !reflect.DeepEqual(train[0], SyntheticExample(17)) {
		t.Fatalf("train[0] = %#v, want %#v", train[0], SyntheticExample(17))
	}
	if !reflect.DeepEqual(train[1], SyntheticExample(17+syntheticStride)) {
		t.Fatalf("train[1] = %#v, want %#v", train[1], SyntheticExample(17+syntheticStride))
	}
	if !reflect.DeepEqual(validation[0], SyntheticExample(17+2*syntheticStride)) {
		t.Fatalf("validation[0] = %#v, want %#v", validation[0], SyntheticExample(17+2*syntheticStride))
	}
	if !reflect.DeepEqual(testSplit[0], SyntheticExample(17+3*syntheticStride)) {
		t.Fatalf("test[0] = %#v, want %#v", testSplit[0], SyntheticExample(17+3*syntheticStride))
	}
}

func TestBuildWikispeedia(t *testing.T) {
	root := t.TempDir()
	graphDir := filepath.Join(root, "wikispeedia_paths-and-graph")
	articlesDir := filepath.Join(root, "plaintext_articles")
	if err := os.MkdirAll(graphDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(graphDir) error = %v", err)
	}
	if err := os.MkdirAll(articlesDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(articlesDir) error = %v", err)
	}

	trainTarget := findTargetForSplit(t, "train")
	validationTarget := findTargetForSplit(t, "validation")
	testTarget := findTargetForSplit(t, "test")
	links := strings.Join([]string{
		"# source\ttarget",
		"Start\t" + trainTarget,
		"Start\t" + validationTarget,
		"Start\t" + testTarget,
		"Start\tOther_One",
		"Start\tOther_Two",
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(graphDir, "links.tsv"), []byte(links), 0o644); err != nil {
		t.Fatalf("WriteFile(links.tsv) error = %v", err)
	}
	paths := strings.Join([]string{
		"# header",
		"user-a\tsession-a\t0\tStart;" + trainTarget,
		"user-b\tsession-b\t0\tStart;" + validationTarget,
		"user-c\tsession-c\t0\tStart;" + testTarget,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(graphDir, "paths_finished.tsv"), []byte(paths), 0o644); err != nil {
		t.Fatalf("WriteFile(paths_finished.tsv) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(articlesDir, "Start.txt"), []byte("alpha\n\nbeta   gamma"), 0o644); err != nil {
		t.Fatalf("WriteFile(Start.txt) error = %v", err)
	}

	output := filepath.Join(root, "out")
	counts, err := BuildWikispeedia(root, output, WikispeediaOptions{MaxOptions: 2})
	if err != nil {
		t.Fatalf("BuildWikispeedia() error = %v", err)
	}
	if counts != (SplitSizes{Train: 1, Validation: 1, Test: 1}) {
		t.Fatalf("BuildWikispeedia() counts = %#v", counts)
	}

	train, err := LoadJSONL(filepath.Join(output, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadJSONL(train) error = %v", err)
	}
	validation, err := LoadJSONL(filepath.Join(output, "validation.jsonl"))
	if err != nil {
		t.Fatalf("LoadJSONL(validation) error = %v", err)
	}
	testSplit, err := LoadJSONL(filepath.Join(output, "test.jsonl"))
	if err != nil {
		t.Fatalf("LoadJSONL(test) error = %v", err)
	}

	assertWikiExample := func(t *testing.T, got ChoiceExample, rawTarget string) {
		t.Helper()
		if len(got.Options) != 2 {
			t.Fatalf("len(options) = %d, want 2", len(got.Options))
		}
		if got.Label < 0 || got.Label >= len(got.Options) {
			t.Fatalf("label = %d for %d options", got.Label, len(got.Options))
		}
		if !strings.Contains(got.Context, "Target article: "+strings.ReplaceAll(rawTarget, "_", " ")) {
			t.Fatalf("context %q missing target title %q", got.Context, rawTarget)
		}
		if !strings.Contains(got.Context, "Current article: Start") {
			t.Fatalf("context %q missing current article", got.Context)
		}
		if !strings.Contains(got.Context, "alpha beta gamma") {
			t.Fatalf("context %q missing normalized article body", got.Context)
		}
		for _, option := range got.Options {
			if strings.Contains(option, "_") {
				t.Fatalf("option %q still contains underscore", option)
			}
		}
	}

	assertWikiExample(t, train[0], trainTarget)
	assertWikiExample(t, validation[0], validationTarget)
	assertWikiExample(t, testSplit[0], testTarget)
}

func findTargetForSplit(t *testing.T, split string) string {
	t.Helper()
	for i := 0; i < 10000; i++ {
		candidate := "Target_" + strings.ReplaceAll(split, " ", "_") + "_" + string(rune('a'+(i%26)))
		candidate = candidate + strings.Repeat("x", i/26)
		if splitName(candidate) == split {
			return candidate
		}
	}
	t.Fatalf("no target found for split %q", split)
	return ""
}
