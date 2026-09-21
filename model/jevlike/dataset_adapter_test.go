package jevlike

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestValidateDatasetChoiceExampleStrict(t *testing.T) {
	valid, err := ValidateDatasetChoiceExample(ChoiceExample{
		Context: "Task: choose one.",
		Options: []string{"alpha", "beta"},
		Label:   1,
	})
	if err != nil {
		t.Fatalf("ValidateDatasetChoiceExample() error = %v", err)
	}
	if !reflect.DeepEqual(valid, ChoiceExample{Context: "Task: choose one.", Options: []string{"alpha", "beta"}, Label: 1}) {
		t.Fatalf("ValidateDatasetChoiceExample() = %#v", valid)
	}

	tooMany := make([]string, 33)
	for i := range tooMany {
		tooMany[i] = string(rune('a' + i))
	}
	invalid := []struct {
		name    string
		example ChoiceExample
		wantErr string
	}{
		{
			name:    "blank context",
			example: ChoiceExample{Context: " \n", Options: []string{"a", "b"}, Label: 0},
			wantErr: "context must be nonblank",
		},
		{
			name:    "invalid utf8 context",
			example: ChoiceExample{Context: string([]byte{0xff}), Options: []string{"a", "b"}, Label: 0},
			wantErr: "context must be valid UTF-8",
		},
		{
			name:    "duplicate option",
			example: ChoiceExample{Context: "ctx", Options: []string{"a", "a"}, Label: 0},
			wantErr: "options must contain 2..32 distinct valid UTF-8 nonblank strings",
		},
		{
			name:    "blank option",
			example: ChoiceExample{Context: "ctx", Options: []string{"a", "  "}, Label: 0},
			wantErr: "options must contain 2..32 distinct valid UTF-8 nonblank strings",
		},
		{
			name:    "too many options",
			example: ChoiceExample{Context: "ctx", Options: tooMany, Label: 0},
			wantErr: "options must contain 2..32 distinct valid UTF-8 nonblank strings",
		},
		{
			name:    "label out of range",
			example: ChoiceExample{Context: "ctx", Options: []string{"a", "b"}, Label: 2},
			wantErr: "label must be an option index",
		},
	}
	for _, tt := range invalid {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ValidateDatasetChoiceExample(tt.example)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ValidateDatasetChoiceExample() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestAdaptDatasetRowMultiNLI(t *testing.T) {
	row := json.RawMessage(`{"pairID":"pair-1","promptID":"prompt-7","premise":"A soccer game with multiple males playing.","hypothesis":"Some men are playing a sport.","genre":"fiction","label":0}`)
	example, err := AdaptDatasetRow("multi_nli", "matched", row)
	if err != nil {
		t.Fatalf("AdaptDatasetRow() error = %v", err)
	}
	if example.SourceID != "pair-1" || example.GroupID != "prompt-7" {
		t.Fatalf("source/group = %q/%q", example.SourceID, example.GroupID)
	}
	if example.Task != "natural_language_inference" || example.Genre != "fiction" {
		t.Fatalf("task/genre = %q/%q", example.Task, example.Genre)
	}
	if example.Choice.Label != 0 || !reflect.DeepEqual(example.Choice.Options, []string{"entailment", "neutral", "contradiction"}) {
		t.Fatalf("choice = %#v", example.Choice)
	}
	if !strings.Contains(example.Choice.Context, "Task: Decide whether") || !strings.Contains(example.Choice.Context, "Premise: A soccer game") || !strings.Contains(example.Choice.Context, "Hypothesis: Some men") {
		t.Fatalf("context = %q", example.Choice.Context)
	}
}

func TestAdaptDatasetRowQAAdapters(t *testing.T) {
	tests := []struct {
		name    string
		source  string
		config  string
		row     string
		wantID  string
		wantLbl int
		wantGen string
		wantCtx string
	}{
		{
			name:   "commonsense qa",
			source: "commonsense_qa",
			row:    `{"id":"csqa-1","question":"Where would you find a pillow?","question_concept":"bedroom","choices":{"label":["A","B","C"],"text":["bathroom","bed","garage"]},"answerKey":"B"}`,
			wantID: "csqa-1", wantLbl: 1, wantGen: "bedroom", wantCtx: "commonsense question",
		},
		{
			name:   "ai2 arc",
			source: "ai2_arc",
			config: "ARC-Challenge",
			row:    `{"id":"arc-1","question":"What planet do humans live on?","choices":{"label":["1","2","3"],"text":["Mars","Earth","Venus"]},"answerKey":"2"}`,
			wantID: "arc-1", wantLbl: 1, wantGen: "ARC-Challenge", wantCtx: "science question",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			example, err := AdaptDatasetRow(tt.source, tt.config, json.RawMessage(tt.row))
			if err != nil {
				t.Fatalf("AdaptDatasetRow() error = %v", err)
			}
			if example.SourceID != tt.wantID || example.GroupID != tt.wantID {
				t.Fatalf("source/group = %q/%q", example.SourceID, example.GroupID)
			}
			if example.Choice.Label != tt.wantLbl {
				t.Fatalf("label = %d", example.Choice.Label)
			}
			if example.Genre != tt.wantGen {
				t.Fatalf("genre = %q, want %q", example.Genre, tt.wantGen)
			}
			if !strings.Contains(example.Choice.Context, tt.wantCtx) {
				t.Fatalf("context = %q, want substring %q", example.Choice.Context, tt.wantCtx)
			}
		})
	}
}

func TestAdaptDatasetRowRejectsBadQALabelSchemas(t *testing.T) {
	tests := []struct {
		name    string
		source  string
		row     string
		wantErr string
	}{
		{
			name:    "missing answer key",
			source:  "commonsense_qa",
			row:     `{"id":"x","question":"Q?","choices":{"label":["A","B"],"text":["one","two"]}}`,
			wantErr: "answerKey is required",
		},
		{
			name:    "duplicate labels",
			source:  "ai2_arc",
			row:     `{"id":"x","question":"Q?","choices":{"label":["A","A"],"text":["one","two"]},"answerKey":"A"}`,
			wantErr: "choice labels must be unique",
		},
		{
			name:    "blank label",
			source:  "commonsense_qa",
			row:     `{"id":"x","question":"Q?","choices":{"label":["A",""],"text":["one","two"]},"answerKey":"A"}`,
			wantErr: "choice labels must be explicit, valid UTF-8 and nonblank",
		},
		{
			name:    "unmapped answer key",
			source:  "ai2_arc",
			row:     `{"id":"x","question":"Q?","choices":{"label":["A","B"],"text":["one","two"]},"answerKey":"C"}`,
			wantErr: "answerKey \"C\" is not present in choices.label",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := AdaptDatasetRow(tt.source, "", json.RawMessage(tt.row))
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("AdaptDatasetRow() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestAdaptCLINC(t *testing.T) {
	labels := []string{"balance", "transfer", "out_of_scope", "weather"}
	candidates := []int{2, 0, 3}
	row := json.RawMessage(`{"id":"clinc-1","utterance":"What is the temperature?","intent":"out_of_scope"}`)
	example, err := AdaptCLINC(row, labels, candidates)
	if err != nil {
		t.Fatalf("AdaptCLINC() error = %v", err)
	}
	if example.SourceID != "clinc-1" || example.GroupID != "clinc-1" {
		t.Fatalf("source/group = %q/%q", example.SourceID, example.GroupID)
	}
	if example.Task != "intent_classification" {
		t.Fatalf("task = %q", example.Task)
	}
	if !reflect.DeepEqual(example.Choice.Options, []string{"out_of_scope", "balance", "weather"}) {
		t.Fatalf("options = %#v", example.Choice.Options)
	}
	if example.Choice.Label != 0 {
		t.Fatalf("label = %d, want 0", example.Choice.Label)
	}
	if !strings.Contains(example.Choice.Context, "user utterance") || !strings.Contains(example.Choice.Context, "What is the temperature?") {
		t.Fatalf("context = %q", example.Choice.Context)
	}
}

func TestAdaptCLINCRejectsMissingGoldAndAdaptDatasetRowRequiresExplicitAPI(t *testing.T) {
	labels := []string{"balance", "transfer", "out_of_scope"}
	_, err := AdaptCLINC(json.RawMessage(`{"text":"hello","label":1}`), labels, []int{0, 2})
	if err == nil || !strings.Contains(err.Error(), "not present in candidateIDs") {
		t.Fatalf("AdaptCLINC() error = %v", err)
	}
	_, err = AdaptDatasetRow("clinc_oos", "", json.RawMessage(`{"text":"hello","label":1}`))
	if err == nil || !strings.Contains(err.Error(), "AdaptCLINC") {
		t.Fatalf("AdaptDatasetRow(clinc_oos) error = %v", err)
	}
}

func TestAdaptDatasetRejectsInvalidUTF8BeforeJSONReplacement(t *testing.T) {
	row := []byte(`{"id":"x","question":"q","choices":{"label":["A","B"],"text":["a","b"]},"answerKey":"A"}`)
	row[bytes.Index(row, []byte(`"q"`))+1] = 0xff
	if _, err := AdaptDatasetRow("commonsense_qa", "", row); err == nil {
		t.Fatal("accepted invalid source UTF-8")
	}
}

func TestMultiNLINumericPromptGroupID(t *testing.T) {
	row := json.RawMessage(`{"pairID":"31193n","promptID":31193,"premise":"A true premise.","hypothesis":"A possible claim.","genre":"government","label":1}`)
	ex, err := AdaptDatasetRow("multi_nli", "", row)
	if err != nil || ex.GroupID != "31193" {
		t.Fatalf("numeric group: %+v %v", ex, err)
	}
}

func TestCLINCCandidatePolicy(t *testing.T) {
	labels := []string{"book flight", "cancel flight", "book hotel", "out of scope", "card payment"}
	row := json.RawMessage(`{"text":"reserve a plane ticket","intent":0}`)
	a, err := CLINCCandidates(row, labels, 3, 7, 3)
	if err != nil {
		t.Fatal(err)
	}
	b, err := CLINCCandidates(row, labels, 3, 7, 3)
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatal(a, b, err)
	}
	if a[0] != 0 || a[1] != 3 || (a[2] != 1 && a[2] != 2) {
		t.Fatal("gold/OOS/confusable absent", a)
	}
	ex, err := AdaptCLINC(row, labels, a)
	if err != nil || ex.Choice.Options[ex.Choice.Label] != "book flight" {
		t.Fatal(ex, err)
	}
	if _, err = CLINCCandidates(row, labels, 33, 7, 3); err == nil {
		t.Fatal("accepted unsupported count")
	}
}
