package decider

import (
	"reflect"
	"strings"
	"testing"
)

// Frozen observations from Apache-2.0 Mapika/decider systemone.py
// a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f (SHA-256 884e4b0b…).
func TestNoulQuestionV15UpstreamObservations(t *testing.T) {
	cases := []struct {
		name     string
		q        Question
		question string
		options  []string
	}{
		{"true_only", Question{Type: Noul, Criteria: []Criterion{{Name: "true", Description: "money back"}}}, "Which answer fits the context?", []string{"no", "yes: money back"}},
		{"false_only", Question{Type: Noul, Instructions: "", Criteria: []Criterion{{Name: "false", Description: "no refund"}}}, "Which answer fits the context?", []string{"no: no refund", "yes"}},
		{"both", Question{Type: Noul, Criteria: []Criterion{{Name: "true", Description: "money back"}, {Name: "false", Description: "anything else"}}}, "Which answer fits the context?", []string{"no: anything else", "yes: money back"}},
		{"provided", Question{Type: Noul, Instructions: "Refund asked?", Criteria: []Criterion{{Name: "true", Description: "money back"}}}, "Refund asked?", []string{"no", "yes: money back"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			question, options, err := NoulQuestionV15(tc.q)
			if err != nil || question != tc.question || !reflect.DeepEqual(options, tc.options) {
				t.Fatalf("got %q %q err=%v want %q %q", question, options, err, tc.question, tc.options)
			}
			prompt, err := BuildPrompt("context", question, options)
			if err != nil || !strings.Contains(prompt, "Question: "+tc.question+"\nOptions:\n(A) "+tc.options[0]+"\n(B) "+tc.options[1]+"\nAnswer: (") {
				t.Fatalf("prompt=%q err=%v", prompt, err)
			}
			options[0] = "caller changed" // the returned options belong to this request
		})
	}
}

func TestNoulQuestionV15RejectsAndPreservesPinnedRenderer(t *testing.T) {
	for _, q := range []Question{
		{Type: Noul},
		{Type: Noul, Criteria: []Criterion{{Name: "true"}}},
		{Type: Noul, Criteria: []Criterion{{Name: "false", Description: ""}, {Name: "true", Description: nil}}},
		{Type: Noul, Criteria: []Criterion{{Name: "maybe", Description: "yes"}}},
		{Type: Noul, Criteria: []Criterion{{Name: "true", Description: "yes"}, {Name: "true", Description: "again"}}},
		{Type: Choice, Criteria: []Criterion{{Name: "true", Description: "yes"}}},
	} {
		if got, options, err := NoulQuestionV15(q); err == nil || got != "" || options != nil {
			t.Fatalf("accepted %+v: %q %q err=%v", q, got, options, err)
		}
	}
	if _, err := renderQuestion(Question{Type: Noul, Criteria: []Criterion{{Name: "true", Description: "yes"}}}); err == nil {
		t.Fatal("pinned renderer accepted instructionless Noul")
	}
	provided := Question{Type: Noul, Instructions: "Refund?", Criteria: []Criterion{{Name: "false", Description: "other"}, {Name: "true", Description: "money"}}}
	old, err := renderQuestion(provided)
	if err != nil {
		t.Fatal(err)
	}
	question, options, err := NoulQuestionV15(provided)
	if err != nil || question != old.text || !reflect.DeepEqual(options, old.options) {
		t.Fatalf("provided instructions changed: %q %q vs %+v err=%v", question, options, old, err)
	}
}
