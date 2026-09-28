package decider

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestFormatAnswerV15Choice(t *testing.T) {
	q := Question{Type: Choice, Instructions: "Which?", Criteria: []Criterion{{Name: "red"}, {Name: "blue"}}}
	p := []float32{0.82, 0.18}
	legacyInput := append([]float32(nil), p...)
	legacyQuestion, err := renderQuestion(q)
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := formatAnswer(legacyQuestion, legacyInput)
	if err != nil {
		t.Fatal(err)
	}
	got, err := FormatAnswerV15(q, p)
	if err != nil {
		t.Fatal(err)
	}
	if got.Choice != "red" || got.Confidence != 0.64 || got.XPMax != 0.82 || got.Certainty != legacy.Certainty || !reflect.DeepEqual(got.Probabilities, legacy.Probabilities) {
		t.Fatalf("v1.5 choice=%+v legacy=%+v", got, legacy)
	}
	if !reflect.DeepEqual(p, []float32{0.82, 0.18}) || legacy.Confidence != 0.82 {
		t.Fatalf("caller input or legacy contract changed: %v %+v", p, legacy)
	}
	b, err := json.Marshal(got)
	if err != nil || !strings.Contains(string(b), `"x_p_max":0.82`) || !strings.Contains(string(b), `"confidence":0.64`) {
		t.Fatalf("v1.5 JSON=%s err=%v", b, err)
	}
	for _, tc := range []struct {
		p                   []float32
		confidence, maximum float32
	}{
		{[]float32{.5, .5}, 0, .5},
		{[]float32{.2, .8}, .6, .8},
		{[]float32{1, 0}, 1, 1},
	} {
		v, err := FormatAnswerV15(q, tc.p)
		if err != nil || v.Confidence != tc.confidence || v.XPMax != tc.maximum {
			t.Fatalf("choice %+v => %+v err=%v", tc, v, err)
		}
		encoded, err := json.Marshal(v)
		if err != nil || !strings.Contains(string(encoded), `"confidence":`) || !strings.Contains(string(encoded), `"x_p_max":`) {
			t.Fatalf("choice JSON=%s err=%v", encoded, err)
		}
	}
	three := Question{Type: Choice, Instructions: "Which?", Criteria: []Criterion{{Name: "a"}, {Name: "b"}, {Name: "c"}}}
	if v, err := FormatAnswerV15(three, []float32{.8, .1, .1}); err != nil || v.Confidence != .7 || v.XPMax != .8 {
		t.Fatalf("three choice=%+v err=%v", v, err)
	}
}

func TestFormatAnswerV15ScoreAndNoul(t *testing.T) {
	q := Question{Type: Score, Instructions: "How much?", Levels: []any{"low", "medium", "high"}}
	for _, tc := range []struct {
		p                          []float32
		confidence, maximum, score float32
	}{
		{[]float32{0, .95, .05}, .925, .95, 1.05},
		{[]float32{.2, .2, .2, .2, .2}, 0, .2, 2},
	} {
		levels := q
		if len(tc.p) == 5 {
			levels.Levels = []any{"a", "b", "c", "d", "e"}
		}
		got, err := FormatAnswerV15(levels, tc.p)
		if err != nil || got.Confidence != tc.confidence || got.XPMax != tc.maximum || got.Score != tc.score {
			t.Fatalf("score %+v => %+v err=%v", tc, got, err)
		}
		encoded, err := json.Marshal(got)
		if err != nil || !strings.Contains(string(encoded), `"confidence":`) || !strings.Contains(string(encoded), `"x_p_max":`) {
			t.Fatalf("score JSON=%s err=%v", encoded, err)
		}
	}
	two := Question{Type: Score, Instructions: "How much?", Levels: []any{"no", "yes"}}
	if got, err := FormatAnswerV15(two, []float32{.3, .7}); err != nil || got.Confidence != .4 || got.XPMax != .7 {
		t.Fatalf("two levels=%+v err=%v", got, err)
	}
	noul, err := FormatAnswerV15(Question{Type: Noul, Instructions: "Refund?"}, []float32{.3, .7})
	if err != nil || noul.Noul != .7 || noul.XPMax != 0 || noul.Confidence != 0 {
		t.Fatalf("noul=%+v err=%v", noul, err)
	}
	b, err := json.Marshal(noul)
	if err != nil || strings.Contains(string(b), "x_p_max") || strings.Contains(string(b), "confidence") {
		t.Fatalf("noul JSON=%s err=%v", b, err)
	}
}

func TestFormatAnswerV15RejectsInvalid(t *testing.T) {
	q := Question{Type: Choice, Instructions: "Which?", Criteria: []Criterion{{Name: "a"}, {Name: "b"}}}
	for _, p := range [][]float32{{.5}, {0, 0}, {-1, 2}, {float32(math.NaN()), 1}, {float32(math.Inf(1)), 1}} {
		if got, err := FormatAnswerV15(q, p); err == nil || got.Answer.Type != "" {
			t.Fatalf("accepted invalid probabilities %v: %+v err=%v", p, got, err)
		}
	}
	if _, err := FormatAnswerV15(Question{Type: Noul}, []float32{.5, .5}); err == nil {
		t.Fatal("accepted missing instructions in pinned Go contract")
	}
}
