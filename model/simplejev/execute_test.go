package simplejev

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

type testProvider func(string, Question) ([]Label, error)

func (fn testProvider) LabelLogits(state string, question Question) ([]Label, error) {
	return fn(state, question)
}

func TestEvaluateOrderedChoiceAndOrdinal(t *testing.T) {
	req := Request{State: "evidence", Questions: []Question{
		{ID: "pick", Instruction: "choose", Kind: "choice", Labels: []string{"alpha", "beta"}},
		{ID: "rate", Instruction: "rate", Kind: "ordinal", Labels: []string{"low", "middle", "high"}},
	}}
	calls := []string{}
	shared := []Label{{TokenID: 4, ID: "alpha", Logit: 2}, {TokenID: 7, ID: "beta", Logit: 2}}
	provider := testProvider(func(state string, q Question) ([]Label, error) {
		if state != "evidence" {
			t.Fatal("provider state mismatch")
		}
		calls = append(calls, q.ID)
		if q.ID == "pick" {
			return shared, nil
		}
		return []Label{{TokenID: 9, ID: "low", Logit: 0}, {TokenID: 2, ID: "middle", Logit: 0}, {TokenID: 3, ID: "high", Logit: 0}}, nil
	})
	answers, err := Evaluate(req, provider)
	if err != nil || !reflect.DeepEqual(calls, []string{"pick", "rate"}) || len(answers) != 2 || answers[0].SelectedID != "alpha" || math.Abs(answers[1].ExpectedValue-1) > 1e-6 {
		t.Fatalf("answers=%+v calls=%v err=%v", answers, calls, err)
	}
	if shared[0].Value != 0 || shared[1].Value != 0 {
		t.Fatal("provider row mutated")
	}
	shared[0].Logit = 99
	if answers[0].Probabilities[0] != .5 {
		t.Fatal("answer aliases provider buffer")
	}
}

func TestEvaluateProviderCannotMutateRequest(t *testing.T) {
	req := Request{State: "evidence", Questions: []Question{{ID: "q", Instruction: "choose", Kind: "choice", Labels: []string{"first", "second"}}}}
	provider := testProvider(func(_ string, q Question) ([]Label, error) {
		q.Labels[0], q.Labels[1] = q.Labels[1], q.Labels[0]
		return []Label{{TokenID: 1, ID: "second"}, {TokenID: 2, ID: "first"}}, nil
	})
	answers, err := Evaluate(req, provider)
	if err == nil || answers != nil || !reflect.DeepEqual(req.Questions[0].Labels, []string{"first", "second"}) {
		t.Fatalf("mutated authoritative request: answers=%v labels=%v err=%v", answers, req.Questions[0].Labels, err)
	}
}

func TestEvaluateRejectsInvalidProviderAtomically(t *testing.T) {
	req := Request{State: "evidence", Questions: []Question{
		{ID: "first", Instruction: "choose", Kind: "choice", Labels: []string{"a", "b"}},
		{ID: "second", Instruction: "choose", Kind: "choice", Labels: []string{"a", "b"}},
	}}
	valid := []Label{{TokenID: 1, ID: "a", Logit: 0}, {TokenID: 2, ID: "b", Logit: 1}}
	cases := map[string]func() ([]Label, error){
		"error":           func() ([]Label, error) { return nil, errors.New("backend unavailable") },
		"missing":         func() ([]Label, error) { return valid[:1], nil },
		"reordered":       func() ([]Label, error) { return []Label{valid[1], valid[0]}, nil },
		"duplicate token": func() ([]Label, error) { return []Label{valid[0], {TokenID: 1, ID: "b"}}, nil },
		"non-finite": func() ([]Label, error) {
			return []Label{valid[0], {TokenID: 2, ID: "b", Logit: float32(math.Inf(1))}}, nil
		},
	}
	for name, failure := range cases {
		t.Run(name, func(t *testing.T) {
			calls := 0
			answers, err := Evaluate(req, testProvider(func(_ string, _ Question) ([]Label, error) {
				calls++
				if calls == 1 {
					return valid, nil
				}
				return failure()
			}))
			if err == nil || answers != nil || calls != 2 {
				t.Fatalf("accepted partial result: answers=%v calls=%d err=%v", answers, calls, err)
			}
		})
	}
	if answers, err := Evaluate(req, nil); err == nil || answers != nil {
		t.Fatal("nil provider accepted")
	}
	req.State = ""
	if answers, err := Evaluate(req, testProvider(func(string, Question) ([]Label, error) {
		t.Fatal("called provider on invalid request")
		return nil, nil
	})); err == nil || answers != nil {
		t.Fatal("invalid request accepted")
	}
}
