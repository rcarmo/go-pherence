package simplejev

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

type renderFn func(string, Question) (string, []RenderedLabel, error)

func (f renderFn) Render(s string, q Question) (string, []RenderedLabel, error) { return f(s, q) }

type tokenizeFn func(string) ([]int, error)

func (f tokenizeFn) Encode(s string) ([]int, error) { return f(s) }

type logitsFn func([]int, []int) ([]float32, error)

func (f logitsFn) SelectedLogits(p []int, ids []int) ([]float32, error) { return f(p, ids) }

func adapterFixture() (TokenLogitAdapter, Request) {
	adapter := TokenLogitAdapter{
		Renderer: renderFn(func(_ string, q Question) (string, []RenderedLabel, error) {
			return "prefix:", []RenderedLabel{{PublicID: "one", ModelText: "A"}, {PublicID: "two", ModelText: "B"}}, nil
		}),
		Tokenizer: tokenizeFn(func(s string) ([]int, error) {
			switch s {
			case "prefix:":
				return []int{2, 3}, nil
			case "prefix:A":
				return []int{2, 3, 5}, nil
			case "prefix:B":
				return []int{2, 3, 7}, nil
			}
			return nil, errors.New("unexpected prompt")
		}),
		Backend: logitsFn(func(p []int, ids []int) ([]float32, error) {
			if !reflect.DeepEqual(p, []int{2, 3}) || !reflect.DeepEqual(ids, []int{5, 7}) {
				return nil, errors.New("unexpected selected logits request")
			}
			return []float32{.4, .8}, nil
		}),
	}
	req := Request{State: "evidence", Questions: []Question{{ID: "q", Kind: "choice", Instruction: "choose", Labels: []string{"one", "two"}}}}
	return adapter, req
}

func TestTokenLogitAdapterEvaluate(t *testing.T) {
	a, req := adapterFixture()
	answers, err := Evaluate(req, a)
	if err != nil || len(answers) != 1 || answers[0].SelectedID != "two" || answers[0].Probabilities[1] <= answers[0].Probabilities[0] {
		t.Fatalf("answers=%v err=%v", answers, err)
	}
	if !reflect.DeepEqual(req.Questions[0].Labels, []string{"one", "two"}) {
		t.Fatal("request labels mutated")
	}
}

func TestTokenLogitAdapterRejectsBoundaryAndBackend(t *testing.T) {
	base, req := adapterFixture()
	cases := map[string]func(*TokenLogitAdapter){
		"renderer error": func(a *TokenLogitAdapter) {
			a.Renderer = renderFn(func(string, Question) (string, []RenderedLabel, error) { return "", nil, errors.New("render") })
		},
		"empty prefix": func(a *TokenLogitAdapter) {
			a.Renderer = renderFn(func(string, Question) (string, []RenderedLabel, error) {
				return "", []RenderedLabel{{PublicID: "one", ModelText: "A"}, {PublicID: "two", ModelText: "B"}}, nil
			})
		},
		"too many labels": func(a *TokenLogitAdapter) {
			a.Renderer = renderFn(func(string, Question) (string, []RenderedLabel, error) {
				return "prefix:", []RenderedLabel{{PublicID: "one", ModelText: "A"}}, nil
			})
		},
		"swapped public IDs": func(a *TokenLogitAdapter) {
			a.Renderer = renderFn(func(string, Question) (string, []RenderedLabel, error) {
				return "prefix:", []RenderedLabel{{PublicID: "two", ModelText: "A"}, {PublicID: "one", ModelText: "B"}}, nil
			})
		},
		"empty model label": func(a *TokenLogitAdapter) {
			a.Renderer = renderFn(func(string, Question) (string, []RenderedLabel, error) {
				return "prefix:", []RenderedLabel{{PublicID: "one"}, {PublicID: "two", ModelText: "B"}}, nil
			})
		},
		"empty tokenization": func(a *TokenLogitAdapter) { a.Tokenizer = tokenizeFn(func(string) ([]int, error) { return nil, nil }) },
		"tokenizer error": func(a *TokenLogitAdapter) {
			a.Tokenizer = tokenizeFn(func(string) ([]int, error) { return nil, errors.New("tokenizer") })
		},
		"changed prefix": func(a *TokenLogitAdapter) {
			a.Tokenizer = tokenizeFn(func(s string) ([]int, error) {
				if s == "prefix:" {
					return []int{2, 3}, nil
				}
				return []int{8, 3, 5}, nil
			})
		},
		"multi-token": func(a *TokenLogitAdapter) {
			a.Tokenizer = tokenizeFn(func(s string) ([]int, error) {
				if s == "prefix:" {
					return []int{2, 3}, nil
				}
				return []int{2, 3, 5, 6}, nil
			})
		},
		"duplicate token": func(a *TokenLogitAdapter) {
			a.Tokenizer = tokenizeFn(func(s string) ([]int, error) {
				if s == "prefix:" {
					return []int{2, 3}, nil
				}
				return []int{2, 3, 5}, nil
			})
		},
		"negative token": func(a *TokenLogitAdapter) {
			a.Tokenizer = tokenizeFn(func(s string) ([]int, error) {
				if s == "prefix:" {
					return []int{2, 3}, nil
				}
				return []int{2, 3, -1}, nil
			})
		},
		"backend error": func(a *TokenLogitAdapter) {
			a.Backend = logitsFn(func([]int, []int) ([]float32, error) { return nil, errors.New("backend") })
		},
		"missing logits": func(a *TokenLogitAdapter) {
			a.Backend = logitsFn(func([]int, []int) ([]float32, error) { return []float32{.5}, nil })
		},
		"nonfinite logits": func(a *TokenLogitAdapter) {
			a.Backend = logitsFn(func([]int, []int) ([]float32, error) { return []float32{.5, float32(math.NaN())}, nil })
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			a := base
			mutate(&a)
			answers, err := Evaluate(req, a)
			if err == nil || answers != nil {
				t.Fatalf("accepted invalid adapter: answers=%v err=%v", answers, err)
			}
		})
	}
	if _, err := base.LabelLogits("", req.Questions[0]); err == nil {
		t.Fatal("accepted invalid state")
	}
	base.Renderer = nil
	if _, err := base.LabelLogits(req.State, req.Questions[0]); err == nil {
		t.Fatal("accepted nil renderer")
	}
}

func TestTokenLogitAdapterClonesTokenizerPrefix(t *testing.T) {
	a, req := adapterFixture()
	shared := []int{2, 3, 0}
	a.Tokenizer = tokenizeFn(func(s string) ([]int, error) {
		switch s {
		case "prefix:":
			return shared[:2], nil
		case "prefix:A":
			shared[0], shared[2] = 2, 5
			return shared, nil
		case "prefix:B":
			shared[0], shared[2] = 99, 7
			return shared, nil
		}
		return nil, errors.New("unexpected prompt")
	})
	answers, err := Evaluate(req, a)
	if err == nil || answers != nil {
		t.Fatalf("accepted tokenizer-mutated prefix: answers=%v err=%v", answers, err)
	}
}

func TestTokenLogitAdapterIsolatesCallbacks(t *testing.T) {
	a, req := adapterFixture()
	a.Renderer = renderFn(func(_ string, q Question) (string, []RenderedLabel, error) {
		q.Labels[0] = "forged"
		return "prefix:", []RenderedLabel{{PublicID: "one", ModelText: "A"}, {PublicID: "two", ModelText: "B"}}, nil
	})
	a.Backend = logitsFn(func(p []int, ids []int) ([]float32, error) { p[0] = 99; ids[0] = 99; return []float32{.4, .8}, nil })
	answers, err := Evaluate(req, a)
	if err != nil || len(answers) != 1 || answers[0].SelectedID != "two" || req.Questions[0].Labels[0] != "one" {
		t.Fatalf("aliasing: answers=%v req=%v err=%v", answers, req, err)
	}
}

func TestTokenLogitAdapterBounds(t *testing.T) {
	a, req := adapterFixture()
	a.Renderer = renderFn(func(string, Question) (string, []RenderedLabel, error) {
		return strings.Repeat("p", MaxPromptBytes+1), []RenderedLabel{{PublicID: "one", ModelText: "A"}, {PublicID: "two", ModelText: "B"}}, nil
	})
	if _, err := a.LabelLogits(req.State, req.Questions[0]); err == nil {
		t.Fatal("accepted oversized prefix")
	}
	a, req = adapterFixture()
	a.Renderer = renderFn(func(string, Question) (string, []RenderedLabel, error) {
		return "prefix:", []RenderedLabel{{PublicID: "one", ModelText: strings.Repeat("A", MaxModelLabelBytes+1)}, {PublicID: "two", ModelText: "B"}}, nil
	})
	if _, err := a.LabelLogits(req.State, req.Questions[0]); err == nil {
		t.Fatal("accepted oversized model label")
	}
	a, req = adapterFixture()
	a.Tokenizer = tokenizeFn(func(string) ([]int, error) { return make([]int, MaxPromptTokens+1), nil })
	if _, err := a.LabelLogits(req.State, req.Questions[0]); err == nil {
		t.Fatal("accepted oversized prompt tokenization")
	}
}
