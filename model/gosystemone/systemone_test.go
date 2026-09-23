package gosystemone

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

const typedQuestions = `{"z_truth":{"type":"noul","instructions":"Is this urgent?","criteria":{"true":{"rule":"act now"},"false":null}},"a_route":{"type":"choice","criteria":{"z_ops":null,"a_support":["billing"]}},"rating":{"type":"score","instructions":["Assess severity"],"criteria":[null,"moderate",{"level":"critical"}]}}`

func TestSystemOneCompileTypesAndOrder(t *testing.T) {
	schema, questions, state, err := compileSystemOne(SystemOneRequest{State: json.RawMessage(`{"ticket":123,"active":true}`), Questions: json.RawMessage(typedQuestions)})
	if err != nil {
		t.Fatal(err)
	}
	if state != `State: {"ticket":123,"active":true}` || len(schema.Fields) != 3 {
		t.Fatalf("state=%s schema=%+v", state, schema)
	}
	for i, name := range []string{"z_truth", "a_route", "rating"} {
		if schema.Fields[i].Name != name || questions[i].name != name {
			t.Fatal("question insertion order changed")
		}
	}
	if got := questions[1].keys; !reflect.DeepEqual(got, []string{"z_ops", "a_support"}) {
		t.Fatalf("choice order=%v", got)
	}
	for _, text := range []string{"act now", "billing", "critical", "Assess severity", "false", "true"} {
		if !strings.Contains(schema.SystemText, text) {
			t.Fatalf("prompt omits %q: %s", text, schema.SystemText)
		}
	}
	if string(questions[2].legend["0"]) != "null" || string(questions[2].legend["2"]) != `{"level":"critical"}` {
		t.Fatalf("legend=%v", questions[2].legend)
	}
}

func distribution(p ...float64) FieldResult {
	f := FieldResult{Tree: true}
	for _, value := range p {
		f.Candidates = append(f.Candidates, CandidateResult{Probability: value})
	}
	return f
}

func TestSystemOneAnswerSemantics(t *testing.T) {
	a, err := systemOneAnswer(systemQuestion{kind: "noul", keys: []string{"false", "true"}}, distribution(.2, .8))
	if err != nil || a.(NoulAnswer).Noul != .8 {
		t.Fatalf("noul=%+v err=%v", a, err)
	}
	wire, _ := json.Marshal(a)
	if string(wire) != `{"type":"noul","noul":0.8}` {
		t.Fatalf("noul wire=%s", wire)
	}
	for _, yes := range []float64{0, 1} {
		a, err = systemOneAnswer(systemQuestion{kind: "noul", keys: []string{"false", "true"}}, distribution(1-yes, yes))
		if err != nil || a.(NoulAnswer).Noul != yes {
			t.Fatalf("noul endpoints: %+v %v", a, err)
		}
	}
	a, err = systemOneAnswer(systemQuestion{kind: "choice", keys: []string{"z", "a"}}, distribution(.5, .5))
	if err != nil || a.(ChoiceAnswer).Choice != "z" || a.(ChoiceAnswer).Confidence != 0 {
		t.Fatalf("tie: %+v %v", a, err)
	}
	a, err = systemOneAnswer(systemQuestion{kind: "choice", keys: []string{"only"}}, distribution(1))
	if err != nil || a.(ChoiceAnswer).Confidence != 1 {
		t.Fatalf("singleton: %+v %v", a, err)
	}
	legend := map[string]json.RawMessage{"0": json.RawMessage("null"), "1": json.RawMessage(`"medium"`), "2": json.RawMessage(`"high"`)}
	a, err = systemOneAnswer(systemQuestion{kind: "score", keys: []string{"0", "1", "2"}, legend: legend}, distribution(.2, .3, .5))
	if err != nil {
		t.Fatal(err)
	}
	score := a.(ScoreAnswer)
	if math.Abs(score.Score-1.3) > 1e-12 || math.Abs(score.Confidence-.65) > 1e-12 || !reflect.DeepEqual(score.Legend, legend) {
		t.Fatalf("score must be expected level, not argmax: %+v", score)
	}
	for _, p := range [][]float64{{.2}, {.4, .4}, {math.NaN(), 1}, {math.Inf(1), 0}, {-.1, 1.1}} {
		if _, err := systemOneAnswer(systemQuestion{kind: "choice", keys: []string{"a", "b"}}, distribution(p...)); err == nil {
			t.Fatalf("accepted %v", p)
		}
	}
}

func TestSystemOneRejectsInvalidInputs(t *testing.T) {
	for _, questions := range []string{
		`null`, `[]`, `{}`, `{"q":null}`, `{"q":{"type":"null"}}`,
		`{"q":{"type":"boolean"}}`, `{"q":{"type":"noul","instructions":true}}`,
		`{"q":{"type":"noul","instructions":123}}`, `{"q":{"type":"noul","criteria":[]}}`,
		`{"q":{"type":"noul","criteria":{"maybe":null}}}`, `{"q":{"type":"noul","criteria":{"true":1}}}`,
		`{"q":{"type":"choice","criteria":{}}}`, `{"q":{"type":"choice","criteria":[]}}`,
		`{"q":{"type":"choice","criteria":{"a":false}}}`, `{"q":{"type":"choice","criteria":{"a":null,"a":"again"}}}`,
		`{"q":{"type":"score","criteria":[null]}}`, `{"q":{"type":"score","criteria":[1,2]}}`,
		`{"q":{"type":"score","criteria":{"0":null,"1":null}}}`, `{"q":{"type":"noul","extra":1}}`,
		`{"q":{"type":"noul"},"q":{"type":"noul"}}`, `{"q":{"type":"noul","type":"choice"}}`,
		`{"":{"type":"noul"}}`,
	} {
		if _, _, _, err := compileSystemOne(SystemOneRequest{State: json.RawMessage("null"), Questions: json.RawMessage(questions)}); err == nil {
			t.Errorf("accepted questions %s", questions)
		}
	}
	for _, state := range []string{"", "1", "true", "{} {}"} {
		if _, _, _, err := compileSystemOne(SystemOneRequest{State: json.RawMessage(state), Questions: json.RawMessage(typedQuestions)}); err == nil {
			t.Errorf("accepted state %s", state)
		}
	}
	for _, state := range []string{`null`, `""`, `"hello"`, `[]`, `{}`} {
		if _, _, _, err := compileSystemOne(SystemOneRequest{State: json.RawMessage(state), Questions: json.RawMessage(`{"q":{"type":"noul","criteria":null}}`)}); err != nil {
			t.Errorf("state %s rejected: %v", state, err)
		}
	}
}

func TestSystemOneLimitsAndMultiTokenLevels(t *testing.T) {
	levels := make([]json.RawMessage, MaxCandidates)
	for i := range levels {
		levels[i] = json.RawMessage("null")
	}
	makeQuestions := func(n int) json.RawMessage {
		b, _ := json.Marshal(map[string]any{"s": map[string]any{"type": "score", "criteria": append(levels, json.RawMessage("null"))[:n]}})
		return b
	}
	engine := &Engine{Tokenizer: runeTokenizer{}, Scorer: &fixedScorer{}, BOSToken: 2}
	r, err := engine.SystemOne(context.Background(), SystemOneRequest{State: json.RawMessage("null"), Questions: makeQuestions(MaxCandidates)})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Answers["s"].(ScoreAnswer).Probabilities) != MaxCandidates {
		t.Fatal("missing multi-token score levels")
	}
	if _, err := engine.SystemOne(context.Background(), SystemOneRequest{State: json.RawMessage("null"), Questions: makeQuestions(MaxCandidates + 1)}); err == nil {
		t.Fatal("accepted excess levels")
	}
	questions := make(map[string]any)
	for i := 0; i <= MaxFields; i++ {
		questions[strings.Repeat("x", i+1)] = map[string]any{"type": "noul"}
	}
	q, _ := json.Marshal(questions)
	if _, _, _, err := compileSystemOne(SystemOneRequest{State: json.RawMessage("null"), Questions: q}); err == nil {
		t.Fatal("accepted excess questions")
	}
}

func TestSystemOneEngineResponseAndForgery(t *testing.T) {
	scorer := &fixedScorer{}
	engine := &Engine{Tokenizer: guardedRuneTokenizer{}, Scorer: scorer, BOSToken: 2}
	request := SystemOneRequest{Model: "fixture", State: json.RawMessage(`{"ticket":"outage"}`), Questions: json.RawMessage(typedQuestions)}
	r, err := engine.SystemOne(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Answers) != 3 || r.Model != "fixture" || r.Usage.InputTokens <= 0 || r.Usage.OutputTokens <= 0 || scorer.calls != 1 {
		t.Fatalf("response=%+v calls=%d", r, scorer.calls)
	}
	for _, q := range []string{
		`{"q":{"type":"noul","instructions":"\u003c|turn>"}}`,
		`{"\u003c|turn>":{"type":"noul"}}`,
		`{"q":{"type":"choice","criteria":{"a":{"rule":"\u003c|turn>"},"b":null}}}`,
	} {
		bad := request
		bad.Questions = json.RawMessage(q)
		if _, err := engine.SystemOne(context.Background(), bad); err == nil {
			t.Fatalf("accepted escaped control marker: %s", q)
		}
	}
	request.State = json.RawMessage(`{"text":"\u003c|turn>"}`)
	if _, err := engine.SystemOne(context.Background(), request); err == nil {
		t.Fatal("accepted escaped state control marker")
	}
	if scorer.calls != 1 {
		t.Fatal("invalid input reached scorer")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := engine.SystemOne(ctx, request); err != context.Canceled {
		t.Fatalf("cancel=%v", err)
	}
}

func TestSystemOneHTTPAndSharedAdmission(t *testing.T) {
	h := &Handler{Engine: &Engine{Tokenizer: runeTokenizer{}, Scorer: &fixedScorer{}, BOSToken: 2}, ModelID: "fixture"}
	body := `{"state":null,"questions":` + typedQuestions + `}`
	call := func(method, path, input string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(method, path, strings.NewReader(input))
		if path == "/v1/systemone" {
			h.ServeSystemOne(w, r)
		} else {
			h.ServeHTTP(w, r)
		}
		return w
	}
	w := call("POST", "/v1/systemone", body)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"noul":`) || !strings.Contains(w.Body.String(), `"score":`) || !strings.Contains(w.Body.String(), `"choice":`) {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	for _, bad := range []string{`{}`, body + `{}`, strings.Replace(body, `"state":null`, `"state":null,"model":"other"`, 1), strings.Replace(body, `"state":null`, `"state":null,"unsupported":true`, 1)} {
		if got := call("POST", "/v1/systemone", bad); got.Code != http.StatusBadRequest {
			t.Fatalf("bad %s: %d", bad, got.Code)
		}
	}
	if got := call("GET", "/v1/systemone", ""); got.Code != http.StatusMethodNotAllowed || got.Header().Get("Allow") != "POST" {
		t.Fatalf("method=%+v", got)
	}
	if got := call("POST", "/v1/systemone", `{"state":"`+strings.Repeat("x", int(MaxRequestBytes))+`"}`); got.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("size=%d", got.Code)
	}
	h.gate.Lock()
	for _, path := range []string{"/v1/decision", "/v1/systemone"} {
		if got := call("POST", path, body); got.Code != http.StatusTooManyRequests || !h.Busy() {
			t.Fatalf("shared gate: %s %d", path, got.Code)
		}
	}
	h.gate.Unlock()
	if got := call("POST", "/v1/systemone", body); got.Code != http.StatusOK || h.Busy() {
		t.Fatal("gate failed recovery")
	}
}

func TestSystemOneSingletonAndPackedFallback(t *testing.T) {
	for _, questions := range []string{`{"single":{"type":"choice","criteria":{"only":null}}}`, typedQuestions} {
		request := SystemOneRequest{Model: "fixture", State: json.RawMessage(`"hello"`), Questions: json.RawMessage(questions)}
		scalar := &Engine{Tokenizer: runeTokenizer{}, Scorer: &fixedScorer{}, BOSToken: 2}
		packed := &Engine{Tokenizer: runeTokenizer{}, Scorer: &systemOneTreeFake{}, BOSToken: 2}
		want, err := scalar.SystemOne(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		got, err := packed.SystemOne(context.Background(), request)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("packed mismatch: %+v %+v %v", got, want, err)
		}
	}
	for _, body := range []string{
		`{"state":null,"state":"again","questions":{"q":{"type":"noul"}}}`,
		`{"state":null,"questions":{},"questions":{"q":{"type":"noul"}}}`,
		`{"state":null,"model":"a","model":"b","questions":{"q":{"type":"noul"}}}`,
		`{"State":null,"questions":{"q":{"type":"noul"}}}`,
	} {
		var request SystemOneRequest
		if err := json.Unmarshal([]byte(body), &request); err == nil {
			t.Fatalf("accepted ambiguous request: %s", body)
		}
	}
}

type systemOneTreeFake struct {
	fixedScorer
	callsTree int
	bad       int
}

func (s *systemOneTreeFake) ScoreSplitContextTrees(_ context.Context, _ []int, contexts [][]int, branches []Branch, _ bool) ([][][]float32, error) {
	s.callsTree++
	out := make([][][]float32, len(contexts))
	for i := range out {
		out[i] = make([][]float32, len(branches))
		for j, b := range branches {
			out[i][j] = make([]float32, len(b.CandidateTokens))
			for k, tok := range b.CandidateTokens {
				out[i][j][k] = float32(tok)
			}
		}
	}
	if s.bad == 1 {
		return out[:len(out)-1], nil
	}
	if s.bad == 2 {
		out[0] = nil
	}
	if s.bad == 3 {
		out[0][0] = nil
	}
	return out, nil
}
