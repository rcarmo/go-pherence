package laya

import (
	"encoding/json"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"github.com/rcarmo/go-pherence/model/modernbert"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

type p struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}
type fx struct {
	LayaPin         string          `json:"laya_pin"`
	TransformersPin string          `json:"transformers_pin"`
	EncoderConfig   json.RawMessage `json:"encoder_config"`
	Tokens          []int           `json:"tokens"`
	Mask            []int           `json:"attention_mask"`
	Markers         []int           `json:"marker_pos"`
	QType           int             `json:"qtype"`
	Tensors         map[string]p    `json:"tensors"`
	Logits          p               `json:"logits"`
	Actions         p               `json:"action_logits"`
}

func fixture(t *testing.T) (*Model, fx) {
	t.Helper()
	b, e := os.ReadFile("testdata/tiny.json")
	if e != nil {
		t.Fatal(e)
	}
	var f fx
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	et, ht := map[string]modernbert.Tensor{}, map[string]Tensor{}
	for n, v := range f.Tensors {
		if strings.HasPrefix(n, "encoder.") {
			x := modernbert.Tensor{Shape: v.Shape, Data: v.Data}
			et[strings.TrimPrefix(n, "encoder.")] = x
		} else {
			ht[n] = Tensor{Shape: v.Shape, Data: v.Data}
		}
	}
	enc, e := modernbert.New(f.EncoderConfig, et)
	if e != nil {
		t.Fatal(e)
	}
	m, e := New(enc, ht)
	if e != nil {
		t.Fatal(e)
	}
	return m, f
}
func cmp(t *testing.T, n string, g, w []float32) {
	for i, x := range g {
		d := math.Abs(float64(x - w[i]))
		if d > 3e-5+3e-4*math.Abs(float64(w[i])) {
			t.Fatalf("%s[%d] %g != %g", n, i, x, w[i])
		}
	}
}
func TestUpstreamParity(t *testing.T) {
	m, f := fixture(t)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	got, e := m.Forward(f.Tokens, mask, f.Markers, QuestionType(f.QType))
	if e != nil {
		t.Fatal(e)
	}
	cmp(t, "logits", got.Logits, f.Logits.Data)
	cmp(t, "actions", got.ActionLogits, f.Actions.Data)
}
func TestAdmission(t *testing.T) {
	m, f := fixture(t)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	for _, markers := range [][]int{nil, {1}, {-1, 2}, {1, 99}} {
		if _, e := m.Forward(f.Tokens, mask, markers, Choice); e == nil {
			t.Fatal("markers")
		}
	}
	if _, e := m.Forward(f.Tokens, mask, f.Markers, QuestionType(7)); e == nil {
		t.Fatal("type")
	}
}
func BenchmarkReleasedForward(b *testing.B) {
	dir := os.Getenv("GO_PHERENCE_LAYA_MODEL")
	cfg := os.Getenv("GO_PHERENCE_MODERNBERT_CONFIG")
	if dir == "" || cfg == "" {
		b.Skip("set released model")
	}
	m, _, e := Load(dir, cfg)
	if e != nil {
		b.Fatal(e)
	}
	raw, e := os.ReadFile("testdata/released.json")
	if e != nil {
		b.Fatal(e)
	}
	var fixture struct {
		Items []struct {
			IDs     []int `json:"ids"`
			Markers []int `json:"markers"`
		} `json:"items"`
	}
	if e = json.Unmarshal(raw, &fixture); e != nil {
		b.Fatal(e)
	}
	ids, markers := fixture.Items[0].IDs, fixture.Items[0].Markers
	mask := make([]bool, len(ids))
	for i := range mask {
		mask[i] = true
	}
	s, e := m.NewSession(len(ids), len(markers))
	if e != nil {
		b.Fatal(e)
	}
	logits := make([]float32, len(markers))
	actions := make([]float32, m.actions)
	probs := make([]float32, len(markers))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, e = s.ForwardInto(logits, actions, probs, ids, mask, markers, Choice); e != nil {
			b.Fatal(e)
		}
	}
}

func BenchmarkTinyForward(b *testing.B) {
	m, f := fixtureB(b)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	s, e := m.NewSession(len(f.Tokens), len(f.Markers))
	if e != nil {
		b.Fatal(e)
	}
	logits := make([]float32, len(f.Markers))
	actions := make([]float32, m.actions)
	probs := make([]float32, len(f.Markers))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, e = s.ForwardInto(logits, actions, probs, f.Tokens, mask, f.Markers, QuestionType(f.QType)); e != nil {
			b.Fatal(e)
		}
	}
}
func fixtureB(tb testing.TB) (*Model, fx) {
	tb.Helper()
	b, e := os.ReadFile("testdata/tiny.json")
	if e != nil {
		tb.Fatal(e)
	}
	var f fx
	if e = json.Unmarshal(b, &f); e != nil {
		tb.Fatal(e)
	}
	et, ht := map[string]modernbert.Tensor{}, map[string]Tensor{}
	for n, v := range f.Tensors {
		if strings.HasPrefix(n, "encoder.") {
			et[strings.TrimPrefix(n, "encoder.")] = modernbert.Tensor{Shape: v.Shape, Data: v.Data}
		} else {
			ht[n] = Tensor{Shape: v.Shape, Data: v.Data}
		}
	}
	enc, e := modernbert.New(f.EncoderConfig, et)
	if e != nil {
		tb.Fatal(e)
	}
	m, e := New(enc, ht)
	if e != nil {
		tb.Fatal(e)
	}
	return m, f
}
func TestAPIAndAdmission(t *testing.T) {
	m, f := fixture(t)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	if got := QuestionType(99).String(); got != "" {
		t.Fatal(got)
	}
	badQuestions := []Question{{Type: Choice, Criteria: []Criterion{{ID: "one"}}}, {Type: Choice, Criteria: []Criterion{{ID: ""}, {ID: "two"}}}, {Type: Score, Criteria: []Criterion{{Description: "one"}}}, {Type: Noul, Criteria: []Criterion{{ID: "maybe"}}}, {Type: QuestionType(99)}}
	for _, q := range badQuestions {
		if _, e := renderOptions(q); e == nil {
			t.Fatalf("accepted %+v", q)
		}
	}
	if _, _, e := BuildSequence(nil, "", Question{}, 1, 1, false); e == nil {
		t.Fatal("nil tokenizer")
	}
	if _, e := m.NewSession(0, 2); e == nil {
		t.Fatal("zero sequence")
	}
	if _, e := m.NewSession(2, 3); e == nil {
		t.Fatal("too many options")
	}
	s, _ := m.NewSession(len(f.Tokens), len(f.Markers))
	logits := make([]float32, len(f.Markers))
	actions := make([]float32, m.actions)
	probs := make([]float32, len(f.Markers))
	if _, e := s.ForwardInto(logits, actions, probs, f.Tokens, mask[:len(mask)-1], f.Markers, Choice); e == nil {
		t.Fatal("short mask")
	}
	if _, e := m.answerFromOutput(Output{Logits: []float32{float32(math.NaN()), 0}, ActionLogits: []float32{0, 0}}, Question{Type: Noul}); e == nil {
		t.Fatal("nan")
	}
	if _, e := m.answerFromOutput(Output{Logits: []float32{0, 0}}, Question{Type: Choice, Criteria: []Criterion{{ID: "a"}}}); e == nil {
		t.Fatal("option mismatch")
	}
	if _, e := m.SystemOne(nil, "", []NamedQuestion{{ID: "q"}}, Config{}); e == nil {
		t.Fatal("nil tokenizer")
	}
}

func TestOwnershipAndMalformedModels(t *testing.T) {
	m, f := fixture(t)
	before := m.typeEmb[0]
	v := f.Tensors["type_emb.weight"]
	v.Data[0] += 100
	f.Tensors["type_emb.weight"] = v
	if m.typeEmb[0] != before {
		t.Fatal("caller tensor alias")
	}
	_, base := fixture(t)
	et, ht := map[string]modernbert.Tensor{}, map[string]Tensor{}
	for n, v := range base.Tensors {
		if strings.HasPrefix(n, "encoder.") {
			et[strings.TrimPrefix(n, "encoder.")] = modernbert.Tensor{Shape: v.Shape, Data: v.Data}
		} else {
			ht[n] = Tensor{Shape: v.Shape, Data: v.Data}
		}
	}
	enc, e := modernbert.New(base.EncoderConfig, et)
	if e != nil {
		t.Fatal(e)
	}
	cases := []func(map[string]Tensor){func(x map[string]Tensor) { delete(x, "scorer.1.bias") }, func(x map[string]Tensor) { v := x["scorer.0.weight"]; v.Shape = []int{1, 1}; x["scorer.0.weight"] = v }, func(x map[string]Tensor) { x["unexpected"] = Tensor{Shape: []int{1}, Data: []float32{1}} }, func(x map[string]Tensor) { delete(x, "head.layers.0.self_attn.in_proj_weight") }, func(x map[string]Tensor) { v := x["act_head.2.weight"]; v.Shape = []int{1}; x["act_head.2.weight"] = v }}
	for i, mutate := range cases {
		x := maps.Clone(ht)
		mutate(x)
		if _, e := New(enc, x); e == nil {
			t.Fatalf("malformed case %d accepted", i)
		}
	}
}

func TestLoadRejections(t *testing.T) {
	if _, _, e := Load(t.TempDir(), "missing"); e == nil {
		t.Fatal("missing config")
	}
	dir := t.TempDir()
	if e := os.WriteFile(filepath.Join(dir, "rl_agent_config.json"), []byte("{"), 0600); e != nil {
		t.Fatal(e)
	}
	if _, _, e := Load(dir, "missing"); e == nil {
		t.Fatal("bad json")
	}
	if e := os.WriteFile(filepath.Join(dir, "rl_agent_config.json"), []byte(`{"max_len":1,"head_max_len":2}`), 0600); e != nil {
		t.Fatal(e)
	}
	if _, _, e := Load(dir, "missing"); e == nil {
		t.Fatal("bad limits")
	}
}

func writeTinyLayaDir(t *testing.T) (string, string) {
	t.Helper()
	_, f := fixture(t)
	dir := t.TempDir()
	et, ht := map[string]checkpoint.Tensor{}, map[string]checkpoint.Tensor{}
	for n, v := range f.Tensors {
		x := checkpoint.Tensor{Shape: v.Shape, Data: v.Data}
		if strings.HasPrefix(n, "encoder.") {
			et[n] = x
		} else {
			ht[n] = x
		}
	}
	for n, v := range et {
		ht[n] = v
	}
	cfg := Config{MaxLen: 64, HeadMaxLen: 32, HeadLayers: 2, ActCosts: map[string]float64{"escalate": .5}, Temperature: []float32{1, 1, 1}, TemperatureByOptions: map[string]float32{"choice:2": 1}}
	raw, _ := json.Marshal(cfg)
	if e := os.WriteFile(filepath.Join(dir, "rl_agent_config.json"), raw, 0600); e != nil {
		t.Fatal(e)
	}
	if e := checkpoint.Save(filepath.Join(dir, "model.safetensors"), &checkpoint.Checkpoint{FormatVersion: 1, Tensors: ht}); e != nil {
		t.Fatal(e)
	}
	encCfg := filepath.Join(dir, "encoder.json")
	if e := os.WriteFile(encCfg, f.EncoderConfig, 0600); e != nil {
		t.Fatal(e)
	}
	return dir, encCfg
}

func TestTinyLoadAndConfigRejections(t *testing.T) {
	dir, cfg := writeTinyLayaDir(t)
	m, c, e := Load(dir, cfg)
	if e != nil {
		t.Fatal(e)
	}
	if m.hidden != 16 || c.MaxLen != 64 {
		t.Fatal("tiny load")
	}
	var raw map[string]any
	b, _ := os.ReadFile(filepath.Join(dir, "rl_agent_config.json"))
	json.Unmarshal(b, &raw)
	for name, value := range map[string]any{"temperature": []float64{1}, "temperature_by_options": map[string]float64{"bad": 1}, "head_layers": 1, "act_costs": map[string]float64{}} {
		copy := maps.Clone(raw)
		copy[name] = value
		data, _ := json.Marshal(copy)
		os.WriteFile(filepath.Join(dir, "rl_agent_config.json"), data, 0600)
		if _, _, e := Load(dir, cfg); e == nil {
			t.Fatalf("accepted %s", name)
		}
	}
}

func TestBuildSequenceAndPublicAPI(t *testing.T) {
	dir := t.TempDir()
	src := "../../checkpoints/modernbert-large/tokenizer.json"
	if _, e := os.Stat(src); e != nil {
		t.Skip("released tokenizer unavailable")
	}
	data, e := os.ReadFile(src)
	if e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0600); e != nil {
		t.Fatal(e)
	}
	tok, e := modernbert.LoadTokenizer(dir)
	if e != nil {
		t.Fatal(e)
	}
	choice := Question{Type: Choice, Instructions: "[MASK] " + strings.Repeat("long ", 50), Criteria: []Criterion{{ID: "a", Description: strings.Repeat("word ", 60)}, {ID: "b", Description: strings.Repeat("word ", 60)}}}
	right, marks, e := BuildSequence(tok, strings.Repeat("state ", 50), choice, 40, 20, false)
	if e != nil {
		t.Fatal(e)
	}
	left, _, e := BuildSequence(tok, strings.Repeat("state ", 50), choice, 40, 20, true)
	if e != nil {
		t.Fatal(e)
	}
	if len(right) > 40 || len(marks) != 2 || slices.Equal(right, left) {
		t.Fatalf("truncation right=%v left=%v markers=%v", right, left, marks)
	}
	m, f := fixture(t)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	q := Question{Type: Noul, Instructions: "is true"}
	markers := f.Markers[:2]
	if _, e := m.Answer(f.Tokens, mask, markers[:1], q); e == nil {
		t.Fatal("option mismatch")
	}
	if _, e := m.Answer(f.Tokens, mask, markers, q); e != nil {
		t.Fatal(e)
	}
	cfg := Config{MaxLen: 64, HeadMaxLen: 32}
	if _, e := m.SystemOne(tok, "x", nil, cfg); e == nil {
		t.Fatal("empty questions")
	}
	if _, e := m.SystemOne(tok, "x", []NamedQuestion{{ID: "", Question: q}}, cfg); e == nil {
		t.Fatal("empty id")
	}
	if _, e := m.SystemOne(tok, "x", []NamedQuestion{{ID: "q", Question: q}, {ID: "q", Question: q}}, cfg); e == nil {
		t.Fatal("duplicate id")
	}
	for _, n := range []int{2, 3, 6, 11} {
		if !validTemperatureBucket(optionBucket(Choice, n)) {
			t.Fatal(n)
		}
	}
	if validTemperatureBucket("bad") {
		t.Fatal("bad bucket")
	}
}

func TestResponseJSONContract(t *testing.T) {
	a := Answer{Type: "score", Score: 1.5, Legend: map[string]string{"0": "no", "1": "yes"}, Probabilities: map[string]float32{"0": .25, "1": .75}, Confidence: .2, Action: Action{ActProbability: .8}}
	got, e := json.Marshal(a)
	if e != nil {
		t.Fatal(e)
	}
	var decoded map[string]any
	if e = json.Unmarshal(got, &decoded); e != nil {
		t.Fatal(e)
	}
	for _, key := range []string{"type", "score", "legend", "probabilities", "confidence", "action"} {
		if _, ok := decoded[key]; !ok {
			t.Fatalf("missing %s: %s", key, got)
		}
	}
	if _, ok := decoded["choice"]; ok {
		t.Fatalf("unexpected choice: %s", got)
	}
	var round Answer
	if e = json.Unmarshal(got, &round); e != nil || !reflect.DeepEqual(a, round) {
		t.Fatalf("roundtrip=%+v err=%v", round, e)
	}
}

func TestSessionParityAndAllocations(t *testing.T) {
	m, f := fixture(t)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	s, e := m.NewSession(len(f.Tokens), len(f.Markers))
	if e != nil {
		t.Fatal(e)
	}
	logits := make([]float32, len(f.Markers))
	actions := make([]float32, m.actions)
	probs := make([]float32, len(f.Markers))
	if _, e = s.ForwardInto(logits, actions, probs, f.Tokens, mask, f.Markers, QuestionType(f.QType)); e != nil {
		t.Fatal(e)
	}
	cmp(t, "session logits", logits, f.Logits.Data)
	cmp(t, "session actions", actions, f.Actions.Data)
	a := testing.AllocsPerRun(100, func() {
		if _, e := s.ForwardInto(logits, actions, probs, f.Tokens, mask, f.Markers, QuestionType(f.QType)); e != nil {
			panic(e)
		}
	})
	if a != 0 {
		t.Fatalf("warm allocations %.2f", a)
	}
}
func TestReleasedUpstreamParity(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_LAYA_MODEL")
	encCfg := os.Getenv("GO_PHERENCE_MODERNBERT_CONFIG")
	if dir == "" || encCfg == "" {
		t.Skip("set released Laya and encoder config")
	}
	m, c, e := Load(dir, encCfg)
	if e != nil {
		t.Fatal(e)
	}
	if c.MaxLen != 512 || c.HeadLayers != 2 {
		t.Fatal(c)
	}
	var f struct {
		SourcePin  string   `json:"source_pin"`
		ModelPin   string   `json:"model_pin"`
		Input      [][]int  `json:"input_ids"`
		Mask       [][]int  `json:"attention_mask"`
		Markers    [][]int  `json:"marker_pos"`
		MarkerMask [][]bool `json:"marker_mask"`
		QTypes     []int    `json:"qtype"`
		Items      []struct {
			IDs     []int `json:"ids"`
			Markers []int `json:"markers"`
			QType   int   `json:"qtype"`
		} `json:"items"`
		Logits   [][]float32 `json:"logits"`
		Actions  [][]float32 `json:"action_logits"`
		Response struct {
			Model   string `json:"model"`
			Usage   Usage  `json:"usage"`
			Answers map[string]struct {
				Type          string             `json:"type"`
				Choice        string             `json:"choice"`
				Score         float32            `json:"score"`
				Noul          float32            `json:"noul"`
				Probabilities map[string]float32 `json:"probabilities"`
				Confidence    float32            `json:"confidence"`
				Action        Action             `json:"action"`
				Legend        map[string]string  `json:"legend"`
			} `json:"answers"`
		} `json:"response"`
	}
	b, e := os.ReadFile("testdata/released.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	if f.SourcePin != "42626c348753fbb17572a813127df2278a1ec527" || f.ModelPin != "1c5edc17a7acd8701df6fc341c0d179f1c62c982" {
		t.Fatal("pins")
	}
	tok, err := modernbert.LoadTokenizer(filepath.Join(dir, "tokenizer"))
	if err != nil {
		t.Fatal(err)
	}
	questions := []Question{{Type: Choice, Instructions: "What color is the bicycle?", Criteria: []Criterion{{ID: "red"}, {ID: "blue", Description: "A blue bicycle"}}}, {Type: Score, Instructions: "How strongly is redness supported?", Criteria: []Criterion{{Description: "Unsupported"}, {Description: "Partly supported"}, {Description: "Supported"}}}, {Type: Noul, Instructions: "Is the bicycle red?"}}
	for i, q := range questions {
		ids, markers, err := BuildSequence(tok, "The bicycle is red.", q, c.MaxLen, c.HeadMaxLen, false)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, f.Items[i].IDs) || !slices.Equal(markers, f.Items[i].Markers) {
			t.Fatalf("sequence%d differs ids=%v markers=%v", i, ids, markers)
		}
	}
	ordered := []NamedQuestion{{ID: "color", Question: questions[0]}, {ID: "support", Question: questions[1]}, {ID: "is_red", Question: questions[2]}}
	response, err := m.SystemOne(tok, "The bicycle is red.", ordered, c)
	if err != nil {
		t.Fatal(err)
	}
	if response.Model != f.Response.Model || response.Usage != f.Response.Usage {
		t.Fatalf("response metadata=%+v want=%+v", response, f.Response)
	}
	for _, named := range ordered {
		got, want := response.Answers[named.ID], f.Response.Answers[named.ID]
		if got.Type != want.Type || got.Choice != want.Choice || math.Abs(float64(got.Score-want.Score)) > 2e-4 || math.Abs(float64(got.Noul-want.Noul)) > 2e-4 || math.Abs(float64(got.Confidence-want.Confidence)) > 2e-4 || math.Abs(float64(got.Action.ActProbability-want.Action.ActProbability)) > 2e-4 || !maps.Equal(got.Legend, want.Legend) {
			t.Fatalf("response %s=%+v want=%+v", named.ID, got, want)
		}
		for key, p := range want.Probabilities {
			if math.Abs(float64(got.Probabilities[key]-p)) > 2e-4 {
				t.Fatalf("response %s probability %s=%g want=%g", named.ID, key, got.Probabilities[key], p)
			}
		}
	}
	for row := range f.Input {
		mask := make([]bool, len(f.Mask[row]))
		for i, v := range f.Mask[row] {
			mask[i] = v != 0
		}
		markers := []int{}
		for i, v := range f.Markers[row] {
			if f.MarkerMask[row][i] {
				markers = append(markers, v)
			}
		}
		s, e := m.NewSession(len(f.Input[row]), len(markers))
		if e != nil {
			t.Fatal(e)
		}
		logits := make([]float32, len(markers))
		actions := make([]float32, m.actions)
		probs := make([]float32, len(markers))
		if _, e = s.ForwardInto(logits, actions, probs, f.Input[row], mask, markers, QuestionType(f.QTypes[row])); e != nil {
			t.Fatal(e)
		}
		cmp(t, "released logits", logits, f.Logits[row][:len(markers)])
		cmp(t, "released actions", actions, f.Actions[row])
		answer, err := m.answerFromOutput(Output{Logits: append([]float32(nil), f.Logits[row][:len(markers)]...), ActionLogits: append([]float32(nil), f.Actions[row]...)}, questions[row])
		if err != nil {
			t.Fatal(err)
		}
		keys := []string{"color", "support", "is_red"}
		want := f.Response.Answers[keys[row]]
		if answer.Type != want.Type || answer.Choice != want.Choice || answer.Score != want.Score || answer.Noul != want.Noul || answer.Confidence != want.Confidence || answer.Action != want.Action || !maps.Equal(answer.Probabilities, want.Probabilities) || !maps.Equal(answer.Legend, want.Legend) {
			t.Fatalf("answer%d=%+v want=%+v", row, answer, want)
		}
	}
}
