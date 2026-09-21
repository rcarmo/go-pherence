package decider

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

type fixture struct {
	SourcePin     string `json:"source_pin"`
	ModelPin      string `json:"model_pin"`
	BasePin       string `json:"base_pin"`
	RenderedState string `json:"rendered_state"`
	LabelIDs      []int  `json:"label_ids"`
	Prompts       []struct {
		Text          string
		IDs           []int
		NOpts         int `json:"nopts"`
		Logits        []float32
		Probabilities []float32
	}
}

func released(t *testing.T) fixture {
	t.Helper()
	b, err := os.ReadFile("testdata/released.json")
	if err != nil {
		t.Fatal(err)
	}
	var f fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestReleasedContract(t *testing.T) {
	f := released(t)
	if f.SourcePin != SourcePin || f.ModelPin != ModelPin || f.BasePin != BasePin || len(f.LabelIDs) != MaxChoice || len(f.Prompts) != 5 {
		t.Fatalf("fixture pins/shape mismatch")
	}
	state, err := RenderState(Object{{Name: "ticket", Value: "I was charged twice and want a refund."}, {Name: "priority", Value: 2}})
	if err != nil || state != f.RenderedState {
		t.Fatalf("state=%q err=%v", state, err)
	}
	for _, p := range f.Prompts {
		if !strings.HasPrefix(p.Text, "Context:\n"+state+"\n\nQuestion: ") || !strings.HasSuffix(p.Text, "\nAnswer: (") {
			t.Fatalf("bad released prompt %q", p.Text)
		}
	}
}

func TestSchemaAndAssembly(t *testing.T) {
	questions := []NamedQuestion{
		{ID: "team", Question: Question{Type: Choice, Instructions: "Which team?", Criteria: []Criterion{{Name: "billing", Description: "charges"}, {Name: "technical", Description: map[string]any{"what": "bugs"}}, {Name: "other"}}}},
		{ID: "refund", Question: Question{Type: Noul, Instructions: "Refund?"}},
		{ID: "mood", Question: Question{Type: Score, Instructions: "How angry?", Levels: []any{"0: calm", "annoyed", "furious"}}},
	}
	rendered := make([]renderedQuestion, len(questions))
	for i := range questions {
		var err error
		rendered[i], err = renderQuestion(questions[i].Question)
		if err != nil {
			t.Fatal(err)
		}
	}
	rows := planRows(rendered)
	if len(rows) != 5 || rows[2].question != "How angry?\nProposed answer: calm\nDoes the proposed answer fit?" {
		t.Fatalf("rows=%+v", rows)
	}
	p := [][]float32{{.8, .1, .1}, {.05, .95}, {.9, .1}, {.5, .5}, {.6, .4}}
	a, err := assemble(rendered, rows, p)
	if err != nil {
		t.Fatal(err)
	}
	if a[0].Choice != "billing" || a[1].Noul != .95 || a[2].Score != 1.3 || a[2].FitMass != 1 {
		t.Fatalf("answers=%+v", a)
	}
	if math.Abs(float64(certainty([]float32{1.0 / 3, 1.0 / 3, 1.0 / 3}))) > 1e-6 {
		t.Fatal("flat certainty")
	}
	list, err := renderQuestion(Question{Type: Score, Instructions: "q", Levels: []any{"low", "high"}, Listwise: true})
	if err != nil {
		t.Fatal(err)
	}
	la, err := formatAnswer(list, []float32{.25, .75})
	if err != nil || la.Score != .75 {
		t.Fatal(la, err)
	}
	levels, err := SortedLegend(map[string]any{"2": "high", "0": "low", "1": "mid"})
	if err != nil || levels[1] != "mid" {
		t.Fatal(levels, err)
	}
	if _, err = SortedLegend(map[string]any{"x": "bad"}); err == nil {
		t.Fatal("legend")
	}
}

func TestStatePromptAndAdmission(t *testing.T) {
	long := make([]any, 9)
	for i := range long {
		long[i] = map[string]any{"x": i}
	}
	text, err := RenderState(map[string]any{"items": long})
	if err != nil || !strings.Contains(text, `"_index": 7`) {
		t.Fatalf("state=%s err=%v", text, err)
	}
	if got, _ := RenderState("plain"); got != "plain" {
		t.Fatal(got)
	}
	p, err := BuildPrompt("state", "question", []string{"a", "b"})
	if err != nil || p != "Context:\nstate\n\nQuestion: question\nOptions:\n(A) a\n(B) b\nAnswer: (" {
		t.Fatalf("prompt=%q err=%v", p, err)
	}
	for _, tc := range []struct {
		state, q string
		opts     []string
	}{{"", "q", []string{"a", "b"}}, {"s", "", []string{"a", "b"}}, {"s", "q", []string{"a"}}, {"s", "q", []string{"a", ""}}} {
		if _, err := BuildPrompt(tc.state, tc.q, tc.opts); err == nil {
			t.Fatal("expected prompt rejection")
		}
	}
	bad := []Question{{Type: Choice, Instructions: "q", Criteria: []Criterion{{Name: "a"}}}, {Type: Choice, Instructions: "q", Criteria: []Criterion{{Name: "a"}, {Name: "a"}}}, {Type: Score, Instructions: "q", Levels: []any{"one"}}, {Type: Noul, Instructions: "q", Criteria: []Criterion{{Name: "maybe"}, {Name: "true"}}}, {Type: "x", Instructions: "q"}, {Type: Choice, Criteria: []Criterion{{Name: "a"}, {Name: "b"}}}}
	for _, q := range bad {
		if _, err := renderQuestion(q); err == nil {
			t.Fatalf("accepted %+v", q)
		}
	}
}

func tinyDir(t testing.TB) string {
	t.Helper()
	dir := t.TempDir()
	cfg := `{"model_type":"qwen3_5_text","architectures":["Qwen3_5ForCausalLM"],"dtype":"float32","hidden_size":4,"vocab_size":255,"intermediate_size":6,"num_hidden_layers":1,"mtp_num_hidden_layers":0,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"max_position_embeddings":32,"rope_theta":10000,"partial_rotary_factor":1,"linear_conv_kernel_dim":3,"linear_key_head_dim":2,"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_value_head_dim":2,"full_attention_interval":1,"layer_types":["full_attention"]}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "decider_config.json"), []byte(`{"temperature":1.03,"temperature_schema_first":1.03,"version":"0.8b-v1","base":"Qwen/Qwen3.5-0.8B-Base","max_options":255,"max_state_tokens":32768,"isolated_levels":true}`), 0600); err != nil {
		t.Fatal(err)
	}
	ones := func(shape ...int) checkpoint.Tensor {
		n := 1
		for _, v := range shape {
			n *= v
		}
		d := make([]float32, n)
		for i := range d {
			d[i] = 1
		}
		return checkpoint.Tensor{Shape: shape, Data: d}
	}
	zeros := func(shape ...int) checkpoint.Tensor { x := ones(shape...); clear(x.Data); return x }
	p := "model.layers.0."
	ts := map[string]checkpoint.Tensor{p + "input_layernorm.weight": zeros(4), p + "post_attention_layernorm.weight": zeros(4), p + "self_attn.q_proj.weight": zeros(8, 4), p + "self_attn.k_proj.weight": zeros(2, 4), p + "self_attn.v_proj.weight": zeros(2, 4), p + "self_attn.o_proj.weight": zeros(4, 4), p + "self_attn.q_norm.weight": zeros(2), p + "self_attn.k_norm.weight": zeros(2), p + "mlp.gate_proj.weight": zeros(6, 4), p + "mlp.up_proj.weight": zeros(6, 4), p + "mlp.down_proj.weight": zeros(4, 6), "model.language_model.embed_tokens.weight": ones(255, 4), "model.language_model.norm.weight": zeros(4)}
	if err := checkpoint.Save(filepath.Join(dir, "model.safetensors"), &checkpoint.Checkpoint{FormatVersion: 1, Tensors: ts}); err != nil {
		t.Fatal(err)
	}
	var vocab strings.Builder
	vocab.WriteString(`{"model":{"vocab":{`)
	for i := 0; i < MaxChoice; i++ {
		if i > 0 {
			vocab.WriteByte(',')
		}
		fmt.Fprintf(&vocab, "%q:%d", labelName(i), i+32)
	}
	vocab.WriteString(`},"merges":null},"added_tokens":[]}`)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(vocab.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"added_tokens_decoder":{}}`), 0600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestLoadAndSystemOneTiny(t *testing.T) {
	dir := tinyDir(t)
	r, err := LoadFixture(dir, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if ids, err := r.LabelTokenIDs(3); err != nil || !slices.Equal(ids, []int{32, 33, 34}) {
		t.Fatal(ids, err)
	}
	if _, err := r.LabelTokenIDs(0); err == nil {
		t.Fatal("label count")
	}
	if _, err := r.TokenizePrompt(""); err == nil {
		t.Fatal("empty prompt")
	}
	opts := make([]string, 11)
	for i := range opts {
		opts[i] = fmt.Sprintf("o%d", i)
	}
	wide, err := r.BuildPromptIDs("x", "q", opts)
	if err != nil || len(wide) == 0 {
		t.Fatal(wide, err)
	}
	res, err := r.SystemOne("x", []NamedQuestion{{ID: "q", Question: Question{Type: Choice, Instructions: "q", Criteria: []Criterion{{Name: "a"}, {Name: "b"}}}}, {ID: "bool", Question: Question{Type: Noul, Instructions: "yes?"}}, {ID: "score", Question: Question{Type: Score, Instructions: "level?", Levels: []any{"low", "high"}}}})
	if err != nil || len(res.Answers) != 3 || res.Model != ModelID || res.Usage.InputTokens < 1 {
		t.Fatal(res, err)
	}
	prompt, _ := BuildPrompt("x", "q", []string{"a", "b"})
	if logits, ids, err := r.ScorePrompt(prompt, 2); err != nil || len(logits) != 2 || len(ids) == 0 {
		t.Fatal(logits, ids, err)
	}
	if _, err = r.SystemOne("x", []NamedQuestion{{ID: "q", Question: Question{Type: Choice, Instructions: "q", Criteria: []Criterion{{Name: "a"}, {Name: "b"}}}}, {ID: "q", Question: Question{Type: Noul, Instructions: "q"}}}); err == nil {
		t.Fatal("duplicate ID")
	}
	if _, err = r.SystemOne("", []NamedQuestion{{ID: "q", Question: Question{Type: Noul, Instructions: "q"}}}); err == nil {
		t.Fatal("empty state")
	}
	if _, err = r.SystemOne("x", nil); err == nil {
		t.Fatal("empty questions")
	}
	if _, err := LoadFixture(dir, 0); err == nil {
		t.Fatal("limit")
	}
	missing := t.TempDir()
	if _, err := LoadFixture(missing, 32); err == nil {
		t.Fatal("missing config")
	}
	badJSON := t.TempDir()
	os.WriteFile(filepath.Join(badJSON, "decider_config.json"), []byte("{"), 0600)
	if _, err := LoadFixture(badJSON, 32); err == nil {
		t.Fatal("bad config")
	}
	os.WriteFile(filepath.Join(dir, "decider_config.json"), []byte(`{}`), 0600)
	if _, err := LoadFixture(dir, 32); err == nil {
		t.Fatal("bad release contract")
	}
	if _, err := Load(tinyDir(t), 32); err == nil {
		t.Fatal("unreleased geometry")
	}
}

func TestAPIErrorPaths(t *testing.T) {
	if _, e := (&Runtime{}).SystemOne("x", nil); e == nil {
		t.Fatal("empty")
	}
	q := renderedQuestion{typeName: Choice, options: []string{"a", "b"}, names: []any{"a", "b"}}
	if _, e := assemble([]renderedQuestion{q}, nil, [][]float32{{.5, .5}}); e == nil {
		t.Fatal("rows")
	}
	if _, e := assemble([]renderedQuestion{q}, []scoringRow{{owner: 2, options: []string{"a", "b"}}}, [][]float32{{.5, .5}}); e == nil {
		t.Fatal("owner")
	}
	sq := renderedQuestion{typeName: Score, options: []string{"0: a", "1: b"}, names: []any{0, 1}, legend: []string{"a", "b"}}
	if _, e := assemble([]renderedQuestion{sq}, []scoringRow{{owner: 0, level: 3, options: []string{"no", "yes"}}}, [][]float32{{.5, .5}}); e == nil {
		t.Fatal("level")
	}
	if _, e := formatAnswer(renderedQuestion{typeName: "x", options: []string{"a", "b"}}, []float32{.5, .5}); e == nil {
		t.Fatal("type")
	}
	if got := certainty([]float32{1}); got != 1 {
		t.Fatal(got)
	}
	dup := Object{{Name: "x", Value: 1}, {Name: "x", Value: 2}}
	if _, e := dup.MarshalJSON(); e == nil {
		t.Fatal("duplicate object")
	}
}

func TestRuntimeHelpers(t *testing.T) {
	r := &Runtime{embedding: rawTensor{dtype: "F32", shape: []int{2, 2}, raw: []byte{0, 0, 0x80, 0x3f, 0, 0, 0, 0x40, 0, 0, 0x40}}}
	row, err := r.embeddingRow(0)
	if err != nil || !slices.Equal(row, []float32{1, 2}) {
		t.Fatal(row, err)
	}
	v, err := r.embeddingRowDot(0, []float32{1, 1})
	if err != nil || v != 3 {
		t.Fatal(v, err)
	}
	if _, err = r.embeddingRow(-1); err == nil {
		t.Fatal("bad row")
	}
	if _, err = r.embeddingRowDot(2, []float32{1, 1}); err == nil {
		t.Fatal("bad dot")
	}
	bf := &Runtime{embedding: rawTensor{dtype: "BF16", shape: []int{1, 2}, raw: []byte{0x80, 0x3f, 0, 0x40}}}
	if got, e := bf.embeddingRow(0); e != nil || got[0] != 1 || got[1] != 2 {
		t.Fatal(got, e)
	}
	if v, e := bf.embeddingRowDot(0, []float32{1, 1}); e != nil || v != 3 {
		t.Fatal(v, e)
	}
	bad := &Runtime{embedding: rawTensor{dtype: "X", shape: []int{1, 2}, raw: make([]byte, 8)}}
	if _, e := bad.embeddingRow(0); e == nil {
		t.Fatal("dtype")
	}
	if _, e := bad.embeddingRowDot(0, []float32{1, 1}); e == nil {
		t.Fatal("dtype dot")
	}
	if text, err := textValue(map[string]any{"what": "bugs", "not_for": "delivery"}); err != nil || text != `{"not_for": "delivery", "what": "bugs"}` {
		t.Fatal(text, err)
	}
	p, err := softmax([]float32{1, 2}, 1)
	if err != nil || p[1] <= p[0] {
		t.Fatal(p, err)
	}
	if _, err = softmax([]float32{1}, 0); err == nil {
		t.Fatal("temperature")
	}
	if _, err = softmax([]float32{float32(math.NaN()), 1}, 1); err == nil {
		t.Fatal("nan")
	}
	for _, tc := range []struct {
		q renderedQuestion
		p []float32
	}{{renderedQuestion{typeName: Choice, options: []string{"a", "b"}, names: []any{"a", "b"}}, []float32{1}}, {renderedQuestion{typeName: Choice, options: []string{"a", "b"}, names: []any{"a", "b"}}, []float32{0, 0}}, {renderedQuestion{typeName: Choice, options: []string{"a", "b"}, names: []any{"a", "b"}}, []float32{float32(math.NaN()), 1}}} {
		if _, e := formatAnswer(tc.q, tc.p); e == nil {
			t.Fatal("bad probabilities")
		}
	}
}

func BenchmarkTinySystemOne(b *testing.B) {
	dir := tinyDir(b)
	r, err := LoadFixture(dir, 32)
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	qs := []NamedQuestion{{ID: "q", Question: Question{Type: Choice, Instructions: "q", Criteria: []Criterion{{Name: "a"}, {Name: "b"}}}}}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := r.SystemOne("x", qs); err != nil {
			b.Fatal(err)
		}
	}
}

func TestReleasedRuntimeParity(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_DECIDER_MODEL")
	if dir == "" {
		t.Skip("set released Decider model")
	}
	r, err := Load(dir, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	f := released(t)
	if !slices.Equal(r.labels, f.LabelIDs) {
		t.Fatal("label IDs")
	}
	for i, p := range f.Prompts {
		logits, ids, err := r.ScorePrompt(p.Text, p.NOpts)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, p.IDs) {
			t.Fatalf("prompt %d token mismatch", i)
		}
		for j := range logits {
			if math.Abs(float64(logits[j]-p.Logits[j])) > .12 {
				t.Fatalf("prompt %d logit %d got %.5f want %.5f", i, j, logits[j], p.Logits[j])
			}
		}
	}
}
