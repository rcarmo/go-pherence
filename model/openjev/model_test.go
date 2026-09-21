package openjev

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
	ModelPin    string `json:"model_pin"`
	ModelSHA256 string `json:"model_sha256"`
	Template    string `json:"template"`
	Labels      []string
	Pairs       []struct {
		Premise, Hypothesis, Text string
		IDs                       []int
		Logits, Probabilities     []float32
	}
}

func released(t testing.TB) fixture {
	t.Helper()
	b, e := os.ReadFile("testdata/released.json")
	if e != nil {
		t.Fatal(e)
	}
	var f fixture
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	return f
}
func TestReleasedContract(t *testing.T) {
	f := released(t)
	if f.ModelPin != ModelPin || f.Template != Template || !slices.Equal(f.Labels, Labels[:]) || len(f.Pairs) != 4 {
		t.Fatal("fixture")
	}
	for _, p := range f.Pairs {
		text, e := Render(p.Premise, p.Hypothesis)
		if e != nil || text != p.Text {
			t.Fatal(text, e)
		}
	}
}
func TestSummarizeAndAPIAdmission(t *testing.T) {
	p, e := Summarize("p", "h", []int{1, 2}, []float32{0, 2, 1})
	if e != nil || p.Label != "entailment" || p.Probabilities[1] <= p.Probabilities[2] {
		t.Fatal(p, e)
	}
	p.Logits[0] = 9
	if p.TokenIDs[0] != 1 {
		t.Fatal("ownership")
	}
	if _, e = Summarize("p", "h", nil, []float32{1, 2}); e == nil {
		t.Fatal("width")
	}
	if _, e = Summarize("p", "h", nil, []float32{1, float32(math.NaN()), 2}); e == nil {
		t.Fatal("nan")
	}
	if _, e = Render(" ", "h"); e == nil {
		t.Fatal("premise")
	}
	if _, e = Render("p", " "); e == nil {
		t.Fatal("hypothesis")
	}
	if _, e = (&Runtime{}).Predict(nil); e == nil {
		t.Fatal("pairs")
	}
	if _, e = (&Runtime{}).Rerank("q", []string{"one"}); e == nil {
		t.Fatal("options")
	}
	if _, e = (&Runtime{}).Rerank("q", []string{"one", ""}); e == nil {
		t.Fatal("empty option")
	}
	if _, e = (&Runtime{}).Grade("", "r", "c"); e == nil {
		t.Fatal("grade")
	}
}

func tinyDir(t testing.TB) string {
	t.Helper()
	dir := t.TempDir()
	cfg := `{"architectures":["Qwen3_5ForSequenceClassification"],"model_type":"qwen3_5","nli_template":"Premise: {premise}\nHypothesis: {hypothesis}","problem_type":"single_label_classification","id2label":{"0":"contradiction","1":"entailment","2":"neutral"},"text_config":{"model_type":"qwen3_5_text","dtype":"float32","hidden_size":4,"vocab_size":8,"intermediate_size":6,"num_hidden_layers":1,"mtp_num_hidden_layers":0,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"max_position_embeddings":32,"rope_theta":10000,"partial_rotary_factor":1,"linear_conv_kernel_dim":3,"linear_key_head_dim":2,"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_value_head_dim":2,"full_attention_interval":1,"layer_types":["full_attention"]}}`
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0600)
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
	ts := map[string]checkpoint.Tensor{p + "input_layernorm.weight": zeros(4), p + "post_attention_layernorm.weight": zeros(4), p + "self_attn.q_proj.weight": zeros(8, 4), p + "self_attn.k_proj.weight": zeros(2, 4), p + "self_attn.v_proj.weight": zeros(2, 4), p + "self_attn.o_proj.weight": zeros(4, 4), p + "self_attn.q_norm.weight": zeros(2), p + "self_attn.k_norm.weight": zeros(2), p + "mlp.gate_proj.weight": zeros(6, 4), p + "mlp.up_proj.weight": zeros(6, 4), p + "mlp.down_proj.weight": zeros(4, 6), "model.language_model.embed_tokens.weight": ones(8, 4), "model.language_model.norm.weight": zeros(4), "score.weight": ones(3, 4)}
	if e := checkpoint.Save(filepath.Join(dir, "model.safetensors"), &checkpoint.Checkpoint{FormatVersion: 1, Tensors: ts}); e != nil {
		t.Fatal(e)
	}
	tok := `{"model":{"vocab":{"Premise":0,":":1,"Ġx":2,"Ċ":3,"Hypothesis":4,"Ġq":5,"Ġone":6,"Ġtwo":7},"merges":null},"added_tokens":[]}`
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tok), 0600)
	os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"added_tokens_decoder":{}}`), 0600)
	return dir
}
func TestLoadAndTinyRuntime(t *testing.T) {
	dir := tinyDir(t)
	r, e := LoadFixture(dir, 32)
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	p, e := r.Score("x", "q")
	if e != nil || len(p.Logits) != 3 {
		t.Fatal(p, e)
	}
	r.maxInput = 2
	if ids, e := r.Tokenize("x", "q"); e != nil || len(ids) != 2 {
		t.Fatal(ids, e)
	}
	r.maxInput = 32
	if _, e := r.Tokenize("", "q"); e == nil {
		t.Fatal("empty premise")
	}
	if _, e := (&Runtime{}).Tokenize("x", "q"); e == nil {
		t.Fatal("nil runtime")
	}
	if _, e := r.ScoreTokenIDs(nil); e == nil {
		t.Fatal("empty ids")
	}
	ps, e := r.Predict([][2]string{{"x", "q"}, {"x", "q"}})
	if e != nil || len(ps) != 2 {
		t.Fatal(ps, e)
	}
	rr, e := r.Rerank("x", []string{"one", "two"})
	if e != nil || rr.Index != 0 {
		t.Fatal(rr, e)
	}
	if _, e = r.Grade("x", "one", "one"); e != nil {
		t.Fatal(e)
	}
	if _, e := Load(dir, 32); e == nil {
		t.Fatal("unreleased geometry")
	}
	if _, e := LoadFixture(dir, 0); e == nil {
		t.Fatal("limit")
	}
	bad := t.TempDir()
	if _, e := LoadFixture(bad, 32); e == nil {
		t.Fatal("config")
	}
	os.WriteFile(filepath.Join(bad, "config.json"), []byte("{"), 0600)
	if _, e := LoadFixture(bad, 32); e == nil {
		t.Fatal("bad json")
	}
	badContract := tinyDir(t)
	os.WriteFile(filepath.Join(badContract, "config.json"), []byte(`{"architectures":["x"]}`), 0600)
	if _, e := LoadFixture(badContract, 32); e == nil {
		t.Fatal("bad contract")
	}
}
func TestRuntimeHelpers(t *testing.T) {
	r := &Runtime{embedding: rawTensor{dtype: "F32", shape: []int{2, 2}, raw: []byte{0, 0, 0x80, 0x3f, 0, 0, 0, 0x40, 0, 0, 0, 0x40}}, head: rawTensor{dtype: "F32", shape: []int{3, 2}, raw: make([]byte, 24)}}
	row, e := r.embeddingRow(0)
	if e != nil || !slices.Equal(row, []float32{1, 2}) {
		t.Fatal(row, e)
	}
	if _, e = r.embeddingRow(-1); e == nil {
		t.Fatal("row")
	}
	if _, e = r.headRowDot(3, []float32{1, 2}); e == nil {
		t.Fatal("head")
	}
	if _, e = r.headRowDot(0, []float32{1}); e == nil {
		t.Fatal("head width")
	}
	bf := &Runtime{embedding: rawTensor{dtype: "BF16", shape: []int{1, 2}, raw: []byte{0x80, 0x3f, 0, 0x40}}, head: rawTensor{dtype: "BF16", shape: []int{1, 2}, raw: []byte{0x80, 0x3f, 0, 0x40}}}
	if v, e := bf.headRowDot(0, []float32{1, 1}); e != nil || v != 3 {
		t.Fatal(v, e)
	}
	bad := &Runtime{embedding: rawTensor{dtype: "X", shape: []int{1, 2}, raw: make([]byte, 8)}, head: rawTensor{dtype: "X", shape: []int{1, 2}, raw: make([]byte, 8)}}
	if _, e := bad.embeddingRow(0); e == nil {
		t.Fatal("dtype")
	}
	if _, e := bad.headRowDot(0, []float32{1, 1}); e == nil {
		t.Fatal("head dtype")
	}
	r.head = rawTensor{dtype: "F32", shape: []int{3, 2}, raw: []byte{0, 0, 0x80, 0x3f, 0, 0, 0, 0x40, 0, 0, 0x40, 0, 0, 0x40, 0, 0, 0x40, 0, 0, 0x40, 0, 0, 0x40}}
	if v, e := r.headRowDot(0, []float32{1, 1}); e != nil || v != 3 {
		t.Fatal(v, e)
	}
}
func BenchmarkTinyScore(b *testing.B) {
	r, e := LoadFixture(tinyDir(b), 32)
	if e != nil {
		b.Fatal(e)
	}
	defer r.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, e := r.Score("x", "q"); e != nil {
			b.Fatal(e)
		}
	}
}
func TestCloseAndScoreErrors(t *testing.T) {
	var r *Runtime
	if e := r.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e := r.Score("p", "h"); e == nil {
		t.Fatal("nil score")
	}
	if _, e := (&Runtime{}).ScoreTokenIDs([]int{1}); e == nil {
		t.Fatal("nil ids")
	}
}

func TestLoadFailureBranches(t *testing.T) {
	dir := tinyDir(t)
	os.Remove(filepath.Join(dir, "model.safetensors"))
	if _, e := LoadFixture(dir, 32); e == nil {
		t.Fatal("weights")
	}
	dir = tinyDir(t)
	os.Remove(filepath.Join(dir, "tokenizer_config.json"))
	if _, e := LoadFixture(dir, 32); e == nil {
		t.Fatal("tokenizer")
	}
	dir = tinyDir(t)
	f, e := os.OpenFile(filepath.Join(dir, "model.safetensors"), os.O_RDWR, 0600)
	if e != nil {
		t.Fatal(e)
	}
	_ = f.Close()
	cfg, _ := os.ReadFile(filepath.Join(dir, "config.json"))
	var c map[string]any
	json.Unmarshal(cfg, &c)
	c["id2label"] = map[string]string{"0": "wrong", "1": "entailment", "2": "neutral"}
	cfg, _ = json.Marshal(c)
	os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0600)
	if _, e := LoadFixture(dir, 32); e == nil {
		t.Fatal("labels")
	}
}

func TestReleasedRuntimeParity(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_OPENJEV_MODEL")
	if dir == "" {
		t.Skip("set released OpenJEV model")
	}
	r, e := Load(dir, 4096)
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	f := released(t)
	for i, p := range f.Pairs {
		ids, e := r.Tokenize(p.Premise, p.Hypothesis)
		if e != nil || !slices.Equal(ids, p.IDs) {
			t.Fatalf("pair %d tokens err=%v", i, e)
		}
		logits, e := r.ScoreTokenIDs(ids)
		if e != nil {
			t.Fatal(e)
		}
		for j := range logits {
			if math.Abs(float64(logits[j]-p.Logits[j])) > .12 {
				t.Fatalf("pair %d logit %d got %.5f want %.5f", i, j, logits[j], p.Logits[j])
			}
		}
	}
}

var _ = fmt.Sprint
var _ = strings.Builder{}
