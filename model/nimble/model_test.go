package nimble

import (
	"encoding/binary"
	"encoding/json"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
	"github.com/rcarmo/go-pherence/tensor"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func loadReleased(t *testing.T) struct {
	SourcePin string `json:"source_pin"`
	ModelPin  string `json:"model_pin"`
	BasePin   string `json:"base_pin"`
	Context   string `json:"context"`
	Schema    map[string]struct {
		Type               string            `json:"type"`
		Choices            []any             `json:"choices"`
		Description        string            `json:"description"`
		ChoiceDescriptions map[string]string `json:"choice_descriptions"`
	} `json:"schema"`
	Names   []string `json:"names"`
	Choices [][]any  `json:"choices"`
	Rows    []struct {
		IDs           []int     `json:"ids"`
		CandidateIDs  []int     `json:"candidate_ids"`
		Logits        []float32 `json:"logits"`
		Probabilities []float32 `json:"probabilities"`
	} `json:"rows"`
} {
	t.Helper()
	var f struct {
		SourcePin string `json:"source_pin"`
		ModelPin  string `json:"model_pin"`
		BasePin   string `json:"base_pin"`
		Context   string `json:"context"`
		Schema    map[string]struct {
			Type               string            `json:"type"`
			Choices            []any             `json:"choices"`
			Description        string            `json:"description"`
			ChoiceDescriptions map[string]string `json:"choice_descriptions"`
		} `json:"schema"`
		Names   []string `json:"names"`
		Choices [][]any  `json:"choices"`
		Rows    []struct {
			IDs           []int     `json:"ids"`
			CandidateIDs  []int     `json:"candidate_ids"`
			Logits        []float32 `json:"logits"`
			Probabilities []float32 `json:"probabilities"`
		} `json:"rows"`
	}
	b, e := os.ReadFile("testdata/released.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	return f
}
func TestReleasedContract(t *testing.T) {
	f := loadReleased(t)
	if f.SourcePin != SourcePin || f.ModelPin != ModelPin || f.BasePin != BasePin {
		t.Fatal("pins")
	}
	fields := []Field{{Name: "priority", Type: "enum", Description: "Urgency based on current business impact.", Choices: []any{"HIGH", "LOW"}, ChoiceDescriptions: map[string]string{"HIGH": "A critical business operation is currently blocked.", "LOW": "An optional enhancement with no current business impact."}}, {Name: "requires_review", Type: "boolean", Description: "Whether customers are unable to complete a purchase."}}
	prompts, e := RenderPrompts(f.Context, fields)
	if e != nil {
		t.Fatal(e)
	}
	if len(prompts) != 2 {
		t.Fatal(len(prompts))
	}
	for i, p := range prompts {
		if p.Name != f.Names[i] || !strings.Contains(p.Text, "Requested field: \""+p.Name+"\"") {
			t.Fatalf("prompt %d", i)
		}
		expected := f.Rows[i].IDs
		tokPath := os.Getenv("GO_PHERENCE_NIMBLE_TOKENIZER")
		if tokPath != "" {
			tok, err := tokenizer.LoadWithConfig(filepath.Dir(tokPath))
			if err != nil {
				t.Fatal(err)
			}
			if ids := tok.Encode(p.Text); !slices.Equal(ids, expected) {
				t.Fatalf("prompt token ids %s differ: got %d want %d", p.Name, len(ids), len(expected))
			}
		}
	}
	candidateIDs, logits := [][]int{}, [][]float32{}
	for _, row := range f.Rows {
		candidateIDs = append(candidateIDs, row.CandidateIDs)
		logits = append(logits, row.Logits)
	}
	got, e := Summarize(f.Context, "bespokelabs/Bespoke-Nimble-9B", ModelPin, fields, candidateIDs, logits, 1)
	if e != nil {
		t.Fatal(e)
	}
	if got.Output["priority"] != "HIGH" || got.Output["requires_review"] != true {
		t.Fatal(got.Output)
	}
	for i, name := range f.Names {
		choices := choicesFor(fields[i])
		for j, v := range f.Rows[i].Probabilities {
			key := choiceKey(choices[j])
			if math.Abs(float64(got.Fields[name].Scores[key]-v)) > 2e-6 {
				t.Fatalf("%s %s=%g want %g", name, key, got.Fields[name].Scores[key], v)
			}
		}
	}
}
func tinyRuntime() *Runtime {
	meta := loaderconfig.QwenNativeMTPMetadata{HiddenSize: 4, VocabSize: 4, IntermediateSize: 6, NumHiddenLayers: 1, LayerTypes: []string{"full_attention"}, NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 2, MaxPositionEmbeddings: 32, PartialRotaryFactor: 1, RopeTheta: 10000, ZeroCenteredRMSNorm: true}
	layer := &qwen.Qwen35FullAttentionLayer{InputNorm: tensor.Zeros([]int{4}), PostNorm: tensor.Zeros([]int{4}), QW: tensor.Zeros([]int{8, 4}), KW: tensor.Zeros([]int{2, 4}), VW: tensor.Zeros([]int{2, 4}), OW: tensor.Zeros([]int{4, 4}), QNorm: tensor.Zeros([]int{2}), KNorm: tensor.Zeros([]int{2}), GateW: tensor.Zeros([]int{6, 4}), UpW: tensor.Zeros([]int{6, 4}), DownW: tensor.Zeros([]int{4, 6})}
	base := &qwen.Qwen35BaseModel{Layers: []qwen.Qwen35BaseLayer{{Kind: qwen.Qwen35FullAttentionLayerKind, Full: layer}}}
	emb := f32bytes(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
	lm := f32bytes(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
	tok := &tokenizer.Tokenizer{Vocab: map[string]int{"x": 1, "A": 2, "B": 3}, InvVocab: map[int]string{1: "x", 2: "A", 3: "B"}, AddedSpecial: map[string]int{}}
	return &Runtime{Bundle: &qwen.Qwen35NativeMTPBundle{Meta: meta, Base: base}, Tokenizer: tok, embedding: rawTensor{raw: emb, dtype: "F32", shape: []int{4, 4}}, norm: make([]float32, 4), lm: rawTensor{raw: lm, dtype: "F32", shape: []int{4, 4}}, rope: qwen.NewQwen35RoPEFreqs(meta, 32), maxInput: 8}
}
func writeTinyMergedModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	cfg := `{"model_type":"qwen3_5_text","dtype":"float32","hidden_size":4,"vocab_size":8,"intermediate_size":6,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"max_position_embeddings":32,"rope_theta":10000,"partial_rotary_factor":1,"linear_conv_kernel_dim":3,"linear_key_head_dim":2,"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_value_head_dim":2,"full_attention_interval":1,"layer_types":["full_attention"]}`
	if e := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0600); e != nil {
		t.Fatal(e)
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
	ts := map[string]checkpoint.Tensor{p + "input_layernorm.weight": zeros(4), p + "post_attention_layernorm.weight": zeros(4), p + "self_attn.q_proj.weight": zeros(8, 4), p + "self_attn.k_proj.weight": zeros(2, 4), p + "self_attn.v_proj.weight": zeros(2, 4), p + "self_attn.o_proj.weight": zeros(4, 4), p + "self_attn.q_norm.weight": zeros(2), p + "self_attn.k_norm.weight": zeros(2), p + "mlp.gate_proj.weight": zeros(6, 4), p + "mlp.up_proj.weight": zeros(6, 4), p + "mlp.down_proj.weight": zeros(4, 6), "model.language_model.embed_tokens.weight": ones(8, 4), "model.language_model.norm.weight": zeros(4), "lm_head.weight": ones(8, 4)}
	if e := checkpoint.Save(filepath.Join(dir, "model.safetensors"), &checkpoint.Checkpoint{FormatVersion: 1, Tensors: ts}); e != nil {
		t.Fatal(e)
	}
	tok := `{"model":{"vocab":{"x":0,"A":1,"B":2,"<think>":3,"</think>":4,"<|im_start|>":5,"<|im_end|>":6},"merges":null},"added_tokens":[{"id":5,"content":"<|im_start|>","special":true},{"id":6,"content":"<|im_end|>","special":true},{"id":3,"content":"<think>","special":false},{"id":4,"content":"</think>","special":false}]}`
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tok), 0600)
	os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"added_tokens_decoder":null}`), 0600)
	return dir
}
func TestLoadMergedTiny(t *testing.T) {
	dir := writeTinyMergedModel(t)
	r, e := LoadMerged(dir, 32)
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	got, e := r.ScoreTokenIDs([]int{0}, []int{1, 2})
	if e != nil || len(got) != 2 {
		t.Fatal(got, e)
	}
	if _, e := LoadMerged(dir, 2049); e == nil {
		t.Fatal("oversized max input")
	}
	os.Remove(filepath.Join(dir, "model.safetensors"))
	if _, e := LoadMerged(dir, 32); e == nil {
		t.Fatal("missing weights")
	}
}

func TestRuntimeScoreAssembly(t *testing.T) {
	r := tinyRuntime()
	fields := []Field{{Name: "f", Type: "enum", Description: "d", Choices: []any{"a", "b"}}}
	prompts, e := RenderPrompts("x", fields)
	if e != nil {
		t.Fatal(e)
	}
	r.maxInput = 2048
	for _, code := range []string{"A", "B"} {
		if _, ok := r.Tokenizer.Vocab[code]; !ok {
			t.Fatal(code)
		}
	}
	result, e := r.Score("x", fields, 1)
	if e != nil {
		t.Fatal(e)
	}
	if result.Model != "bespokelabs/Bespoke-Nimble-9B" || len(prompts) != 1 || len(result.Fields) != 1 {
		t.Fatal(result)
	}
}

func BenchmarkTinyRuntimeScoreTokenIDs(b *testing.B) {
	r := tinyRuntime()
	b.ReportAllocs()
	for b.Loop() {
		if _, e := r.ScoreTokenIDs([]int{0, 1}, []int{2, 3}); e != nil {
			b.Fatal(e)
		}
	}
}
func TestTinyRuntimeExecution(t *testing.T) {
	r := tinyRuntime()
	ids, candidates, e := r.Tokenize([]Prompt{{Name: "p", Text: "x", Choices: []any{"a", "b"}}})
	if e != nil || !slices.Equal(ids[0], []int{1}) || !slices.Equal(candidates[0], []int{2, 3}) {
		t.Fatalf("ids=%v candidates=%v err=%v", ids, candidates, e)
	}
	for split := 0; split <= 2; split++ {
		got, e := r.ScoreTokenIDsSplit([]int{0, 1}, []int{2, 3}, split)
		if e != nil || len(got) != 2 || !isFinite(got[0]) {
			t.Fatalf("split%d got=%v err=%v", split, got, e)
		}
	}
	if _, e := r.ScoreTokenIDsSplit(nil, []int{2}, 0); e == nil {
		t.Fatal("empty ids")
	}
	if _, e := r.ScoreTokenIDsSplit([]int{0}, []int{2}, 2); e == nil {
		t.Fatal("bad split")
	}
	if _, e := r.ScoreTokenIDsSplit([]int{0}, nil, 1); e == nil {
		t.Fatal("empty candidates")
	}
	if _, e := r.ScoreTokenIDsSplit([]int{99}, []int{2}, 1); e == nil {
		t.Fatal("bad token")
	}
	if _, e := r.projectCandidates(nil, []int{2}); e == nil {
		t.Fatal("bad hidden")
	}
	if _, _, e := r.forwardTokens(nil, qwen.Qwen35BaseForwardState{}); e == nil {
		t.Fatal("empty forward")
	}
	if _, e := r.embeddingRow(99); e == nil {
		t.Fatal("bad token")
	}
	if _, e := r.lmRowDot(99, make([]float32, 4)); e == nil {
		t.Fatal("bad row")
	}
}

func TestRuntimeHelperAdmission(t *testing.T) {
	if _, e := LoadMerged("missing", 0); e == nil {
		t.Fatal("bad capacity")
	}
	if _, e := LoadMerged("missing", 1); e == nil {
		t.Fatal("missing model")
	}
	if e := (*Runtime)(nil).Close(); e != nil {
		t.Fatal(e)
	}
	r := &Runtime{maxInput: 2, embedding: rawTensor{shape: []int{2, 2}, dtype: "F32", raw: append(append([]byte{}, f32bytes(1, 2)...), f32bytes(3, 4)...)}, lm: rawTensor{shape: []int{2, 2}, dtype: "F32", raw: append(append([]byte{}, f32bytes(1, 0)...), f32bytes(0, 1)...)}}
	if _, e := r.embeddingRow(-1); e == nil {
		t.Fatal("negative token")
	}
	if got, e := r.embeddingRow(1); e != nil || !slices.Equal(got, []float32{3, 4}) {
		t.Fatalf("embedding=%v err=%v", got, e)
	}
	if v, e := r.lmRowDot(0, []float32{2, 3}); e != nil || v != 2 {
		t.Fatalf("lm=%g err=%v", v, e)
	}
	bf := &Runtime{embedding: rawTensor{shape: []int{1, 2}, dtype: "BF16", raw: []byte{0x80, 0x3f, 0, 0x40}}, lm: rawTensor{shape: []int{1, 2}, dtype: "BF16", raw: []byte{0x80, 0x3f, 0, 0x40}}}
	if got, e := bf.embeddingRow(0); e != nil || got[0] != 1 || got[1] != 2 {
		t.Fatal(got, e)
	}
	if v, e := bf.lmRowDot(0, []float32{1, 1}); e != nil || v != 3 {
		t.Fatal(v, e)
	}
	r.embedding.dtype = "bad"
	if _, e := r.embeddingRow(0); e == nil {
		t.Fatal("bad embedding dtype")
	}
	r.lm.dtype = "bad"
	if _, e := r.lmRowDot(0, []float32{1, 2}); e == nil {
		t.Fatal("bad lm dtype")
	}
	if _, _, e := r.Tokenize(nil); e == nil {
		t.Fatal("nil tokenizer")
	}
	r.Tokenizer = &tokenizer.Tokenizer{Vocab: map[string]int{"x": 0, "A": 1, "B": 2}, InvVocab: map[int]string{0: "x", 1: "A", 2: "B"}, AddedSpecial: map[string]int{}}
	r.maxInput = 1
	if _, _, e := r.Tokenize([]Prompt{{Name: "p", Text: "xx", Choices: []any{"a", "b"}}}); e == nil {
		t.Fatal("long prompt")
	}
	r.maxInput = 8
	if _, _, e := r.Tokenize([]Prompt{{Name: "p", Text: "x", Choices: []any{"a", "b", "c"}}}); e == nil {
		t.Fatal("non-token code")
	}
	r.Tokenizer = nil
	dir := t.TempDir()
	if e := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"vocab":{"<think>":1,"</think>":2},"merges":null},"added_tokens":[{"id":1,"content":"<think>","special":false},{"id":2,"content":"</think>","special":false}]}`), 0600); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"added_tokens_decoder":null}`), 0600); e != nil {
		t.Fatal(e)
	}
	tok, e := loadTokenizer(dir)
	if e != nil || tok.AddedTokens["<think>"] != 1 {
		t.Fatalf("fallback tokenizer=%v err=%v", tok, e)
	}
	valid := `{"added_tokens_decoder":{"1":{"content":"<think>","special":false},"2":{"content":"</think>","special":false}}}`
	if e = os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(valid), 0600); e != nil {
		t.Fatal(e)
	}
	if tok, e = loadTokenizer(dir); e != nil || tok.AddedTokens["</think>"] != 2 {
		t.Fatalf("sidecar tokenizer=%v err=%v", tok, e)
	}
	os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte("{"), 0600)
	if _, e = loadTokenizer(dir); e == nil {
		t.Fatal("bad sidecar")
	}
}
func f32bytes(values ...float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func TestLoadContract(t *testing.T) {
	path := filepath.Join(t.TempDir(), "contract.json")
	if e := os.WriteFile(path, []byte(`{"task":"schema_candidate_classification_v1"}`), 0600); e != nil {
		t.Fatal(e)
	}
	got, e := LoadContract(path)
	if e != nil || got["task"] != "schema_candidate_classification_v1" {
		t.Fatalf("got=%v err=%v", got, e)
	}
	if _, e := LoadContract(filepath.Join(t.TempDir(), "missing")); e == nil {
		t.Fatal("missing accepted")
	}
}

func TestSchemaAdmission(t *testing.T) {
	bad := [][]Field{nil, {{Name: "", Type: "enum", Description: "x", Choices: []any{"a"}}}, {{Name: "x", Type: "bad", Description: "x"}}, {{Name: "x", Type: "enum", Description: "", Choices: []any{"a"}}}, {{Name: "x", Type: "enum", Description: "x", Choices: []any{"a", "a"}}}, {{Name: "x", Type: "enum", Description: "x", Choices: []any{true}}}, {{Name: "x", Type: "boolean", Description: "x", Choices: []any{true, true}}}, {{Name: "x", Type: "boolean", Description: "x", ChoiceDescriptions: map[string]string{"maybe": "x"}}}, {{Name: "x", Type: "enum", Description: "x", Choices: []any{"a"}, ChoiceDescriptions: map[string]string{"b": "x"}}}}
	for i, x := range bad {
		if e := ValidateSchema(x); e == nil {
			t.Fatalf("bad %d accepted", i)
		}
	}
	if _, e := RenderPrompts(" ", []Field{{Name: "x", Type: "boolean", Description: "x"}}); e == nil {
		t.Fatal("empty context")
	}
	if _, e := Summarize("x", "m", "r", []Field{{Name: "x", Type: "boolean", Description: "x"}}, [][]int{{1, 2}}, [][]float32{{0, float32(math.NaN())}}, 1); e == nil {
		t.Fatal("nan")
	}
	if _, e := Summarize("x", "m", "r", []Field{{Name: "x", Type: "boolean", Description: "x"}}, nil, nil, 1); e == nil {
		t.Fatal("rows")
	}
	if _, e := Summarize("x", "m", "r", []Field{{Name: "x", Type: "boolean", Description: "x"}}, [][]int{{1}}, [][]float32{{0}}, 1); e == nil {
		t.Fatal("candidates")
	}
	if _, e := Summarize("x", "m", "r", []Field{{Name: "x", Type: "boolean", Description: "x"}}, [][]int{{1, 2}}, [][]float32{{0, 0}}, 0); e == nil {
		t.Fatal("temperature")
	}
	many := make([]any, 27)
	for i := range many {
		many[i] = string(rune('a' + i))
	}
	if e := ValidateSchema([]Field{{Name: "x", Type: "enum", Description: "x", Choices: many}}); e == nil {
		t.Fatal("too many choices")
	}
}
func TestReleasedRuntimeParity(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_NIMBLE_MODEL")
	if dir == "" {
		t.Skip("set merged Nimble model")
	}
	f := loadReleased(t)
	r, e := LoadMerged(dir, 2048)
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	fields := []Field{{Name: "priority", Type: "enum", Description: "Urgency based on current business impact.", Choices: []any{"HIGH", "LOW"}, ChoiceDescriptions: map[string]string{"HIGH": "A critical business operation is currently blocked.", "LOW": "An optional enhancement with no current business impact."}}, {Name: "requires_review", Type: "boolean", Description: "Whether customers are unable to complete a purchase."}}
	got, e := r.Score(f.Context, fields, 1)
	if e != nil {
		t.Fatal(e)
	}
	mergedWant := [][]float32{{25.625, 20}, {21, 25.375}}
	for i, name := range f.Names {
		result := got.Fields[name]
		for j, key := range []string{choiceKey(choicesFor(fields[i])[0]), choiceKey(choicesFor(fields[i])[1])} {
			if math.Abs(float64(result.Logits[key]-mergedWant[i][j])) > .1 {
				t.Fatalf("row%d logit%d=%g want %g", i, j, result.Logits[key], mergedWant[i][j])
			}
		}
	}
	if got.Output["priority"] != "HIGH" || got.Output["requires_review"] != true {
		t.Fatal(got.Output)
	}
}

func TestPromptExactTextAndOrder(t *testing.T) {
	fields := []Field{{Name: "z", Type: "enum", Description: "d", Choices: []any{"x", "y"}}, {Name: "a", Type: "boolean", Description: "b", Choices: []any{true, false}}}
	p, e := RenderPrompts("<unsafe>", fields)
	if e != nil {
		t.Fatal(e)
	}
	if p[0].Name != "z" || p[1].Name != "a" || !strings.Contains(p[0].Text, "\\u003cunsafe\\u003e") || !slices.Equal(p[1].Choices, []any{true, false}) {
		t.Fatal(p)
	}
}
