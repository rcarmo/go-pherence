package modernbert

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

type payload struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}
type fixture struct {
	TransformersPin string             `json:"transformers_pin"`
	ModelPin        string             `json:"model_pin"`
	Config          json.RawMessage    `json:"config"`
	Tokens          []int              `json:"tokens"`
	Mask            []int              `json:"attention_mask"`
	Tensors         map[string]payload `json:"tensors"`
	States          []payload          `json:"hidden_states"`
	Last            payload            `json:"last_hidden_state"`
}

func loadFixture(t *testing.T) (*Model, fixture) {
	t.Helper()
	b, e := os.ReadFile("testdata/tiny.json")
	if e != nil {
		t.Fatal(e)
	}
	var f fixture
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	if f.TransformersPin != "c587bc884db2c2e31fc2b8102314656b17aa07b1" || f.ModelPin != "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13" {
		t.Fatal("pins")
	}
	ts := map[string]Tensor{}
	for k, v := range f.Tensors {
		ts[k] = Tensor{Shape: v.Shape, Data: v.Data}
	}
	m, e := New(f.Config, ts)
	if e != nil {
		t.Fatal(e)
	}
	return m, f
}
func compare(t *testing.T, name string, got, want []float32, abs, rel float64) {
	t.Helper()
	bad := 0
	maxd := 0.
	for i, x := range got {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatalf("%s nonfinite at %d", name, i)
		}
		d := math.Abs(float64(x - want[i]))
		if d > maxd {
			maxd = d
		}
		if d > abs+rel*math.Abs(float64(want[i])) {
			bad++
		}
	}
	if bad > 0 {
		t.Fatalf("%s: %d mismatches max %g", name, bad, maxd)
	}
}
func TestTinyUpstreamParity(t *testing.T) {
	m, f := loadFixture(t)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	got, states, e := m.ForwardLayers(f.Tokens, mask)
	if e != nil {
		t.Fatal(e)
	}
	if len(states) != len(f.States) {
		t.Fatal("states")
	}
	for i, s := range states {
		compare(t, fmt.Sprintf("layer%d", i), s, f.States[i].Data, 3e-5, 2e-4)
	}
	compare(t, "final", got, f.Last.Data, 3e-5, 2e-4)
}
func TestAdmission(t *testing.T) {
	m, f := loadFixture(t)
	if _, e := m.Forward(nil, nil); e == nil {
		t.Fatal("empty")
	}
	if _, e := m.Forward(f.Tokens[:2], []bool{true}); e == nil {
		t.Fatal("mask")
	}
	if _, e := m.Forward([]int{999}, []bool{true}); e == nil {
		t.Fatal("token")
	}
	_ = m
}
func TestReleasedConfigDefaults(t *testing.T) {
	b, e := os.ReadFile("testdata/released-config.json")
	if e != nil {
		t.Fatal(e)
	}
	c, e := ParseConfig(b)
	if e != nil {
		t.Fatal(e)
	}
	if c.HiddenSize != 1024 || c.Layers != 28 || c.Heads != 16 || c.IntermediateSize != 2624 || len(c.LayerTypes) != 28 || c.LayerTypes[0] != "full_attention" || c.LayerTypes[1] != "sliding_attention" || c.RopeParameters["full_attention"].Theta != 160000 || c.RopeParameters["sliding_attention"].Theta != 10000 {
		t.Fatalf("released defaults %+v", c)
	}
}
func TestSessionWarmAllocations(t *testing.T) {
	m, f := loadFixture(t)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	s, e := m.NewSession(len(f.Tokens))
	if e != nil {
		t.Fatal(e)
	}
	out := make([]float32, len(f.Tokens)*m.Config.HiddenSize)
	if e = s.ForwardInto(out, f.Tokens, mask); e != nil {
		t.Fatal(e)
	}
	allocs := testing.AllocsPerRun(100, func() {
		if err := s.ForwardInto(out, f.Tokens, mask); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm ForwardInto allocations %.2f", allocs)
	}
}
func BenchmarkTinyForward(b *testing.B) {
	m, f := loadFixtureB(b)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	s, e := m.NewSession(len(f.Tokens))
	if e != nil {
		b.Fatal(e)
	}
	out := make([]float32, len(f.Tokens)*m.Config.HiddenSize)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if e = s.ForwardInto(out, f.Tokens, mask); e != nil {
			b.Fatal(e)
		}
	}
}
func loadFixtureB(tb testing.TB) (*Model, fixture) {
	tb.Helper()
	var f fixture
	b, e := os.ReadFile("testdata/tiny.json")
	if e != nil {
		tb.Fatal(e)
	}
	if e = json.Unmarshal(b, &f); e != nil {
		tb.Fatal(e)
	}
	ts := map[string]Tensor{}
	for k, v := range f.Tensors {
		ts[k] = Tensor{Shape: v.Shape, Data: v.Data}
	}
	m, e := New(f.Config, ts)
	if e != nil {
		tb.Fatal(e)
	}
	return m, f
}
func TestReleasedModelSmoke(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_MODERNBERT_MODEL")
	if dir == "" {
		t.Skip("set local released model")
	}
	m, e := Load(dir)
	if e != nil {
		t.Fatal(e)
	}
	ids := []int{50281, 14177, 50282}
	s, e := m.NewSession(len(ids))
	if e != nil {
		t.Fatal(e)
	}
	out := make([]float32, len(ids)*m.Config.HiddenSize)
	if allocs := testing.AllocsPerRun(3, func() {
		if err := s.ForwardInto(out, ids, []bool{true, true, true}); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("released warm allocations %.2f", allocs)
	}
	if e != nil {
		t.Fatal(e)
	}
	if len(out) != len(ids)*m.Config.HiddenSize {
		t.Fatal("shape")
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("nonfinite %d", i)
		}
	}
	var ref struct {
		ModelPin string             `json:"model_pin"`
		Layers   map[string]payload `json:"layers"`
		Last     payload            `json:"last_hidden_state"`
	}
	b, err := os.ReadFile("testdata/released-smoke.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelPin != "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13" {
		t.Fatal("pin")
	}
	_, states, err := m.ForwardLayers(ids, []bool{true, true, true})
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{0, 13, 27} {
		compare(t, fmt.Sprintf("released-layer%d", i), states[i+1], ref.Layers[fmt.Sprint(i)].Data, 3e-4, 2e-3)
	}
	compare(t, "released-final", out, ref.Last.Data, 3e-4, 2e-3)
}
func BenchmarkReleasedForward(b *testing.B) {
	dir := os.Getenv("GO_PHERENCE_MODERNBERT_MODEL")
	if dir == "" {
		b.Skip("set local model")
	}
	m, e := Load(dir)
	if e != nil {
		b.Fatal(e)
	}
	ids := []int{50281, 14177, 50282}
	mask := []bool{true, true, true}
	s, e := m.NewSession(len(ids))
	if e != nil {
		b.Fatal(e)
	}
	out := make([]float32, len(ids)*m.Config.HiddenSize)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if e = s.ForwardInto(out, ids, mask); e != nil {
			b.Fatal(e)
		}
	}
}
func TestConfigAndTensorRejection(t *testing.T) {
	m, f := loadFixture(t)
	_ = m
	var fields map[string]any
	if e := json.Unmarshal(f.Config, &fields); e != nil {
		t.Fatal(e)
	}
	for key, value := range map[string]any{"hidden_activation": "relu", "attention_bias": true, "layer_types": []string{"bad", "bad", "bad"}, "local_attention": 1, "num_attention_heads": 3} {
		copy := map[string]any{}
		for k, v := range fields {
			copy[k] = v
		}
		copy[key] = value
		raw, _ := json.Marshal(copy)
		ts := map[string]Tensor{}
		for n, v := range f.Tensors {
			ts[n] = Tensor{Shape: v.Shape, Data: v.Data}
		}
		if _, e := New(raw, ts); e == nil {
			t.Fatal("accepted " + key)
		}
	}
	for _, name := range []string{"embeddings.tok_embeddings.weight", "layers.0.attn.Wqkv.weight", "layers.1.attn_norm.weight", "final_norm.weight"} {
		ts := map[string]Tensor{}
		for n, v := range f.Tensors {
			ts[n] = Tensor{Shape: append([]int(nil), v.Shape...), Data: append([]float32(nil), v.Data...)}
		}
		delete(ts, name)
		if _, e := New(f.Config, ts); e == nil {
			t.Fatal("missing " + name)
		}
	}
}
func TestOwnershipAndSessionAdmission(t *testing.T) {
	m, f := loadFixture(t)
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	before, e := m.Forward(f.Tokens, mask)
	if e != nil {
		t.Fatal(e)
	}
	for _, v := range f.Tensors {
		clear(v.Data)
	}
	after, e := m.Forward(f.Tokens, mask)
	if e != nil {
		t.Fatal(e)
	}
	compare(t, "ownership", after, before, 0, 0)
	if _, e = m.NewSession(0); e == nil {
		t.Fatal("zero capacity")
	}
	s, e := m.NewSession(len(f.Tokens))
	if e != nil {
		t.Fatal(e)
	}
	if e = s.ForwardInto(make([]float32, 1), f.Tokens, mask); e == nil {
		t.Fatal("short output")
	}
	if e = s.ForwardInto(make([]float32, len(before)), f.Tokens, mask[:1]); e == nil {
		t.Fatal("mask mismatch")
	}
}
func TestLoadTinyModelDirectory(t *testing.T) {
	dir := t.TempDir()
	for src, dst := range map[string]string{"testdata/tiny-config.json": "config.json", "testdata/tiny.safetensors": "model.safetensors"} {
		b, e := os.ReadFile(src)
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(filepath.Join(dir, dst), b, 0600); e != nil {
			t.Fatal(e)
		}
	}
	m, e := Load(dir)
	if e != nil {
		t.Fatal(e)
	}
	f := fixture{}
	b, _ := os.ReadFile("testdata/tiny.json")
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	mask := make([]bool, len(f.Mask))
	for i, v := range f.Mask {
		mask[i] = v != 0
	}
	out, e := m.Forward(f.Tokens, mask)
	if e != nil {
		t.Fatal(e)
	}
	compare(t, "loaded", out, f.Last.Data, 3e-5, 2e-4)
}
func TestLoadRejections(t *testing.T) {
	if _, e := Load(t.TempDir()); e == nil {
		t.Fatal("missing config")
	}
	dir := t.TempDir()
	if e := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e := Load(dir); e == nil {
		t.Fatal("missing weights")
	}
	m, f := loadFixture(t)
	_ = m
	var fields map[string]any
	if e := json.Unmarshal(f.Config, &fields); e != nil {
		t.Fatal(e)
	}
	for _, key := range []string{"rope_parameters", "layer_types"} {
		copy := map[string]any{}
		for k, v := range fields {
			copy[k] = v
		}
		copy[key] = map[string]any{"bad": 1}
		raw, _ := json.Marshal(copy)
		ts := map[string]Tensor{}
		for n, v := range f.Tensors {
			ts[n] = Tensor{Shape: v.Shape, Data: v.Data}
		}
		if _, e := New(raw, ts); e == nil {
			t.Fatal("bad " + key)
		}
	}
}
func TestModernBERTNumberRunTokenization(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_MODERNBERT_MODEL")
	if dir == "" {
		t.Skip("set local model")
	}
	tok, e := LoadTokenizer(dir)
	if e != nil {
		t.Fatal(e)
	}
	got, e := tok.Encode(" level 0: Unsupported", false)
	if e != nil {
		t.Fatal(e)
	}
	want := []int{1268, 470, 27, 914, 19391}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestReleasedTokenizer(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_MODERNBERT_MODEL")
	if dir == "" {
		t.Skip("set local model")
	}
	tok, e := LoadTokenizer(dir)
	if e != nil {
		t.Fatal(e)
	}
	if tok.CLS != 50281 || tok.SEP != 50282 || tok.Mask != 50284 {
		t.Fatalf("specials %+v", tok)
	}
	ids, e := tok.Encode("Hello", true)
	if e != nil {
		t.Fatal(e)
	}
	if len(ids) < 3 || ids[0] != tok.CLS || ids[len(ids)-1] != tok.SEP {
		t.Fatal(ids)
	}
}
