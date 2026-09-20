package needle

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

type reference struct {
	Config       json.RawMessage                       `json:"config"`
	Tensors      map[string]checkpoint.Tensor          `json:"tensors"`
	Tokens       []int                                 `json:"tokens"`
	Logits       checkpoint.Tensor                     `json:"logits"`
	Loss         float64                               `json:"loss"`
	Gradients    map[string]checkpoint.Tensor          `json:"gradients"`
	Permutations struct{ P1, P2 struct{ Data []int } } `json:"permutations"`
}

func fixture(t testing.TB) (*Model, reference) { return fixtureFile(t, "needle3.json") }
func fixtureFile(t testing.TB, name string) (*Model, reference) {
	t.Helper()
	b, e := os.ReadFile("testdata/" + name)
	if e != nil {
		t.Fatal(e)
	}
	var ref reference
	if e = json.Unmarshal(b, &ref); e != nil {
		t.Fatal(e)
	}
	m, e := New(&checkpoint.Checkpoint{Config: ref.Config, FormatVersion: 2, Tensors: ref.Tensors})
	if e != nil {
		t.Fatal(e)
	}
	return m, ref
}
func TestNeedle3Upstream(t *testing.T) { testReference(t, "needle3.json") }
func TestNeedle2Upstream(t *testing.T) { testReference(t, "needle2.json") }
func testReference(t *testing.T, name string) {
	m, f := fixtureFile(t, name)
	if m.config.Generation == 3 && (!slices.Equal(m.p1, f.Permutations.P1.Data) || !slices.Equal(m.p2, f.Permutations.P2.Data)) {
		t.Fatalf("permutations got %v %v", m.p1, m.p2)
	}
	out, e := m.Forward(f.Tokens, Options{})
	if e != nil {
		t.Fatal(e)
	}
	compare(t, "logits", out, f.Logits.Data, 3e-5, 3e-4)
	loss, g, e := m.LossGrad(f.Tokens, nil, Options{})
	if e != nil {
		t.Fatal(e)
	}
	if math.Abs(loss-f.Loss) > 2e-6 {
		t.Errorf("loss %.10f want %.10f", loss, f.Loss)
	}
	if len(g) != len(f.Gradients) {
		t.Fatalf("gradients %d vs %d", len(g), len(f.Gradients))
	}
	for key, want := range f.Gradients {
		got, ok := g[key]
		if !ok {
			t.Errorf("missing %s", key)
			continue
		}
		compare(t, key, got.Data, want.Data, 3e-6, 5e-3)
	}
}
func compare(t *testing.T, name string, a, b []float32, abs, rel float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s lengths %d/%d", name, len(a), len(b))
	}
	maxErr := 0.
	idx := 0
	bad := 0
	for i, v := range a {
		err := math.Abs(float64(v - b[i]))
		if err > maxErr {
			maxErr = err
			idx = i
		}
		if math.IsNaN(float64(v)) || err > abs+rel*math.Abs(float64(b[i])) {
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d mismatches max abs %.8g at %d got %.9g want %.9g", name, bad, maxErr, idx, a[idx], b[idx])
	}
}
func TestAdmission(t *testing.T) {
	m, f := fixture(t)
	for _, ids := range [][]int{nil, {-1}, {16}, make([]int, 17)} {
		if _, e := m.Forward(ids, Options{}); e == nil {
			t.Error("invalid tokens accepted")
		}
	}
	if _, e := m.Forward(f.Tokens, Options{MaxWorkBytes: 32}); e == nil {
		t.Error("workspace budget ignored")
	}
	if _, _, e := m.LossGrad(f.Tokens, []float32{0, 0, 0, 0}, Options{}); e == nil {
		t.Error("empty mask accepted")
	}
}
func BenchmarkNeedle3Forward(b *testing.B) {
	m, f := fixture(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, e := m.Forward(f.Tokens, Options{}); e != nil {
			b.Fatal(e)
		}
	}
}

func TestGeneration(t *testing.T) {
	m, f := fixture(t)
	ids, err := m.Generate(context.Background(), f.Tokens, 2, -1, Options{})
	if err != nil || len(ids) != 2 {
		t.Fatalf("generate %v %v", ids, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = m.Generate(ctx, f.Tokens, 2, -1, Options{}); err == nil {
		t.Fatal("ignored cancellation")
	}
	if _, err = m.Generate(context.Background(), f.Tokens, 99, -1, Options{}); err == nil {
		t.Fatal("ignored context cap")
	}
}
