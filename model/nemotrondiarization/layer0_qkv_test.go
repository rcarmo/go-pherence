package nemotrondiarization

import (
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedLayer0QKVParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadLayer0QKV(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	full := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)
	input := full[:16*512]
	original := append([]float32(nil), input...)
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, 16, 512, m.gamma, m.beta, 1e-5) {
		t.Fatal("layer normalisation rejected released shape")
	}
	compareLayer0Fixture(t, "normal", normal)
	q, k, v, err := m.Project(input, 16)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		name string
		got  []float32
	}{{"q", q}, {"k", k}, {"v", v}} {
		compareLayer0Fixture(t, item.name, item.got)
	}
	for i := range input {
		if input[i] != original[i] {
			t.Fatal("mutated caller input")
		}
	}
	if &q[0] == &k[0] || &q[0] == &v[0] || &k[0] == &v[0] || &q[0] == &input[0] || &k[0] == &input[0] || &v[0] == &input[0] {
		t.Fatal("Q/K/V alias one another or caller input")
	}
	stableK := k[0]
	q[0]++
	if k[0] != stableK {
		t.Fatal("Q mutation changed K")
	}
	if _, _, _, err := m.Project(input[:len(input)-1], 16); err == nil {
		t.Fatal("accepted truncated input")
	}
	invalid := append([]float32(nil), input...)
	invalid[0] = float32(math.NaN())
	if _, _, _, err := m.Project(invalid, 16); err == nil {
		t.Fatal("accepted non-finite input")
	}
}

func compareLayer0Fixture(t *testing.T, name string, got []float32) {
	t.Helper()
	ref := readStackingFixture(t, "testdata/jfk_layer0_"+name+".f32.gz", 16*512)
	var maxAbs, sumAbs float64
	var outside int
	for i, actual := range got {
		d := math.Abs(float64(actual - ref[i]))
		if d > maxAbs {
			maxAbs = d
		}
		sumAbs += d
		if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) || d > 3e-4+2e-5*math.Abs(float64(ref[i])) {
			outside++
		}
	}
	mean := sumAbs / float64(len(got))
	t.Logf("layer0 %s max_abs=%g mean_abs=%g outside=%d", name, maxAbs, mean, outside)
	if outside != 0 || mean > 2e-5 {
		t.Fatalf("layer0 %s differs from independent reference distribution", name)
	}
}

func TestLayer0QKVRejectsMalformed(t *testing.T) {
	m := &Layer0QKV{gamma: make([]float32, 512), beta: make([]float32, 512), q: make([]float32, 512*512), k: make([]float32, 512*512), v: make([]float32, 512*512)}
	for _, rows := range []int{0, 377} {
		if _, _, _, err := m.Project(make([]float32, rows*512), rows); err == nil {
			t.Fatalf("accepted rows=%d", rows)
		}
	}
	if _, _, _, err := m.Project(make([]float32, 511), 1); err == nil {
		t.Fatal("accepted truncated input")
	}
	if _, err := LoadLayer0QKV(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
}

func BenchmarkReleasedLayer0QKV16(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	m, err := LoadLayer0QKV(file)
	if err != nil {
		b.Fatal(err)
	}
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	input := readStackingFixture(b, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)[:16*512]
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, _, err := m.Project(input, 16); err != nil {
			b.Fatal(err)
		}
	}
}
