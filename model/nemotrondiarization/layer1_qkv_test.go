package nemotrondiarization

import (
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedLayer1QKVComposedPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	layer0, err := LoadLayer0Complete(file)
	if err != nil {
		t.Fatal(err)
	}
	layer1, err := LoadLayer1QKV(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	input := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)
	original := append([]float32(nil), input...)
	hidden, err := layer0.ForwardOffline(input, 138)
	if err != nil {
		t.Fatal(err)
	}
	normal := make([]float32, len(hidden))
	if !simd.LayerNormLastAxisTo(normal, hidden, 138, projectedWidth, layer1.gamma, layer1.beta, 1e-5) {
		t.Fatal("layer-1 normalization rejected shape")
	}
	compareLayer1Fixture(t, "normal", normal)
	q, k, v, err := layer1.Project(hidden, 138)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		name string
		got  []float32
	}{{"q", q}, {"k", k}, {"v", v}} {
		compareLayer1Fixture(t, item.name, item.got)
	}
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated caller input %d", i)
		}
	}
	if &q[0] == &k[0] || &q[0] == &v[0] || &k[0] == &v[0] || &q[0] == &hidden[0] {
		t.Fatal("Q/K/V aliases another output or input")
	}
	stable := k[0]
	q[0]++
	if k[0] != stable {
		t.Fatal("Q aliases K")
	}
}

func compareLayer1Fixture(t *testing.T, name string, got []float32) {
	t.Helper()
	ref := readStackingFixture(t, "testdata/jfk_full_layer1_"+name+".f32.gz", 138*512)
	var maxAbs, sumAbs float64
	var outside int
	for i, value := range got {
		delta := math.Abs(float64(value - ref[i]))
		maxAbs = math.Max(maxAbs, delta)
		sumAbs += delta
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
			outside++
		}
	}
	mean := sumAbs / float64(len(got))
	t.Logf("composed layer1 %s max_abs=%g mean_abs=%g outside=%d", name, maxAbs, mean, outside)
	if outside != 0 || mean > 2e-5 {
		t.Fatalf("layer1 %s differs from independent PyTorch fixture", name)
	}
}

func TestLayer1QKVRejectsMalformed(t *testing.T) {
	if _, err := LoadLayer1QKV(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	m := &Layer1QKV{}
	if _, _, _, err := m.Project(make([]float32, 512), 1); err == nil {
		t.Fatal("accepted missing weights")
	}
	m = &Layer1QKV{gamma: make([]float32, 512), beta: make([]float32, 512), q: make([]float32, 512*512), k: make([]float32, 512*512), v: make([]float32, 512*512)}
	for _, rows := range []int{0, 377} {
		if _, _, _, err := m.Project(make([]float32, rows*512), rows); err == nil {
			t.Fatalf("accepted rows=%d", rows)
		}
	}
	if _, _, _, err := m.Project(make([]float32, 511), 1); err == nil {
		t.Fatal("accepted truncated input")
	}
	bad := make([]float32, 512)
	bad[0] = float32(math.NaN())
	if _, _, _, err := m.Project(bad, 1); err == nil {
		t.Fatal("accepted non-finite input")
	}
}
