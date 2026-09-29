package nemotronasr

import (
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0QKVComposedPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0QKV(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	input := readStemFixture(t, "encoder0_ff1_residual", 5*encoderWidth)
	original := append([]float32(nil), input...)
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, 5, encoderWidth, m.gamma, m.beta, 1e-5) {
		t.Fatal("attention normalisation rejected fixture shape")
	}
	compareASRStage(t, "encoder0_attn_normal", normal, len(input))
	q, k, v, err := m.Project(input, 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		name string
		got  []float32
	}{{"q", q}, {"k", k}, {"v", v}} {
		compareASRStage(t, "encoder0_attn_"+item.name, item.got, len(input))
	}
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated caller input at %d", i)
		}
	}
	if &q[0] == &k[0] || &q[0] == &v[0] || &k[0] == &v[0] || &q[0] == &input[0] {
		t.Fatal("Q/K/V aliases input or another output")
	}
	stableK := k[0]
	q[0]++
	if k[0] != stableK {
		t.Fatal("Q aliases K")
	}
}

func TestEncoder0QKVRejectsMalformed(t *testing.T) {
	if _, err := LoadEncoder0QKV(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	m := &Encoder0QKV{gamma: make([]float32, encoderWidth), beta: make([]float32, encoderWidth), q: make([]float32, encoderWidth*encoderWidth), k: make([]float32, encoderWidth*encoderWidth), v: make([]float32, encoderWidth*encoderWidth)}
	for _, rows := range []int{0, 18} {
		if _, _, _, err := m.Project(make([]float32, rows*encoderWidth), rows); err == nil {
			t.Fatalf("accepted rows=%d", rows)
		}
	}
	if _, _, _, err := m.Project(make([]float32, encoderWidth-1), 1); err == nil {
		t.Fatal("accepted short input")
	}
	input := make([]float32, encoderWidth)
	input[0] = float32(math.NaN())
	if _, _, _, err := m.Project(input, 1); err == nil {
		t.Fatal("accepted non-finite input")
	}
}

func BenchmarkReleasedEncoder0QKV5(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	m, loadErr := LoadEncoder0QKV(file)
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	if loadErr != nil {
		b.Fatal(loadErr)
	}
	input := readStemFixture(b, "encoder0_ff1_residual", 5*encoderWidth)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, _, err := m.Project(input, 5); err != nil {
			b.Fatal(err)
		}
	}
}
