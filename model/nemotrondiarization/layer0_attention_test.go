package nemotrondiarization

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedLayer0AttentionPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadLayer0Attention(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	full := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)
	input := full[:16*512]
	original := append([]float32(nil), input...)
	attention, residual, err := m.ForwardOffline(input, 16)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		name string
		got  []float32
	}{{"attention", attention}, {"residual", residual}} {
		ref := readStackingFixture(t, "testdata/jfk_layer0_"+item.name+".f32.gz", 16*512)
		var maxAbs, sumAbs float64
		var outside int
		for i, actual := range item.got {
			delta := math.Abs(float64(actual - ref[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
				outside++
			}
		}
		mean := sumAbs / float64(len(item.got))
		t.Logf("layer0 %s max_abs=%g mean_abs=%g outside=%d", item.name, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-5 {
			t.Fatalf("layer0 %s differs from independent PyTorch output", item.name)
		}
	}
	for i := range input {
		if input[i] != original[i] {
			t.Fatalf("mutated caller input %d", i)
		}
	}
	if &attention[0] == &residual[0] || &attention[0] == &input[0] || &residual[0] == &input[0] {
		t.Fatal("attention/residual alias input or one another")
	}
	stable := residual[0]
	attention[0]++
	if residual[0] != stable {
		t.Fatal("attention output aliases residual")
	}
}

func TestReleasedLayer0AttentionFullContextParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadLayer0Attention(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	input := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)
	original := append([]float32(nil), input...)
	attention, residual, err := m.ForwardOffline(input, 138)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		name string
		got  []float32
	}{{"attention", attention}, {"residual", residual}} {
		ref := readStackingFixture(t, "testdata/jfk_full_layer0_"+item.name+".f32.gz", 138*512)
		var maxAbs, sumAbs float64
		var outside int
		for i, actual := range item.got {
			delta := math.Abs(float64(actual - ref[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
				outside++
			}
		}
		mean := sumAbs / float64(len(item.got))
		t.Logf("full layer0 %s max_abs=%g mean_abs=%g outside=%d", item.name, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-5 {
			t.Fatalf("full layer0 %s differs from independent reference", item.name)
		}
	}
	for i := range input {
		if input[i] != original[i] {
			t.Fatalf("mutated full-context input %d", i)
		}
	}
}

func TestLayer0AttentionRejectsMalformed(t *testing.T) {
	m := &Layer0Attention{qkv: &Layer0QKV{inputGamma: make([]float32, 512), inputBeta: make([]float32, 512), gamma: make([]float32, 512), beta: make([]float32, 512), q: make([]float32, 512*512), k: make([]float32, 512*512), v: make([]float32, 512*512)}, outWeight: make([]float32, 512*512), outBias: make([]float32, 512)}
	for _, rows := range []int{0, 377} {
		if _, _, err := m.ForwardOffline(make([]float32, rows*512), rows); err == nil {
			t.Fatalf("accepted rows=%d", rows)
		}
	}
	if _, _, err := m.ForwardOffline(make([]float32, 511), 1); err == nil {
		t.Fatal("accepted short window")
	}
	if _, err := LoadLayer0Attention(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
}

func BenchmarkReleasedLayer0Attention16(b *testing.B)  { benchmarkReleasedLayer0Attention(b, 16) }
func BenchmarkReleasedLayer0Attention138(b *testing.B) { benchmarkReleasedLayer0Attention(b, 138) }

func benchmarkReleasedLayer0Attention(b *testing.B, rows int) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	m, loadErr := LoadLayer0Attention(file)
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	if loadErr != nil {
		b.Fatal(loadErr)
	}
	input := readStackingFixture(b, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)[:rows*512]
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := m.ForwardOffline(input, rows); err != nil {
			b.Fatal(err)
		}
	}
}
