package nemotronasr

import (
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func releasedEncoder0FF1(t testing.TB) *Encoder0FeedForward1 {
	t.Helper()
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0FeedForward1(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	return m
}

func TestReleasedEncoder0FF1PyTorchParity(t *testing.T) {
	m := releasedEncoder0FF1(t)
	input := readStemFixture(t, "projected", 5*encoderWidth)
	original := append([]float32(nil), input...)
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, 5, encoderWidth, m.gamma, m.beta, 1e-5) {
		t.Fatal("normalisation rejected fixture shape")
	}
	compareASRStage(t, "encoder0_ff1_normal", normal, 5*encoderWidth)
	first := make([]float32, 5*encoderFFWidth)
	if !simd.DenseNTTo(first, normal, m.first, 5, encoderFFWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderFFWidth) {
		t.Fatal("linear1 rejected fixture shape")
	}
	compareASRStage(t, "encoder0_ff1_linear1", first, 5*encoderFFWidth)
	if !simd.SiLUTo(first, first) {
		t.Fatal("activation rejected fixture shape")
	}
	compareASRStage(t, "encoder0_ff1_activated", first, 5*encoderFFWidth)
	second := make([]float32, len(input))
	if !simd.DenseNTTo(second, first, m.last, 5, encoderWidth, encoderFFWidth, 1, encoderFFWidth, encoderFFWidth, encoderWidth) {
		t.Fatal("linear2 rejected fixture shape")
	}
	compareASRStage(t, "encoder0_ff1_output", second, 5*encoderWidth)
	got, err := m.ForwardOffline(input, 5)
	if err != nil {
		t.Fatal(err)
	}
	compareASRStage(t, "encoder0_ff1_residual", got, 5*encoderWidth)
	features := readStemFixture(t, "features", 32*128)
	originalFeatures := append([]float32(nil), features...)
	sub := releasedSubsampling(t)
	composedInput, err := sub.ForwardOffline(features, 32, 32)
	if err != nil {
		t.Fatal(err)
	}
	composed, err := m.ForwardOffline(composedInput, 5)
	if err != nil {
		t.Fatal(err)
	}
	refComposed := readStemFixture(t, "encoder0_ff1_residual", 5*encoderWidth)
	var maxAbs, sumAbs float64
	var outside int
	for i, value := range composed {
		delta := math.Abs(float64(value - refComposed[i]))
		maxAbs = math.Max(maxAbs, delta)
		sumAbs += delta
		// Propagated subsampling F32 rounding accounts for the two small
		// outputs outside the standalone FF1 gate. Independently running
		// PyTorch FF1 on Go's projected input reproduced both outliers.
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 1e-3+2e-5*math.Abs(float64(refComposed[i])) {
			outside++
		}
	}
	mean := sumAbs / float64(len(composed))
	t.Logf("composed FF1 max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
	if outside != 0 || mean > 1e-4 {
		t.Fatal("composed FF1 differs from independent PyTorch distribution")
	}
	for i, value := range features {
		if value != originalFeatures[i] {
			t.Fatalf("mutated caller features at %d", i)
		}
	}
	for i := range input {
		if input[i] != original[i] {
			t.Fatalf("mutated caller input at %d", i)
		}
	}
	stable := got[0]
	input[0]++
	if got[0] != stable {
		t.Fatal("returned output aliases caller input")
	}
}

func TestEncoder0FF1RejectsMalformed(t *testing.T) {
	if _, err := LoadEncoder0FeedForward1(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	m := &Encoder0FeedForward1{gamma: make([]float32, encoderWidth), beta: make([]float32, encoderWidth), first: make([]float32, encoderFFWidth*encoderWidth), last: make([]float32, encoderWidth*encoderFFWidth)}
	for _, rows := range []int{0, 18} {
		if _, err := m.ForwardOffline(make([]float32, rows*encoderWidth), rows); err == nil {
			t.Fatalf("accepted rows=%d", rows)
		}
	}
	if _, err := m.ForwardOffline(make([]float32, encoderWidth-1), 1); err == nil {
		t.Fatal("accepted short input")
	}
	invalid := make([]float32, encoderWidth)
	invalid[0] = float32(math.NaN())
	if _, err := m.ForwardOffline(invalid, 1); err == nil {
		t.Fatal("accepted non-finite input")
	}
}

func BenchmarkReleasedEncoder0FF15(b *testing.B) {
	m := releasedEncoder0FF1(b)
	input := readStemFixture(b, "projected", 5*encoderWidth)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := m.ForwardOffline(input, 5); err != nil {
			b.Fatal(err)
		}
	}
}
