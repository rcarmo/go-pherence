package nemotronasr

import (
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0FF2PyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0FeedForward2(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	input := readStemFixture(t, "encoder0_conv_residual", 5*encoderWidth)
	original := append([]float32(nil), input...)
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, 5, encoderWidth, m.gamma, m.beta, 1e-5) {
		t.Fatal("FF2 normalisation rejected fixture shape")
	}
	compareASRStage(t, "encoder0_ff2_normal", normal, len(input))
	first := make([]float32, 5*encoderFFWidth)
	if !simd.DenseNTTo(first, normal, m.first, 5, encoderFFWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderFFWidth) {
		t.Fatal("FF2 first linear rejected fixture shape")
	}
	compareASRStage(t, "encoder0_ff2_linear1", first, len(first))
	if !simd.SiLUTo(first, first) {
		t.Fatal("FF2 activation rejected fixture shape")
	}
	compareASRStage(t, "encoder0_ff2_activated", first, len(first))
	second := make([]float32, len(input))
	if !simd.DenseNTTo(second, first, m.last, 5, encoderWidth, encoderFFWidth, 1, encoderFFWidth, encoderFFWidth, encoderWidth) {
		t.Fatal("FF2 second linear rejected fixture shape")
	}
	compareASRStage(t, "encoder0_ff2_output", second, len(second))
	for i := range second {
		second[i] = input[i] + 0.5*second[i]
	}
	compareASRStage(t, "encoder0_ff2_residual", second, len(second))
	got, err := m.ForwardOffline(input, 5)
	if err != nil {
		t.Fatal(err)
	}
	compareASRStage(t, "encoder0_block_output", got, len(got))
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated caller input %d", i)
		}
	}
	stable := got[0]
	input[0]++
	if got[0] != stable {
		t.Fatal("output aliases caller input")
	}
}

func TestEncoder0FF2RejectsMalformed(t *testing.T) {
	if _, err := LoadEncoder0FeedForward2(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	m := &Encoder0FeedForward2{}
	if _, err := m.ForwardOffline(make([]float32, encoderWidth), 1); err == nil {
		t.Fatal("accepted missing weights")
	}
	m = &Encoder0FeedForward2{gamma: make([]float32, encoderWidth), beta: make([]float32, encoderWidth), first: make([]float32, encoderFFWidth*encoderWidth), last: make([]float32, encoderWidth*encoderFFWidth), outGamma: make([]float32, encoderWidth), outBeta: make([]float32, encoderWidth)}
	for _, rows := range []int{0, 6} {
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

func BenchmarkReleasedEncoder0FF25(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	m, loadErr := LoadEncoder0FeedForward2(file)
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	if loadErr != nil {
		b.Fatal(loadErr)
	}
	input := readStemFixture(b, "encoder0_conv_residual", 5*encoderWidth)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := m.ForwardOffline(input, 5); err != nil {
			b.Fatal(err)
		}
	}
}
