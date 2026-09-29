package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0BlockComposedPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	sub, subErr := LoadSubsampling(file)
	block, blockErr := LoadEncoder0Block(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if subErr != nil {
		t.Fatal(subErr)
	}
	if blockErr != nil {
		t.Fatal(blockErr)
	}
	for _, tc := range []struct {
		label string
		input []float32
	}{
		{"independent subsampling fixture", readStemFixture(t, "projected", 5*encoderWidth)},
		{"composed Go subsampling", nil},
	} {
		if tc.input == nil {
			features := readStemFixture(t, "features", 32*128)
			originalFeatures := append([]float32(nil), features...)
			tc.input, err = sub.ForwardOffline(features, 32, 32)
			if err != nil {
				t.Fatal(err)
			}
			for i, value := range features {
				if value != originalFeatures[i] {
					t.Fatalf("mutated caller feature at %d", i)
				}
			}
		}
		original := append([]float32(nil), tc.input...)
		got, err := block.ForwardOffline(tc.input, 5)
		if err != nil {
			t.Fatal(err)
		}
		ref := readStemFixture(t, "encoder0_block_output", 5*encoderWidth)
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
		t.Logf("%s block max_abs=%g mean_abs=%g outside=%d", tc.label, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-5 {
			t.Fatalf("%s block differs from independent PyTorch output", tc.label)
		}
		for i, value := range tc.input {
			if value != original[i] {
				t.Fatalf("%s mutated caller input at %d", tc.label, i)
			}
		}
	}
}

func TestReleasedEncoder0BlockLookaheadParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0Block(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	input := readStemFixture(t, "projected", 5*encoderWidth)
	original := append([]float32(nil), input...)
	for _, lookahead := range []int{0, 3} {
		got, err := m.ForwardOfflineLookahead(input, 5, lookahead)
		if err != nil {
			t.Fatal(err)
		}
		ref := readStemFixture(t, fmt.Sprintf("encoder0_block_output_look%d", lookahead), len(got))
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
		t.Logf("lookahead=%d block max_abs=%g mean_abs=%g outside=%d", lookahead, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-5 {
			t.Fatalf("lookahead=%d block differs from independent reference", lookahead)
		}
	}
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated caller input at %d", i)
		}
	}
}

func TestEncoder0BlockRejectsMalformed(t *testing.T) {
	if _, err := LoadEncoder0Block(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	m := &Encoder0Block{}
	if _, err := m.ForwardOffline(make([]float32, encoderWidth), 1); err == nil {
		t.Fatal("accepted missing weights")
	}
	if _, err := m.ForwardOfflineLookahead(make([]float32, encoderWidth), 1, 2); err == nil {
		t.Fatal("accepted unsupported lookahead")
	}
}

func BenchmarkReleasedEncoder0Block5(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	m, loadErr := LoadEncoder0Block(file)
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	if loadErr != nil {
		b.Fatal(loadErr)
	}
	input := readStemFixture(b, "projected", 5*encoderWidth)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := m.ForwardOffline(input, 5); err != nil {
			b.Fatal(err)
		}
	}
}
