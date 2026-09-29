package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedIndexedEncoderBlock1LookaheadParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	first, err := LoadEncoder0Block(file)
	if err != nil {
		t.Fatal(err)
	}
	second, err := LoadIndexedEncoderBlock(file, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	projected := readStemFixture(t, "projected", 5*encoderWidth)
	original := append([]float32(nil), projected...)
	for _, lookahead := range []int{0, 3} {
		hidden, err := first.ForwardOfflineLookahead(projected, 5, lookahead)
		if err != nil {
			t.Fatal(err)
		}
		ff1, err := second.FF1.ForwardOffline(hidden, 5)
		if err != nil {
			t.Fatal(err)
		}
		normal := make([]float32, len(ff1))
		if !simd.LayerNormLastAxisTo(normal, ff1, 5, encoderWidth, second.Attention.qkv.gamma, second.Attention.qkv.beta, 1e-5) {
			t.Fatal("second layer norm rejected")
		}
		compareIndexedBlockFixture(t, fmt.Sprintf("encoder1_attn_normal_look%d", lookahead), normal)
		output, err := second.ForwardOfflineLookahead(hidden, 5, lookahead)
		if err != nil {
			t.Fatal(err)
		}
		compareIndexedBlockFixture(t, fmt.Sprintf("encoder1_block_output_look%d", lookahead), output)
	}
	for i, value := range projected {
		if value != original[i] {
			t.Fatalf("mutated input %d", i)
		}
	}
}

func compareIndexedBlockFixture(t *testing.T, name string, got []float32) {
	t.Helper()
	ref := readStemFixture(t, name, len(got))
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
	t.Logf("%s max_abs=%g mean_abs=%g outside=%d", name, maxAbs, mean, outside)
	if outside != 0 || mean > 2e-5 {
		t.Fatalf("%s differs from PyTorch", name)
	}
}

func TestIndexedEncoderBlockRejectsMalformed(t *testing.T) {
	for _, index := range []int{-1, 0, 24, 25} {
		if _, err := LoadIndexedEncoderBlock(nil, index); err == nil {
			t.Fatalf("accepted nil checkpoint layer=%d", index)
		}
	}
}
