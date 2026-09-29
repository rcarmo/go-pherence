package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedOfflineEncoderTowerPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	tower, err := LoadOfflineEncoderTower(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	input := readStemFixture(t, "projected", 5*encoderWidth)
	original := append([]float32(nil), input...)
	for _, lookahead := range []int{0, 3} {
		hidden := input
		for layer, block := range tower.layers {
			hidden, err = block.ForwardOfflineLookahead(hidden, 5, lookahead)
			if err != nil {
				t.Fatalf("layer=%d lookahead=%d: %v", layer, lookahead, err)
			}
			switch layer {
			case 7, 15, 23:
				compareOfflineTowerFixture(t, fmt.Sprintf("encoder%d_block_output_look%d", layer, lookahead), hidden)
			}
		}
		out, err := tower.ForwardOfflineLookahead(input, lookahead)
		if err != nil {
			t.Fatal(err)
		}
		compareOfflineTowerFixture(t, fmt.Sprintf("encoder23_block_output_look%d", lookahead), out)
	}
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated tower input %d", i)
		}
	}
}

func compareOfflineTowerFixture(t *testing.T, name string, got []float32) {
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
	if outside != 0 {
		t.Fatalf("%s differs from PyTorch", name)
	}
}

func TestOfflineEncoderTowerRejectsMalformed(t *testing.T) {
	if _, err := LoadOfflineEncoderTower(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, err := (*OfflineEncoderTower)(nil).ForwardOfflineLookahead(make([]float32, 5*encoderWidth), 0); err == nil {
		t.Fatal("accepted nil tower")
	}
}
