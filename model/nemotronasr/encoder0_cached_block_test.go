package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0CachedBlockChunksPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	block, err := LoadEncoder0Block(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	full := readStemFixture(t, "projected", 5*encoderWidth)
	for _, lookahead := range []int{0, 3} {
		var state Encoder0ChunkState
		for index, bounds := range [][2]int{{0, 1}, {1, 3}, {3, 5}} {
			rows := bounds[1] - bounds[0]
			input := append([]float32(nil), full[bounds[0]*encoderWidth:bounds[1]*encoderWidth]...)
			initial := append([]float32(nil), input...)
			got, err := block.ForwardCachedChunk(input, rows, lookahead, &state)
			if err != nil {
				t.Fatal(err)
			}
			ref := readStemFixture(t, fmt.Sprintf("encoder0_block_chunk_output_look%d_%d", lookahead, index), len(got))
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
			t.Logf("lookahead=%d chunk=%d max_abs=%g mean_abs=%g outside=%d", lookahead, index, maxAbs, mean, outside)
			if outside != 0 {
				t.Fatal("cached block differs from PyTorch")
			}
			_, _, _, seen := state.Attention.Snapshot()
			if seen != bounds[1] {
				t.Fatalf("cache seen=%d want=%d", seen, bounds[1])
			}
			for i, value := range input {
				if value != initial[i] {
					t.Fatalf("mutated input %d", i)
				}
			}
		}
	}
}

func TestEncoder0CachedBlockRejectsMalformed(t *testing.T) {
	if _, err := (*Encoder0Block)(nil).ForwardCachedChunk(make([]float32, encoderWidth), 1, 0, &Encoder0ChunkState{}); err == nil {
		t.Fatal("accepted nil block")
	}
	if _, err := (&Encoder0Block{}).ForwardCachedChunk(make([]float32, encoderWidth), 1, 0, &Encoder0ChunkState{}); err == nil {
		t.Fatal("accepted missing weights")
	}
}

func TestReleasedEncoder0CachedBlockRejectsWithoutStateMutation(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	block, err := LoadEncoder0Block(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	input := readStemFixture(t, "projected", 5*encoderWidth)[:encoderWidth]
	var state Encoder0ChunkState
	if _, err := block.ForwardCachedChunk(input, 1, 0, &state); err != nil {
		t.Fatal(err)
	}
	keys, values, retained, seen := state.Attention.Snapshot()
	conv := state.Conv.Snapshot()
	check := func() {
		t.Helper()
		k, v, r, s := state.Attention.Snapshot()
		if !reflect.DeepEqual(k, keys) || !reflect.DeepEqual(v, values) || r != retained || s != seen || !reflect.DeepEqual(state.Conv.Snapshot(), conv) {
			t.Fatal("rejected chunk changed cache")
		}
	}
	bad := append([]float32(nil), input...)
	bad[0] = float32(math.NaN())
	if _, err := block.ForwardCachedChunk(bad, 1, 0, &state); err == nil {
		t.Fatal("accepted non-finite input")
	}
	check()
	if _, err := block.ForwardCachedChunk(input, 1, 2, &state); err == nil {
		t.Fatal("accepted unsupported lookahead")
	}
	check()
	if _, err := block.ForwardCachedChunk(input, 0, 0, &state); err == nil {
		t.Fatal("accepted zero rows")
	}
	check()
}
