package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0CachedAttentionChunksPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0Attention(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	full := readStemFixture(t, "encoder0_attn_normal", 5*encoderWidth)
	for _, lookahead := range []int{0, 3} {
		var cache Encoder0KVCache
		for index, bounds := range [][2]int{{0, 1}, {1, 3}, {3, 5}} {
			frames := bounds[1] - bounds[0]
			_, _, retained, seenBefore := cache.Snapshot()
			positionRows := retained + frames
			positions := readStemFixture(t, fmt.Sprintf("encoder0_attn_chunk_positions_look%d_%d", lookahead, index), (2*positionRows-1)*encoderWidth)
			for i, value := range encoder0RelativePositions(positionRows) {
				if math.Abs(float64(value-positions[i])) > 2e-6 {
					t.Fatalf("lookahead=%d chunk=%d relative position %d differs", lookahead, index, i)
				}
			}
			mask := readStemFixture(t, fmt.Sprintf("encoder0_attn_chunk_mask_look%d_%d", lookahead, index), frames*positionRows)
			for row := 0; row < frames; row++ {
				for source := 0; source < positionRows; source++ {
					globalSource := seenBefore - retained + source
					allowed := globalSource/(lookahead+1) <= (seenBefore+row)/(lookahead+1)
					want := float32(0)
					if allowed {
						want = 1
					}
					if mask[row*positionRows+source] != want {
						t.Fatalf("lookahead=%d chunk=%d mask row=%d source=%d", lookahead, index, row, source)
					}
				}
			}
			chunk := append([]float32(nil), full[bounds[0]*encoderWidth:bounds[1]*encoderWidth]...)
			original := append([]float32(nil), chunk...)
			got, err := m.ForwardCachedChunk(chunk, frames, lookahead, &cache)
			if err != nil {
				t.Fatal(err)
			}
			ref := readStemFixture(t, fmt.Sprintf("encoder0_attn_chunk_output_look%d_%d", lookahead, index), len(got))
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
				t.Fatalf("lookahead=%d chunk=%d differs from PyTorch", lookahead, index)
			}
			for i, value := range chunk {
				if value != original[i] {
					t.Fatalf("mutated chunk input %d", i)
				}
			}
			_, _, _, seen := cache.Snapshot()
			if seen != bounds[1] {
				t.Fatalf("cache seen=%d want=%d", seen, bounds[1])
			}
		}
	}
}

func TestEncoder0CachedAttentionRejectsMalformed(t *testing.T) {
	var cache Encoder0KVCache
	if _, err := (*Encoder0Attention)(nil).ForwardCachedChunk(make([]float32, encoderWidth), 1, 0, &cache); err == nil {
		t.Fatal("accepted nil model")
	}
	m := &Encoder0Attention{}
	if _, err := m.ForwardCachedChunk(make([]float32, encoderWidth), 1, 0, &cache); err == nil {
		t.Fatal("accepted missing weights")
	}
}

func TestEncoder0CachedAttentionRejectedOutputPreservesCache(t *testing.T) {
	m := &Encoder0Attention{
		qkv: &Encoder0QKV{
			q: make([]float32, encoderWidth*encoderWidth),
			k: make([]float32, encoderWidth*encoderWidth),
			v: make([]float32, encoderWidth*encoderWidth),
		},
		relativeWeight: make([]float32, encoderWidth*encoderWidth),
		biasU:          make([]float32, encoderWidth),
		biasV:          make([]float32, encoderWidth),
		outputWeight:   make([]float32, encoderWidth*encoderWidth),
	}
	// Finite weights and input can overflow at the final projection, after
	// the KV transition has been prepared. The rejected call must not commit it.
	m.qkv.v[0] = math.MaxFloat32
	m.outputWeight[0] = math.MaxFloat32
	input := make([]float32, encoderWidth)
	input[0] = 1
	var cache Encoder0KVCache
	if _, err := m.ForwardCachedChunk(input, 1, 0, &cache); err == nil {
		t.Fatal("accepted non-finite output")
	}
	keys, values, retained, seen := cache.Snapshot()
	if len(keys) != 0 || len(values) != 0 || retained != 0 || seen != 0 || input[0] != 1 {
		t.Fatal("rejected output changed cache or input")
	}
	if _, err := m.ForwardCachedChunk(input, 1, 0, &cache); err == nil {
		t.Fatal("accepted non-finite output on retry")
	}
	keys2, values2, retained2, seen2 := cache.Snapshot()
	if !reflect.DeepEqual(keys, keys2) || !reflect.DeepEqual(values, values2) || retained2 != retained || seen2 != seen {
		t.Fatal("rejected retry changed cache")
	}
	// This operator is qualified only across the five prepared fixture rows.
	cache.seen = 5
	if _, err := m.ForwardCachedChunk(input, 1, 0, &cache); err == nil {
		t.Fatal("accepted an unqualified sixth row")
	}
	if cache.seen != 5 || cache.retained != 0 {
		t.Fatal("rejected sixth row changed cache")
	}
}
