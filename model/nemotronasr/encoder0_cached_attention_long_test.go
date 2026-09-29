package nemotronasr

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0CachedAttentionSlidingWindowPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	refDir := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_ATTENTION_LONG_REF")
	if path == "" || refDir == "" {
		t.Skip("set pinned ASR model and GO_PHERENCE_NEMOTRON_ASR_ATTENTION_LONG_REF")
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
	read := func(name string) []float32 {
		t.Helper()
		file, err := os.Open(filepath.Join(refDir, name+".f32.gz"))
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		gz, err := gzip.NewReader(file)
		if err != nil {
			t.Fatal(err)
		}
		defer gz.Close()
		data, err := io.ReadAll(gz)
		if err != nil || len(data) != 72*encoderWidth*4 {
			t.Fatalf("%s bytes=%d err=%v", name, len(data), err)
		}
		values := make([]float32, 72*encoderWidth)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return values
	}
	input := read("input")
	for _, lookahead := range []int{0, 3} {
		ref := read(fmt.Sprintf("look%d", lookahead))
		var state Encoder0KVCache
		var outside int
		var maxAbs, sumAbs float64
		for start := 0; start < 72; start += 4 {
			chunk := input[start*encoderWidth : (start+4)*encoderWidth]
			got, err := m.ForwardCachedChunk(chunk, 4, lookahead, &state)
			if err != nil {
				t.Fatalf("lookahead=%d start=%d: %v", lookahead, start, err)
			}
			for i, value := range got {
				want := ref[start*encoderWidth+i]
				delta := math.Abs(float64(value - want))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(want)) {
					outside++
				}
			}
			_, _, retained, seen := state.Snapshot()
			if seen != start+4 || retained != min(seen, asrKVWindow-1) {
				t.Fatalf("cache seen=%d retained=%d", seen, retained)
			}
		}
		mean := sumAbs / float64(len(ref))
		t.Logf("lookahead=%d rows=72 max_abs=%g mean_abs=%g outside=%d", lookahead, maxAbs, mean, outside)
		// Independent JFK-derived F32 QKV, relative scores and weighted
		// value sums differ by up to 2.21e-5 mean on scalar; every value
		// remains inside the calibrated per-value threshold.
		if outside != 0 || mean > 2.5e-5 {
			t.Fatal("cached attention differs from PyTorch")
		}
	}
}
