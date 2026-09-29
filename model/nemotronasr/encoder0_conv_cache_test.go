package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0ConvCachePyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0Convolution(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	full := readStemFixture(t, "encoder0_conv_glu", encoderWidth*5)
	var cache Encoder0ConvCache
	for index, bounds := range [][2]int{{0, 1}, {1, 3}, {3, 5}} {
		frames := bounds[1] - bounds[0]
		chunk := make([]float32, encoderWidth*frames)
		for ch := 0; ch < encoderWidth; ch++ {
			copy(chunk[ch*frames:(ch+1)*frames], full[ch*5+bounds[0]:ch*5+bounds[1]])
		}
		original := append([]float32(nil), chunk...)
		padded, depth, err := cache.Update(chunk, frames, m.depth)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"padded", padded}, {"depth", depth}, {"state", cache.Snapshot()}} {
			rows := frames
			if item.name == "padded" {
				rows += encoderConvKernel - 1
			} else if item.name == "state" {
				rows = encoderConvKernel - 1
			}
			ref := readStemFixture(t, fmt.Sprintf("encoder0_conv_cache_%s%d", item.name, index), encoderWidth*rows)
			var maxAbs, sumAbs float64
			var outside int
			for i, value := range item.got {
				delta := math.Abs(float64(value - ref[i]))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
					outside++
				}
			}
			mean := sumAbs / float64(len(item.got))
			t.Logf("chunk=%d %s max_abs=%g mean_abs=%g outside=%d", index, item.name, maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				t.Fatalf("chunk=%d %s differs from independent reference", index, item.name)
			}
		}
		for i, value := range chunk {
			if value != original[i] {
				t.Fatalf("mutated caller chunk=%d i=%d", index, i)
			}
		}
		stable := cache.Snapshot()
		last := len(stable) - 1
		before := stable[last]
		padded[len(padded)-1]++
		depth[len(depth)-1]++
		stable[last]++
		if cache.Snapshot()[last] != before {
			t.Fatal("returned output or snapshot aliases cache")
		}
	}
}

func BenchmarkReleasedEncoder0ConvCacheChunks(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	m, loadErr := LoadEncoder0Convolution(file)
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	if loadErr != nil {
		b.Fatal(loadErr)
	}
	full := readStemFixture(b, "encoder0_conv_glu", encoderWidth*5)
	chunks := make([][]float32, 3)
	for i, bounds := range [][2]int{{0, 1}, {1, 3}, {3, 5}} {
		frames := bounds[1] - bounds[0]
		chunks[i] = make([]float32, encoderWidth*frames)
		for ch := 0; ch < encoderWidth; ch++ {
			copy(chunks[i][ch*frames:(ch+1)*frames], full[ch*5+bounds[0]:ch*5+bounds[1]])
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var cache Encoder0ConvCache
		for _, chunk := range chunks {
			if _, _, err := cache.Update(chunk, len(chunk)/encoderWidth, m.depth); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func TestEncoder0ConvCacheRejectsMalformedAndPreservesState(t *testing.T) {
	weight := make([]float32, encoderWidth*encoderConvKernel)
	var cache Encoder0ConvCache
	if _, _, err := cache.Update(make([]float32, encoderWidth), 1, weight); err != nil {
		t.Fatal(err)
	}
	before := cache.Snapshot()
	for _, bad := range []struct {
		frames int
		input  []float32
		weight []float32
	}{{0, nil, weight}, {6, make([]float32, 6*encoderWidth), weight}, {1, make([]float32, encoderWidth-1), weight}, {1, make([]float32, encoderWidth), weight[:len(weight)-1]}, {1, append([]float32{float32(math.NaN())}, make([]float32, encoderWidth-1)...), weight}} {
		if _, _, err := cache.Update(bad.input, bad.frames, bad.weight); err == nil {
			t.Fatal("accepted malformed cache input")
		}
	}
	for i, value := range cache.Snapshot() {
		if value != before[i] {
			t.Fatalf("mutated cache on error at %d", i)
		}
	}
	if (*Encoder0ConvCache)(nil).Snapshot() != nil {
		t.Fatal("nil cache snapshot")
	}
}
