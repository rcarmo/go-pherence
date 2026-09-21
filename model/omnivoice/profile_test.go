package omnivoice

import (
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"os"
	"testing"
)

func BenchmarkRealBlock(b *testing.B)     { benchmarkRealBlock(b, false) }
func BenchmarkRealBlockInto(b *testing.B) { benchmarkRealBlock(b, true) }
func benchmarkRealBlock(b *testing.B, reuse bool) {
	path := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if path == "" {
		b.Skip("set model directory")
	}
	weights, err := loader.OpenWeights(path)
	if err != nil {
		b.Fatal(err)
	}
	defer weights.Close()
	w, err := weights.Layer(0)
	if err != nil {
		b.Fatal(err)
	}
	block, err := NewBlock(weights.Config.LLMConfig, w)
	if err != nil {
		b.Fatal(err)
	}
	const tokens = 128
	x := make([]float32, tokens*weights.Config.LLMConfig.HiddenSize)
	for i := range x {
		x[i] = float32(i%101) * 0.001
	}
	scratch, err := block.NewWorkspace(tokens)
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]float32, len(x))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if reuse {
			err = block.ForwardInto(dst, x, tokens, nil, nil, scratch)
		} else {
			_, err = block.Forward(x, tokens, nil, nil)
		}
		if err != nil {
			b.Fatal(err)
		}
	}
}
