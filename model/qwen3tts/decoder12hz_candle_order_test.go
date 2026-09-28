package qwen3tts

import (
	"math"
	"slices"
	"sync"
	"testing"
)

func TestDecoderBalancedBlockSize(t *testing.T) {
	for _, tc := range []struct{ k, want int }{{1, 1}, {512, 512}, {513, 257}, {1024, 512}, {1025, 342}, {1536, 512}} {
		if got := decoderBlockSize(tc.k); got != tc.want {
			t.Fatalf("k=%d block=%d want=%d", tc.k, got, tc.want)
		}
	}
}

func TestDecoderCandleOrderOwnedConcurrent(t *testing.T) {
	if !decoderCandleOrder() {
		t.Skip("Candle-order GEMM only admitted on amd64 with SGEMM assembly")
	}
	conv := decoderConv1D{inChannels: 64, outChannels: 6, k: 9, dilation: 2, weight: make([]float32, 64*6*9), bias: make([]float32, 6)}
	trans := decoderTransConv1D{inChannels: 520, outChannels: 3, k: 4, stride: 2, weight: make([]float32, 520*3*4), bias: make([]float32, 3)}
	for i := range conv.weight {
		conv.weight[i] = float32(i%17-8) / 64
	}
	for i := range trans.weight {
		trans.weight[i] = float32(i%19-9) / 128
	}
	for i := range conv.bias {
		conv.bias[i] = float32(i-3) / 16
	}
	for i := range trans.bias {
		trans.bias[i] = float32(i-1) / 16
	}
	x := make([]float32, 64*101)
	y := make([]float32, 520*7)
	for i := range x {
		x[i] = float32(i%23-11) / 16
	}
	for i := range y {
		y[i] = float32(i%29-14) / 32
	}
	convWeights, transWeights := slices.Clone(conv.weight), slices.Clone(trans.weight)
	convInput, transInput := slices.Clone(x), slices.Clone(y)
	check := func() bool {
		a, n, err := conv.forward(x, 101)
		if err != nil || n != 101 || len(a) != 606 {
			return false
		}
		b, m, err := trans.forward(y, 7)
		if err != nil || m != 14 || len(b) != 42 {
			return false
		}
		for _, v := range append(a, b...) {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return false
			}
		}
		a2, _, err := conv.forward(x, 101)
		if err != nil || !slices.Equal(a, a2) {
			return false
		}
		b2, _, err := trans.forward(y, 7)
		return err == nil && slices.Equal(b, b2)
	}
	var wg sync.WaitGroup
	errs := make(chan bool, 8)
	for range 8 {
		wg.Add(1)
		go func() { defer wg.Done(); errs <- check() }()
	}
	wg.Wait()
	close(errs)
	for ok := range errs {
		if !ok {
			t.Fatal("non-deterministic concurrent decoder GEMM")
		}
	}
	if !slices.Equal(x, convInput) || !slices.Equal(y, transInput) || !slices.Equal(conv.weight, convWeights) || !slices.Equal(trans.weight, transWeights) {
		t.Fatal("decoder GEMM mutated input or weights")
	}
	if _, _, err := conv.forward(x[:len(x)-1], 101); err == nil {
		t.Fatal("accepted short convolution input")
	}
	invalid := conv
	invalid.dilation = 0
	if _, _, err := invalid.forward(x, 101); err == nil {
		t.Fatal("accepted zero dilation")
	}
	invalid = conv
	invalid.outChannels = int(^uint(0) >> 1)
	if _, _, err := invalid.forward(x, 101); err == nil {
		t.Fatal("accepted overflowing output geometry")
	}
	if _, _, err := trans.forward(y[:len(y)-1], 7); err == nil {
		t.Fatal("accepted short transposed convolution input")
	}
}
