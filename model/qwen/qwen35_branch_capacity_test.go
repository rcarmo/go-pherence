package qwen

import (
	"context"
	"strings"
	"testing"

	cfg "github.com/rcarmo/go-pherence/loader/config"
)

func TestSIMDBranchCapacityBoundary(t *testing.T) {
	// A non-nil model is enough for pre-scratch forward validation. No tensor
	// payload is allocated, and every rejected call must leave output unchanged.
	for _, tc := range []struct{ capacity, rows int }{{512, 513}, {1024, 1025}, {4096, 4097}} {
		s := &Qwen35SIMDBranch{model: &Qwen35BaseModel{}, maxTokens: tc.capacity}
		dst := make([]float32, tc.rows*1024)
		for i := range dst {
			dst[i] = 17
		}
		err := s.ForwardTreeInto(dst, make([][]float32, tc.rows), 1, 1, []int{tc.rows}, make([]float32, tc.rows*64), 1e-6)
		if err == nil {
			t.Fatal("over-capacity call accepted", tc)
		}
		for _, v := range dst {
			if v != 17 {
				t.Fatal("rejected call changed output")
			}
		}
	}
	for _, n := range []int{513, 1024, 4096} {
		// Cancellation must precede metadata access, including newly admitted sizes.
		s := &Qwen35SIMDBranch{model: &Qwen35BaseModel{}, maxTokens: n}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := s.ForwardTreeIntoContext(ctx, make([]float32, n*1024), make([][]float32, n), 1, 1, []int{n}, nil, 1e-6); err != context.Canceled {
			t.Fatal("cancellation", n, err)
		}
	}
}

func TestSIMDValidationKeepsGPULimit(t *testing.T) {
	meta := cfg.QwenNativeMTPMetadata{NumHiddenLayers: 24, HiddenSize: 1024, IntermediateSize: 3584, NumAttentionHeads: 8, NumKeyValueHeads: 2, HeadDim: 256, LinearNumKeyHeads: 16, LinearNumValueHeads: 16, LinearKeyHeadDim: 128, LinearValueHeadDim: 128, LinearConvKernelDim: 4, PartialRotaryFactor: .25, ZeroCenteredRMSNorm: true}
	m := &Qwen35BaseModel{Layers: make([]Qwen35BaseLayer, 24)}
	// Deliberately malformed layers distinguish an admitted CPU capacity from
	// the GPU/configuration rejection without allocating full model weights.
	for _, n := range []int{-1, 0, 2, 4097} {
		if s, err := NewQwen35SIMDBranch(&Qwen35BaseModel{}, meta, n); err == nil || s != nil {
			t.Fatal("invalid CPU capacity", n)
		}
	}
	for _, n := range []int{513, 1024, 4096} {
		err := ValidateQwen35F32Branch(m, meta, n)
		if err == nil || !strings.Contains(err.Error(), "unsupported accelerated branch configuration") {
			t.Fatal("GPU cap changed", n, err)
		}
		err = ValidateQwen35SIMDBranch(m, meta, n)
		if err == nil || strings.Contains(err.Error(), "unsupported accelerated branch configuration") {
			t.Fatal("CPU capacity rejected before layer checks", n, err)
		}
	}
}
