package gguf

import (
	"math/rand"
	"testing"
)

func BenchmarkDotQ4_0Q8_0LlamaTiles(b *testing.B) {
	const blocks = 80
	raw, y := makeLlamaQ4ExperimentInput(blocks, 0x607)
	q4, _ := packQ4_0x8(raw, 8, blocks)
	q8, _ := packQ8_0x4(y, 4, blocks)
	flat8 := make([]q8_0Block, 8*blocks)
	for token := 0; token < 8; token++ {
		copy(flat8[token*blocks:], y[(token%4)*blocks:(token%4+1)*blocks])
	}
	tiles8 := tileQ8_0Tokens8(flat8, blocks)
	q4Lane := make([][]byte, 2)
	q8Lane := make([][]byte, 2)
	for group := 0; group < 2; group++ {
		q4Lane[group] = packQ4_0Rows4LaneTransposed(raw[group*4*blocks*18:], blocks*18, blocks)
		q8Lane[group] = packQ8_0Tokens2LaneTransposed(y[group*2*blocks:], blocks)
	}

	b.Run("retained-4row-8token-equal-work", func(b *testing.B) {
		var out [8]float32
		if !dotQ4_0Q8_0Tokens8SoA(raw[:blocks*18], tiles8, blocks, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for row := 0; row < 4; row++ {
				dotQ4_0Q8_0Tokens8SoA(raw[row*blocks*18:], tiles8, blocks, &out)
			}
		}
	})
	b.Run("output-major-8row-4token-equal-work", func(b *testing.B) {
		var out [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2(raw, blocks*18, y, blocks, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for rows := 0; rows < 2; rows++ {
				for tokens := 0; tokens < 2; tokens++ {
					dotQ4_0Q8_0Rows4Tokens2(raw[rows*4*blocks*18:], blocks*18, y[tokens*2*blocks:], blocks, &out)
				}
			}
		}
	})
	b.Run("lane-transposed-8row-4token-equal-work", func(b *testing.B) {
		var lanes [8][8]float32
		var out [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4Lane[0], q8Lane[0], blocks, &lanes, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for rows := 0; rows < 2; rows++ {
				for tokens := 0; tokens < 2; tokens++ {
					dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4Lane[rows], q8Lane[tokens], blocks, &lanes, &out)
				}
			}
		}
	})
	b.Run("llama-b607-8row-4token", func(b *testing.B) {
		var out [32]float32
		if err := dotQ4_0x8Q8_0x4LlamaVNNI(q4, q8, blocks, &out); err != nil {
			b.Skip(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dotQ4_0x8Q8_0x4LlamaVNNI(q4, q8, blocks, &out)
		}
	})
}

func BenchmarkDotQ4_0Q8_0LlamaProjection(b *testing.B) {
	const (
		blocks = 80
		rows   = 128
		tokens = 124
	)
	rng := rand.New(rand.NewSource(0x607607))
	baseRaw, baseY := makeLlamaQ4ExperimentInput(blocks, rng.Int63())
	raw := make([]byte, rows*blocks*18)
	for row := 0; row < rows; row++ {
		copy(raw[row*blocks*18:], baseRaw[(row%8)*blocks*18:(row%8+1)*blocks*18])
	}
	y := make([]q8_0Block, tokens*blocks)
	for token := 0; token < tokens; token++ {
		copy(y[token*blocks:], baseY[(token%4)*blocks:(token%4+1)*blocks])
	}
	q4, _ := packQ4_0x8(raw, rows, blocks)
	q8, _ := packQ8_0x4(y, tokens, blocks)
	x := make([]float32, tokens*blocks*qk8_0)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 8
	}
	retained := &QuantMatrix{Name: "benchmark", QType: QuantQ4_0, Raw: raw, InDim: blocks * qk8_0, OutDim: rows}
	out := make([]float32, rows*tokens)
	if err := projectQ4_0LlamaExperimental(q4, q8, rows, tokens, blocks, out); err != nil {
		b.Skip(err)
	}

	b.Run("prepacked", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			projectQ4_0LlamaExperimental(q4, q8, rows, tokens, blocks, out)
		}
	})
	b.Run("activation-pack-included", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			packQ8_0x4To(q8, y, tokens, blocks)
			projectQ4_0LlamaExperimental(q4, q8, rows, tokens, blocks, out)
		}
	})
	b.Run("all-pack-included", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			packQ4_0x8To(q4, raw, rows, blocks)
			packQ8_0x4To(q8, y, tokens, blocks)
			projectQ4_0LlamaExperimental(q4, q8, rows, tokens, blocks, out)
		}
	})
	b.Run("retained-f32-production", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := retained.ProjectBatchF32To(out, x, tokens); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("fused-direct-q8", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := projectBatchQ4_0LlamaExperimental(q4, out, x, rows, tokens, blocks*qk8_0); err != nil {
				b.Fatal(err)
			}
		}
	})
}
