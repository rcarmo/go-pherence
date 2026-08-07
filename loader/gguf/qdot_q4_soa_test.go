package gguf

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func tileQ8_0Tokens8(src []q8_0Block, blocks int) []q8_0Tile8 {
	tiles := make([]q8_0Tile8, blocks)
	for bi := 0; bi < blocks; bi++ {
		for token := 0; token < 8; token++ {
			q := src[token*blocks+bi]
			tiles[bi].d[token] = q.d
			tiles[bi].qs[token] = q.qs
		}
	}
	return tiles
}

func TestDotQ4_0Q8_0Tokens8SoARandomExact(t *testing.T) {
	rng := rand.New(rand.NewSource(0x80a5))
	for iteration := 0; iteration < 100; iteration++ {
		blocks := 1 + rng.Intn(80)
		raw := make([]byte, blocks*18)
		for bi := 0; bi < blocks; bi++ {
			block := raw[bi*18:]
			binary.LittleEndian.PutUint16(block, half.F32ToF16((rng.Float32()*2-1)*0.25))
			for j := 2; j < 18; j++ {
				block[j] = byte(rng.Intn(256))
			}
		}
		flat := make([]q8_0Block, 8*blocks)
		for i := range flat {
			flat[i].d = (rng.Float32()*2 - 1) * 0.25
			for j := range flat[i].qs {
				flat[i].qs[j] = int8(rng.Intn(256) - 128)
			}
		}
		tiles := tileQ8_0Tokens8(flat, blocks)
		var got, want [8]float32
		if !dotQ4_0Q8_0Tokens8SoA(raw, tiles, blocks, &got) || !dotQ4_0Q8_0Tokens8(raw, flat, blocks, &want) {
			t.Skip("AVX-VNNI unavailable")
		}
		if got != want {
			t.Fatalf("iteration=%d blocks=%d soa=%v want=%v", iteration, blocks, got, want)
		}
	}
}

func BenchmarkDotQ4_0Q8_0Tokens8SoA(b *testing.B) {
	const blocks = 80
	raw, base := syntheticQ4_0Q8_0DotInputs(blocks * qk8_0)
	flat := make([]q8_0Block, 0, 8*blocks)
	for token := 0; token < 8; token++ {
		flat = append(flat, base...)
	}
	tiles := tileQ8_0Tokens8(flat, blocks)
	var out [8]float32
	if !dotQ4_0Q8_0Tokens8SoA(raw, tiles, blocks, &out) {
		b.Skip("AVX-VNNI unavailable")
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotQ4_0Q8_0Tokens8SoA(raw, tiles, blocks, &out)
	}
}
