package gguf

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func makeLlamaQ4ExperimentInput(blocks int, seed int64) ([]byte, []q8_0Block) {
	rng := rand.New(rand.NewSource(seed))
	raw := make([]byte, 8*blocks*18)
	for row := 0; row < 8; row++ {
		for block := 0; block < blocks; block++ {
			p := raw[(row*blocks+block)*18:]
			binary.LittleEndian.PutUint16(p, half.F32ToF16((rng.Float32()*2-1)*0.25))
			for i := 2; i < 18; i++ {
				p[i] = byte(rng.Uint32())
			}
		}
	}
	y := make([]q8_0Block, 4*blocks)
	for token := 0; token < 4; token++ {
		for block := 0; block < blocks; block++ {
			p := &y[token*blocks+block]
			p.d = half.F16ToF32(half.F32ToF16((rng.Float32()*2 - 1) * 0.125))
			for i := range p.qs {
				p.qs[i] = int8(rng.Intn(255) - 127)
			}
		}
	}
	return raw, y
}

func TestPackQ4_0x8ByteLayout(t *testing.T) {
	raw := make([]byte, 8*2*18)
	for row := 0; row < 8; row++ {
		for block := 0; block < 2; block++ {
			p := raw[(row*2+block)*18:]
			binary.LittleEndian.PutUint16(p, uint16(0x1000+row*0x10+block))
			for i := 0; i < 16; i++ {
				p[2+i] = byte(row*32 + block*16 + i)
			}
		}
	}
	got, err := packQ4_0x8(raw, 8, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2*q4_0x8BlockBytes {
		t.Fatalf("packed bytes=%d", len(got))
	}
	for block := 0; block < 2; block++ {
		p := got[block*q4_0x8BlockBytes:]
		for row := 0; row < 8; row++ {
			if scale := binary.LittleEndian.Uint16(p[row*2:]); scale != uint16(0x1000+row*0x10+block) {
				t.Fatalf("block %d row %d scale=%#x", block, row, scale)
			}
			for chunk := 0; chunk < 2; chunk++ {
				for j := 0; j < 8; j++ {
					want := byte(row*32+block*16+chunk*8+j) ^ 0x88
					if q := p[16+(chunk*8+row)*8+j]; q != want {
						t.Fatalf("block %d row %d chunk %d byte %d=%#x want %#x", block, row, chunk, j, q, want)
					}
				}
			}
		}
	}
}

func TestPackQ8_0x4ByteLayout(t *testing.T) {
	y := make([]q8_0Block, 4*2)
	for token := 0; token < 4; token++ {
		for block := 0; block < 2; block++ {
			p := &y[token*2+block]
			p.d = float32(token*2+block+1) / 32
			for i := range p.qs {
				p.qs[i] = int8(token*40 + block*32 + i - 100)
			}
		}
	}
	got, err := packQ8_0x4(y, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2*q8_0x4BlockBytes {
		t.Fatalf("packed bytes=%d", len(got))
	}
	for block := 0; block < 2; block++ {
		p := got[block*q8_0x4BlockBytes:]
		for token := 0; token < 4; token++ {
			wantD := half.F32ToF16(y[token*2+block].d)
			if d := binary.LittleEndian.Uint16(p[token*2:]); d != wantD {
				t.Fatalf("block %d token %d scale=%#x want %#x", block, token, d, wantD)
			}
			for chunk := 0; chunk < 4; chunk++ {
				for j := 0; j < 8; j++ {
					want := byte(y[token*2+block].qs[chunk*8+j])
					if q := p[8+(chunk*4+token)*8+j]; q != want {
						t.Fatalf("block %d token %d chunk %d byte %d=%#x want %#x", block, token, chunk, j, q, want)
					}
				}
			}
		}
	}
}

func TestLlamaPackersRejectMalformedSizes(t *testing.T) {
	if _, err := packQ4_0x8(make([]byte, 17), 1, 1); err == nil {
		t.Fatal("short Q4 source accepted")
	}
	if err := packQ4_0x8To(make([]byte, q4_0x8BlockBytes-1), make([]byte, 18), 1, 1); err == nil {
		t.Fatal("short Q4 destination accepted")
	}
	if _, err := packQ8_0x4(make([]q8_0Block, 3), 4, 1); err == nil {
		t.Fatal("short Q8 source accepted")
	}
	if err := packQ8_0x4To(make([]byte, q8_0x4BlockBytes-1), make([]q8_0Block, 4), 4, 1); err == nil {
		t.Fatal("short Q8 destination accepted")
	}
}

func TestLlamaQ4_0x8Q8_0x4VNNIReference(t *testing.T) {
	for blocks := 1; blocks <= 80; blocks++ {
		raw, y := makeLlamaQ4ExperimentInput(blocks, int64(9000+blocks))
		q4, err := packQ4_0x8(raw, 8, blocks)
		if err != nil {
			t.Fatal(err)
		}
		q8, err := packQ8_0x4(y, 4, blocks)
		if err != nil {
			t.Fatal(err)
		}
		var want, got [32]float32
		if err := dotQ4_0x8Q8_0x4LlamaReference(q4, q8, blocks, &want); err != nil {
			t.Fatal(err)
		}
		if err := dotQ4_0x8Q8_0x4LlamaVNNI(q4, q8, blocks, &got); err != nil {
			t.Skip(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("blocks=%d output=%d got=%g (%08x) want=%g (%08x)", blocks, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
}

func TestLlamaProjectionSupertileAndTails(t *testing.T) {
	const (
		rows   = 13
		tokens = 19
		blocks = 3
	)
	baseRaw, baseY := makeLlamaQ4ExperimentInput(blocks, 31337)
	raw := make([]byte, rows*blocks*18)
	for row := 0; row < rows; row++ {
		copy(raw[row*blocks*18:], baseRaw[(row%8)*blocks*18:(row%8+1)*blocks*18])
	}
	y := make([]q8_0Block, tokens*blocks)
	for token := 0; token < tokens; token++ {
		copy(y[token*blocks:], baseY[(token%4)*blocks:(token%4+1)*blocks])
	}
	q4, err := packQ4_0x8(raw, rows, blocks)
	if err != nil {
		t.Fatal(err)
	}
	q8, err := packQ8_0x4(y, tokens, blocks)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, rows*tokens)
	if err := projectQ4_0LlamaExperimental(q4, q8, rows, tokens, blocks, got); err != nil {
		t.Skip(err)
	}
	for tg := 0; tg < (tokens+3)/4; tg++ {
		for rg := 0; rg < (rows+7)/8; rg++ {
			var tile [32]float32
			if err := dotQ4_0x8Q8_0x4LlamaReference(q4[(rg*blocks)*q4_0x8BlockBytes:(rg+1)*blocks*q4_0x8BlockBytes], q8[(tg*blocks)*q8_0x4BlockBytes:(tg+1)*blocks*q8_0x4BlockBytes], blocks, &tile); err != nil {
				t.Fatal(err)
			}
			for token := 0; token < 4 && tg*4+token < tokens; token++ {
				for row := 0; row < 8 && rg*8+row < rows; row++ {
					idx := (tg*4+token)*rows + rg*8 + row
					if math.Float32bits(got[idx]) != math.Float32bits(tile[token*8+row]) {
						t.Fatalf("token=%d row=%d got=%g want=%g", tg*4+token, rg*8+row, got[idx], tile[token*8+row])
					}
				}
			}
		}
	}
}

func TestLlamaTopologyDivergesFromLegacyLaneReduction(t *testing.T) {
	const blocks = 80
	raw, y := makeLlamaQ4ExperimentInput(blocks, 117)
	q4, _ := packQ4_0x8(raw, 8, blocks)
	q8, _ := packQ8_0x4(y, 4, blocks)
	var llama [32]float32
	if err := dotQ4_0x8Q8_0x4LlamaReference(q4, q8, blocks, &llama); err != nil {
		t.Fatal(err)
	}
	diverged := 0
	for token := 0; token < 4; token++ {
		for row := 0; row < 8; row++ {
			legacy := dotQ4_0Q8_0Scalar(raw[row*blocks*18:], y[token*blocks:], blocks)
			if math.Float32bits(legacy) != math.Float32bits(llama[token*8+row]) {
				diverged++
			}
		}
	}
	if diverged == 0 {
		t.Fatal("llama topology unexpectedly matched all legacy lane reductions")
	}
	t.Logf("documented non-exactness boundary: %d/32 outputs diverged", diverged)
}
