package gguf

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

const laneTransposedPanelBytes = 72

// packQ4_0Rows4LaneTransposed packs four ordinary row-major Q4_0 rows into
// block-major panels. Each panel is four FP16 scales followed by four 16-byte
// nibble payloads, so packing is size-preserving.
func packQ4_0Rows4LaneTransposed(raw []byte, rowBytes, blocks int) []byte {
	packed := make([]byte, blocks*laneTransposedPanelBytes)
	for bi := 0; bi < blocks; bi++ {
		panel := packed[bi*laneTransposedPanelBytes:]
		for row := 0; row < 4; row++ {
			block := raw[row*rowBytes+bi*18:]
			copy(panel[row*2:row*2+2], block[:2])
			copy(panel[8+row*16:8+(row+1)*16], block[2:18])
		}
	}
	return packed
}

// packQ8_0Tokens2LaneTransposed packs two token-major Q8_0 rows into
// block-major panels. Each panel is two FP32 scales followed by two 32-byte
// payloads, retaining the original 72-byte footprint.
func packQ8_0Tokens2LaneTransposed(y []q8_0Block, blocks int) []byte {
	packed := make([]byte, blocks*laneTransposedPanelBytes)
	for bi := 0; bi < blocks; bi++ {
		panel := packed[bi*laneTransposedPanelBytes:]
		for token := 0; token < 2; token++ {
			block := &y[token*blocks+bi]
			binary.LittleEndian.PutUint32(panel[token*4:], math.Float32bits(block.d))
			for i, q := range block.qs {
				panel[8+token*qk8_0+i] = byte(q)
			}
		}
	}
	return packed
}

func dotQ4_0Q8_0Rows4Tokens2LaneReference(q4Packed, q8Packed []byte, blocks int) (lanes [8][8]float32, out [8]float32) {
	for bi := 0; bi < blocks; bi++ {
		q4 := q4Packed[bi*laneTransposedPanelBytes:]
		q8 := q8Packed[bi*laneTransposedPanelBytes:]
		for token := 0; token < 2; token++ {
			yScale := math.Float32frombits(binary.LittleEndian.Uint32(q8[token*4:]))
			yQuants := q8[8+token*qk8_0 : 8+(token+1)*qk8_0]
			for row := 0; row < 4; row++ {
				output := token*4 + row
				xScale := half.F16ToF32(binary.LittleEndian.Uint16(q4[row*2:]))
				d := xScale * yScale
				xQuants := q4[8+row*16 : 8+(row+1)*16]
				for lane := 0; lane < 4; lane++ {
					j := lane * 4
					s := (int(xQuants[j]&0x0f)-8)*int(int8(yQuants[j])) +
						(int(xQuants[j+1]&0x0f)-8)*int(int8(yQuants[j+1])) +
						(int(xQuants[j+2]&0x0f)-8)*int(int8(yQuants[j+2])) +
						(int(xQuants[j+3]&0x0f)-8)*int(int8(yQuants[j+3]))
					lanes[output][lane] = float32(math.FMA(float64(d), float64(float32(s)), float64(lanes[output][lane])))
					s = (int(xQuants[j]>>4)-8)*int(int8(yQuants[j+16])) +
						(int(xQuants[j+1]>>4)-8)*int(int8(yQuants[j+17])) +
						(int(xQuants[j+2]>>4)-8)*int(int8(yQuants[j+18])) +
						(int(xQuants[j+3]>>4)-8)*int(int8(yQuants[j+19]))
					lanes[output][lane+4] = float32(math.FMA(float64(d), float64(float32(s)), float64(lanes[output][lane+4])))
				}
			}
		}
	}
	for i := range out {
		out[i] = reduceQ4_0Q8_0LanesReference(&lanes[i])
	}
	return lanes, out
}

func reduceQ4_0Q8_0LanesReference(lanes *[8]float32) float32 {
	r0 := lanes[0] + lanes[4]
	r1 := lanes[1] + lanes[5]
	r2 := lanes[2] + lanes[6]
	r3 := lanes[3] + lanes[7]
	r0 += r2
	r1 += r3
	return r0 + r1
}

func randomRows4Tokens2(rng *rand.Rand, blocks int) ([]byte, []q8_0Block) {
	rowBytes := blocks * 18
	raw := make([]byte, 4*rowBytes)
	for row := 0; row < 4; row++ {
		for bi := 0; bi < blocks; bi++ {
			block := raw[row*rowBytes+bi*18:]
			binary.LittleEndian.PutUint16(block, half.F32ToF16((rng.Float32()*2-1)*0.25))
			for i := 2; i < 18; i++ {
				block[i] = byte(rng.Intn(256))
			}
		}
	}
	y := make([]q8_0Block, 2*blocks)
	for i := range y {
		y[i].d = (rng.Float32()*2 - 1) * 0.25
		for j := range y[i].qs {
			y[i].qs[j] = int8(rng.Intn(256) - 128)
		}
	}
	return raw, y
}

func TestLaneTransposedPanelPacking(t *testing.T) {
	const blocks = 2
	raw, y := randomRows4Tokens2(rand.New(rand.NewSource(0x4a2)), blocks)
	q4 := packQ4_0Rows4LaneTransposed(raw, blocks*18, blocks)
	q8 := packQ8_0Tokens2LaneTransposed(y, blocks)
	if len(q4) != 4*blocks*18 || len(q8) != 2*blocks*36 {
		t.Fatalf("packed sizes q4=%d q8=%d", len(q4), len(q8))
	}
	for bi := 0; bi < blocks; bi++ {
		for row := 0; row < 4; row++ {
			src := raw[row*blocks*18+bi*18:]
			panel := q4[bi*laneTransposedPanelBytes:]
			if string(panel[row*2:row*2+2]) != string(src[:2]) || string(panel[8+row*16:8+(row+1)*16]) != string(src[2:18]) {
				t.Fatalf("Q4 panel mismatch block=%d row=%d", bi, row)
			}
		}
		for token := 0; token < 2; token++ {
			block := y[token*blocks+bi]
			panel := q8[bi*laneTransposedPanelBytes:]
			if got := math.Float32frombits(binary.LittleEndian.Uint32(panel[token*4:])); got != block.d {
				t.Fatalf("Q8 scale mismatch block=%d token=%d got=%g want=%g", bi, token, got, block.d)
			}
			for j, want := range block.qs {
				if got := int8(panel[8+token*qk8_0+j]); got != want {
					t.Fatalf("Q8 quant mismatch block=%d token=%d index=%d got=%d want=%d", bi, token, j, got, want)
				}
			}
		}
	}
}

func TestDotQ4_0Q8_0Rows4Tokens2LaneTransposedRandomExact(t *testing.T) {
	rng := rand.New(rand.NewSource(0x4a2e))
	for iteration := 0; iteration < 100; iteration++ {
		blocks := 1 + rng.Intn(80)
		raw, y := randomRows4Tokens2(rng, blocks)
		q4 := packQ4_0Rows4LaneTransposed(raw, blocks*18, blocks)
		q8 := packQ8_0Tokens2LaneTransposed(y, blocks)
		wantLanes, wantOut := dotQ4_0Q8_0Rows4Tokens2LaneReference(q4, q8, blocks)
		var gotLanes [8][8]float32
		var gotOut [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4, q8, blocks, &gotLanes, &gotOut) {
			t.Skip("AVX-VNNI unavailable")
		}
		if gotLanes != wantLanes {
			for output := range gotLanes {
				for lane := range gotLanes[output] {
					if gotLanes[output][lane] != wantLanes[output][lane] {
						t.Fatalf("iteration=%d blocks=%d output=%d lane=%d got=%g want=%g", iteration, blocks, output, lane, gotLanes[output][lane], wantLanes[output][lane])
					}
				}
			}
		}
		if gotOut != wantOut {
			t.Fatalf("iteration=%d blocks=%d output=%v want=%v", iteration, blocks, gotOut, wantOut)
		}
		var existing [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2(raw, blocks*18, y, blocks, &existing) {
			t.Fatal("existing AVX-VNNI path became unavailable")
		}
		if gotOut != existing {
			t.Fatalf("iteration=%d blocks=%d lane-transposed=%v output-major=%v", iteration, blocks, gotOut, existing)
		}
	}
}

func TestDotQ4_0Q8_0Rows4Tokens2LaneTransposedEdges(t *testing.T) {
	var lanes [8][8]float32
	var out [8]float32
	if !dotQ4_0Q8_0Rows4Tokens2LaneTransposed(nil, nil, 0, &lanes, &out) {
		t.Skip("AVX-VNNI unavailable")
	}
	if lanes != [8][8]float32{} || out != [8]float32{} {
		t.Fatalf("zero-block result lanes=%v out=%v", lanes, out)
	}
	if dotQ4_0Q8_0Rows4Tokens2LaneTransposed(make([]byte, 71), make([]byte, 72), 1, &lanes, &out) ||
		dotQ4_0Q8_0Rows4Tokens2LaneTransposed(make([]byte, 72), make([]byte, 71), 1, &lanes, &out) ||
		dotQ4_0Q8_0Rows4Tokens2LaneTransposed(nil, nil, -1, &lanes, &out) {
		t.Fatal("malformed packed input accepted")
	}
}

func BenchmarkDotQ4_0Q8_0LaneTransposedTiles(b *testing.B) {
	const blocks = 80
	raw, y := randomRows4Tokens2(rand.New(rand.NewSource(0x4a2b)), blocks)
	q4 := packQ4_0Rows4LaneTransposed(raw, blocks*18, blocks)
	q8 := packQ8_0Tokens2LaneTransposed(y, blocks)
	flat8 := make([]q8_0Block, 8*blocks)
	for token := 0; token < 8; token++ {
		copy(flat8[token*blocks:], y[:blocks])
	}
	tiles8 := tileQ8_0Tokens8(flat8, blocks)

	b.Run("retained-1row-8token", func(b *testing.B) {
		var out [8]float32
		if !dotQ4_0Q8_0Tokens8SoA(raw[:blocks*18], tiles8, blocks, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		for i := 0; i < b.N; i++ {
			dotQ4_0Q8_0Tokens8SoA(raw[:blocks*18], tiles8, blocks, &out)
		}
	})
	b.Run("output-major-4row-2token", func(b *testing.B) {
		var out [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2(raw, blocks*18, y, blocks, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		for i := 0; i < b.N; i++ {
			dotQ4_0Q8_0Rows4Tokens2(raw, blocks*18, y, blocks, &out)
		}
	})
	b.Run("lane-transposed-asm-lanes", func(b *testing.B) {
		var lanes [8][8]float32
		var out [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4, q8, blocks, &lanes, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		for i := 0; i < b.N; i++ {
			dotQ4_0Q8_0Rows4Tokens2LaneStates(q4, q8, blocks, &lanes)
		}
	})
	b.Run("lane-transposed-4row-2token", func(b *testing.B) {
		var lanes [8][8]float32
		var out [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4, q8, blocks, &lanes, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		for i := 0; i < b.N; i++ {
			dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4, q8, blocks, &lanes, &out)
		}
	})
}

func BenchmarkDotQ4_0Q8_0LaneTransposedProjection(b *testing.B) {
	const (
		blocks = 80
		rows   = 128
		batch  = 124
	)
	rng := rand.New(rand.NewSource(0x4a2c))
	baseRaw, baseY := randomRows4Tokens2(rng, blocks)
	rowBytes := blocks * 18
	raw := make([]byte, rows*rowBytes)
	for row := 0; row < rows; row++ {
		copy(raw[row*rowBytes:], baseRaw[(row%4)*rowBytes:(row%4+1)*rowBytes])
	}
	y := make([]q8_0Block, batch*blocks)
	for token := 0; token < batch; token++ {
		copy(y[token*blocks:], baseY[(token%2)*blocks:(token%2+1)*blocks])
	}

	q4Panels := make([]byte, rows/4*blocks*laneTransposedPanelBytes)
	for group := 0; group < rows/4; group++ {
		packed := packQ4_0Rows4LaneTransposed(raw[group*4*rowBytes:], rowBytes, blocks)
		copy(q4Panels[group*blocks*laneTransposedPanelBytes:], packed)
	}
	q8Panels := make([]byte, batch/2*blocks*laneTransposedPanelBytes)
	for pair := 0; pair < batch/2; pair++ {
		packed := packQ8_0Tokens2LaneTransposed(y[pair*2*blocks:], blocks)
		copy(q8Panels[pair*blocks*laneTransposedPanelBytes:], packed)
	}
	fullTokens := batch / 8 * 8
	tiles8 := make([]q8_0Tile8, fullTokens/8*blocks)
	for pos := 0; pos < fullTokens; pos += 8 {
		copy(tiles8[pos/8*blocks:], tileQ8_0Tokens8(y[pos*blocks:], blocks))
	}

	b.Run("retained-1row-8token", func(b *testing.B) {
		var out8 [8]float32
		var out4 [4]float32
		if !dotQ4_0Q8_0Tokens8SoA(raw[:rowBytes], tiles8, blocks, &out8) {
			b.Skip("AVX-VNNI unavailable")
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for row := 0; row < rows; row++ {
				rowRaw := raw[row*rowBytes : (row+1)*rowBytes]
				for pos := 0; pos < fullTokens; pos += 8 {
					dotQ4_0Q8_0Tokens8SoA(rowRaw, tiles8[pos/8*blocks:], blocks, &out8)
				}
				dotQ4_0Q8_0Tokens4(rowRaw, y[fullTokens*blocks:], blocks, &out4)
			}
		}
	})
	b.Run("output-major-4row-2token", func(b *testing.B) {
		var out [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2(raw[:4*rowBytes], rowBytes, y, blocks, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for group := 0; group < rows/4; group++ {
				groupRaw := raw[group*4*rowBytes : (group+1)*4*rowBytes]
				for pair := 0; pair < batch/2; pair++ {
					dotQ4_0Q8_0Rows4Tokens2(groupRaw, rowBytes, y[pair*2*blocks:], blocks, &out)
				}
			}
		}
	})
	b.Run("lane-transposed-asm-lanes", func(b *testing.B) {
		var lanes [8][8]float32
		var out [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4Panels[:blocks*laneTransposedPanelBytes], q8Panels, blocks, &lanes, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for group := 0; group < rows/4; group++ {
				q4 := q4Panels[group*blocks*laneTransposedPanelBytes:]
				for pair := 0; pair < batch/2; pair++ {
					q8 := q8Panels[pair*blocks*laneTransposedPanelBytes:]
					dotQ4_0Q8_0Rows4Tokens2LaneStates(q4, q8, blocks, &lanes)
				}
			}
		}
	})
	b.Run("lane-transposed-4row-2token", func(b *testing.B) {
		var lanes [8][8]float32
		var out [8]float32
		if !dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4Panels[:blocks*laneTransposedPanelBytes], q8Panels, blocks, &lanes, &out) {
			b.Skip("AVX-VNNI unavailable")
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for group := 0; group < rows/4; group++ {
				q4 := q4Panels[group*blocks*laneTransposedPanelBytes:]
				for pair := 0; pair < batch/2; pair++ {
					q8 := q8Panels[pair*blocks*laneTransposedPanelBytes:]
					dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4, q8, blocks, &lanes, &out)
				}
			}
		}
	})
}
