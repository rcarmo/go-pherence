package gguf

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf/llamaq4"
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

func TestQuantizeQ8_0x4MatchesQuantizeThenPack(t *testing.T) {
	for _, shape := range []struct {
		tokens int
		width  int
	}{{1, 32}, {3, 64}, {4, 96}, {5, 128}, {16, 256}, {19, 320}, {124, 2560}} {
		rng := rand.New(rand.NewSource(int64(shape.tokens*10000 + shape.width)))
		x := make([]float32, shape.tokens*shape.width)
		for i := range x {
			x[i] = (rng.Float32()*2 - 1) * 8
		}
		// Exercise the zero-scale path as well as ordinary random blocks.
		clear(x[:32])
		blocks := shape.width / qk8_0
		intermediate := make([]q8_0Block, shape.tokens*blocks)
		if err := quantizeQ8_0To(intermediate, x); err != nil {
			t.Fatal(err)
		}
		want, err := packQ8_0x4(intermediate, shape.tokens, blocks)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]byte, len(want))
		if err := quantizeQ8_0x4To(got, x, shape.tokens, shape.width); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("tokens=%d width=%d byte=%d got=%#x want=%#x", shape.tokens, shape.width, i, got[i], want[i])
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
	if err := quantizeQ8_0x4To(make([]byte, q8_0x4BlockBytes), make([]float32, 31), 1, 32); err == nil {
		t.Fatal("short F32 activation source accepted")
	}
	if err := quantizeQ8_0x4To(make([]byte, q8_0x4BlockBytes-1), make([]float32, 32), 1, 32); err == nil {
		t.Fatal("short direct Q8 destination accepted")
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

func TestLlamaFusedProjectionReference(t *testing.T) {
	const (
		rows   = 8
		tokens = 16
	)
	for blocks := 1; blocks <= 80; blocks++ {
		raw, baseY := makeLlamaQ4ExperimentInput(blocks, int64(16000+blocks))
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
		for panel := 0; panel < 4; panel++ {
			var want [32]float32
			start := panel * blocks * q8_0x4BlockBytes
			if err := dotQ4_0x8Q8_0x4LlamaReference(q4, q8[start:start+blocks*q8_0x4BlockBytes], blocks, &want); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				idx := panel*32 + i
				if math.Float32bits(got[idx]) != math.Float32bits(want[i]) {
					t.Fatalf("blocks=%d output=%d got=%g (%08x) want=%g (%08x)", blocks, idx, got[idx], math.Float32bits(got[idx]), want[i], math.Float32bits(want[i]))
				}
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

func TestLlamaProjectionDynamicRowScheduling(t *testing.T) {
	const (
		rows   = 129
		tokens = 19
		blocks = 3
	)
	baseRaw, baseY := makeLlamaQ4ExperimentInput(blocks, 424242)
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
	want := make([]float32, rows*tokens)
	if err := llamaq4.ProjectQ4_0x8Q8_0x4VNNI(q4, q8, rows, tokens, blocks, want); err != nil {
		t.Skip(err)
	}
	previous := runtime.GOMAXPROCS(6)
	defer runtime.GOMAXPROCS(previous)
	got := make([]float32, rows*tokens)
	if err := projectQ4_0LlamaExperimental(q4, q8, rows, tokens, blocks, got); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("output=%d got=%g (%08x) want=%g (%08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
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

func TestPrepareLlamaQ4_0x8ReplacementPreservesRowsAndProjection(t *testing.T) {
	const (
		rows   = 13
		blocks = 3
		tokens = 19
	)
	baseRaw, _ := makeLlamaQ4ExperimentInput(blocks, 0x4b607)
	raw := make([]byte, rows*blocks*18)
	for row := 0; row < rows; row++ {
		copy(raw[row*blocks*18:], baseRaw[(row%8)*blocks*18:(row%8+1)*blocks*18])
	}
	matrix := &QuantMatrix{Name: "replacement", QType: QuantQ4_0, Raw: append([]byte(nil), raw...), InDim: blocks * qk8_0, OutDim: rows}
	before := make([]float32, matrix.InDim)
	after := make([]float32, matrix.InDim)

	prepared, err := matrix.PrepareLlamaQ4_0x8()
	if err != nil {
		t.Fatal(err)
	}
	if !prepared {
		t.Skip("fused Q4_0x8 kernels unavailable")
	}
	if matrix.Raw != nil {
		t.Fatalf("canonical Raw retained after replacement: %d bytes", len(matrix.Raw))
	}
	if len(matrix.llamaQ4_0x8) != (rows+7)/8*blocks*q4_0x8BlockBytes {
		t.Fatalf("packed bytes=%d", len(matrix.llamaQ4_0x8))
	}
	canonical := &QuantMatrix{Name: "canonical", QType: QuantQ4_0, Raw: raw, InDim: matrix.InDim, OutDim: rows}
	for row := 0; row < rows; row++ {
		if err := canonical.DequantRowTo(before, row); err != nil {
			t.Fatal(err)
		}
		if err := matrix.DequantRowTo(after, row); err != nil {
			t.Fatal(err)
		}
		for i := range before {
			if math.Float32bits(before[i]) != math.Float32bits(after[i]) {
				t.Fatalf("row=%d col=%d before=%g after=%g", row, i, before[i], after[i])
			}
		}
	}

	rng := rand.New(rand.NewSource(0x8f16))
	x := make([]float32, tokens*matrix.InDim)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 4
	}
	got := make([]float32, tokens*rows)
	again := make([]float32, tokens*rows)
	if err := matrix.ProjectBatchF32To(got, x, tokens); err != nil {
		t.Fatal(err)
	}
	if err := matrix.ProjectBatchF32To(again, x, tokens); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Float32bits(got[i]) != math.Float32bits(again[i]) {
			t.Fatalf("non-finite or non-deterministic output %d: %g / %g", i, got[i], again[i])
		}
	}
}

func TestPrepareLlamaQ4_0x8KeepsSmallBatchContract(t *testing.T) {
	raw, _ := makeLlamaQ4ExperimentInput(1, 0x816)
	matrix := &QuantMatrix{Name: "small-batch", QType: QuantQ4_0, Raw: raw, InDim: qk8_0, OutDim: 8}
	prepared, err := matrix.PrepareLlamaQ4_0x8()
	if err != nil {
		t.Fatal(err)
	}
	if !prepared {
		t.Skip("fused Q4_0x8 kernels unavailable")
	}
	for _, batch := range []int{1, 2, 4} {
		err := matrix.ProjectBatchF32To(make([]float32, batch*matrix.OutDim), make([]float32, batch*matrix.InDim), batch)
		if !errors.Is(err, ErrUnsupportedBatchProjection) {
			t.Fatalf("batch=%d error=%v", batch, err)
		}
	}
}
