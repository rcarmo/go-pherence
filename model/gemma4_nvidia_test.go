package model

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestGemma4NVIDIAIndependentBranchesMatchSIMDOracle(t *testing.T) {
	if !nvidia.SgemmReady() {
		if nvidia.Available() {
			t.Fatal("CUDA device available but PTX runtime not ready")
		}
		t.Skip("CUDA unavailable")
	}
	m := newGemma4NVIDIAQuantTestModel(t)
	prompt := []int{1, 2}
	branches := [][]int{{0}, {1}, {2, 0}, {2, 1}}
	ctx, err := m.BuildPreparedMTPPromptContext(prompt)
	if err != nil {
		t.Fatal(err)
	}
	gpu, err := NewGemma4NVIDIA(m)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	got, err := gpu.ScoreIndependentBranches(context.Background(), ctx, branches)
	if err != nil {
		t.Fatal(err)
	}
	s := newGemma4DecodeSessionForTest(t, m, 2, Gemma4PromptCacheConfig{})
	if err := s.BeginPreparedPrefill(prompt); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PrefillNext(s.RemainingPrefill()); err != nil {
		t.Fatal(err)
	}
	cp, err := s.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	for i, b := range branches {
		if err := s.Restore(cp); err != nil {
			t.Fatal(err)
		}
		var want []float32
		for _, tok := range b {
			step, err := s.AppendToken(tok)
			if err != nil {
				t.Fatal(err)
			}
			want = step.Logits
		}
		assertFloat32RowsRelative(t, got.Logits[i], want, 1e-2, 5e-3)
		if argmaxFloat32(got.Logits[i]) != argmaxFloat32(want) {
			t.Fatalf("branch %d argmax=%d want %d", i, argmaxFloat32(got.Logits[i]), argmaxFloat32(want))
		}
	}
}

func TestGemma4NVIDIARejectsPLIInsteadOfCPUFallback(t *testing.T) {
	m := newGemma4NVIDIAQuantTestModel(t)
	m.Config.HiddenPerLayer = 1
	m.PerLayerModelProj = []float32{1, 1}
	if _, err := NewGemma4NVIDIA(m); err == nil {
		t.Fatal("accepted unsupported PLI graph")
	}
}

func newGemma4NVIDIAQuantTestModel(t *testing.T) *LlamaModel {
	t.Helper()
	const h, vocab = 256, 3
	identity := make([]float32, h*h)
	for i := 0; i < h; i++ {
		identity[i*h+i] = 0.5
	}
	embed := make([]float32, vocab*h)
	for token := 0; token < vocab; token++ {
		for i := 0; i < h; i++ {
			embed[token*h+i] = float32(((token+1)*(i+3))%17-8) * 0.015
		}
	}
	lm := append([]float32(nil), embed...)
	layer := LlamaLayer{
		InputNorm: tensor.Ones([]int{h}), PostNorm: tensor.Ones([]int{h}),
		PreFFNNorm: tensor.Ones([]int{h}), PostFFNNorm: tensor.Ones([]int{h}),
		QNorm: tensor.Ones([]int{h}), KNorm: tensor.Ones([]int{h}),
		LayerScalar: 0.375, HasKV: true,
	}
	layer.QWGGUF = quantizeQ6KFixture(t, identity, h, h)
	layer.KWGGUF = quantizeQ5KFixture(t, identity, h, h)
	layer.VWGGUF = quantizeQ5KFixture(t, identity, h, h)
	layer.OWGGUF = quantizeQ4KFixture(t, identity, h, h)
	layer.GateWGGUF = quantizeQ4KFixture(t, identity, h, h)
	layer.UpWGGUF = quantizeQ5KFixture(t, identity, h, h)
	layer.DownWGGUF = quantizeQ6KFixture(t, identity, h, h)
	return &LlamaModel{
		Config:      LlamaConfig{ModelType: "gemma4_text", VocabSize: vocab, HiddenSize: h, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: h, Intermediate: h, RMSNormEps: 1e-6, HiddenAct: "gelu_pytorch_tanh", FinalLogitSoftcapping: 3},
		EmbedTokens: tensor.FromFloat32(embed, []int{vocab, h}), Norm: tensor.Ones([]int{h}),
		LMHeadGGUF: quantizeQ5KFixture(t, lm, h, vocab), Layers: []LlamaLayer{layer},
	}
}

func quantizeQ4KFixture(t *testing.T, data []float32, in, out int) *gguf.QuantMatrix {
	return quantizeKFixture(t, data, in, out, gguf.QuantQ4_K, 144, func(blk []byte, row []float32) {
		d := float32(0.02)
		binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(d))
		for i := 0; i < 12; i++ {
			blk[4+i] = 1
		}
		for i, v := range row {
			q := byte(math.Round(float64(v / d)))
			if q > 15 {
				q = 15
			}
			idx := i % 32
			if (i/32)%2 == 0 {
				blk[16+(i/64)*32+idx] |= q
			} else {
				blk[16+(i/64)*32+idx] |= q << 4
			}
		}
	})
}
func quantizeQ5KFixture(t *testing.T, data []float32, in, out int) *gguf.QuantMatrix {
	return quantizeKFixture(t, data, in, out, gguf.QuantQ5_K, 176, func(blk []byte, row []float32) {
		d := float32(0.02)
		binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(d))
		for i := 0; i < 12; i++ {
			blk[4+i] = 1
		}
		for i, v := range row {
			q := int(math.Round(float64(v / d)))
			if q < 0 {
				q = 0
			}
			if q > 31 {
				q = 31
			}
			group := i / 64
			lane := i % 32
			if i%64 < 32 {
				blk[48+group*32+lane] |= byte(q & 15)
				if q&16 != 0 {
					blk[16+lane] |= 1 << uint(group*2)
				}
			} else {
				blk[48+group*32+lane] |= byte(q&15) << 4
				if q&16 != 0 {
					blk[16+lane] |= 1 << uint(group*2+1)
				}
			}
		}
	})
}
func quantizeQ6KFixture(t *testing.T, data []float32, in, out int) *gguf.QuantMatrix {
	return quantizeKFixture(t, data, in, out, gguf.QuantQ6_K, 210, func(blk []byte, row []float32) {
		d := float32(0.02)
		binary.LittleEndian.PutUint16(blk[208:210], half.F32ToF16(d))
		for i := 0; i < 16; i++ {
			blk[192+i] = 1
		}
		for i, v := range row {
			q := int(math.Round(float64(v/d))) + 32
			if q < 0 {
				q = 0
			}
			if q > 63 {
				q = 63
			}
			halfBlock, group, lane := i/128, (i%128)/32, i%32
			idx := halfBlock*64 + (group%2)*32 + lane
			if group < 2 {
				blk[idx] |= byte(q & 15)
			} else {
				blk[idx] |= byte(q&15) << 4
			}
			blk[128+halfBlock*32+lane] |= byte((q>>4)&3) << uint(group*2)
		}
	})
}
func quantizeKFixture(t *testing.T, data []float32, in, out int, qt gguf.QuantType, blockSize int, fill func([]byte, []float32)) *gguf.QuantMatrix {
	t.Helper()
	if in%256 != 0 || len(data) != in*out {
		t.Fatal("bad fixture matrix")
	}
	blocks := in / 256
	raw := make([]byte, out*blocks*blockSize)
	for r := 0; r < out; r++ {
		for b := 0; b < blocks; b++ {
			row := data[r*in+b*256 : r*in+(b+1)*256]
			off := (r*blocks + b) * blockSize
			fill(raw[off:off+blockSize], row)
		}
	}
	return &gguf.QuantMatrix{Name: "fixture", QType: qt, Raw: raw, InDim: in, OutDim: out}
}
func argmaxFloat32(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

func assertFloat32RowsRelative(t *testing.T, got, want []float32, abs, rel float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len %d want %d", len(got), len(want))
	}
	for i := range got {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		limit := abs
		w := want[i]
		if w < 0 {
			w = -w
		}
		if rel*w > limit {
			limit = rel * w
		}
		if d > limit {
			t.Fatalf("[%d] got=%g want=%g diff=%g limit=%g", i, got[i], want[i], d, limit)
		}
	}
}
