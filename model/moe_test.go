package model

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
)

func TestMoeForwardRejectsMalformedInputs(t *testing.T) {
	cfg := LlamaConfig{NumExperts: 0, NumExpertsPerTok: 1, MoEIntermediate: 4}
	if got := moeForward([]float32{1, 2}, &LlamaLayer{}, cfg); got != nil {
		t.Fatalf("zero experts output=%v, want nil", got)
	}
	cfg.NumExperts = 2
	if got := moeForward(nil, &LlamaLayer{}, cfg); got != nil {
		t.Fatalf("nil input output=%v, want nil", got)
	}
	if got := moeForward([]float32{1, 2}, nil, cfg); got != nil {
		t.Fatalf("nil layer output=%v, want nil", got)
	}
	cfg.MoEIntermediate = 0
	if got := moeForward([]float32{1, 2}, &LlamaLayer{}, cfg); got != nil {
		t.Fatalf("zero intermediate output=%v, want nil", got)
	}
}

type switchMLXRawSource map[string]struct {
	raw   []byte
	dtype string
	shape []int
}

func (s switchMLXRawSource) GetRaw(name string) ([]byte, string, []int, error) {
	v, ok := s[name]
	if !ok {
		return nil, "", nil, errSwitchMLXMissing(name)
	}
	return v.raw, v.dtype, v.shape, nil
}

type errSwitchMLXMissing string

func (e errSwitchMLXMissing) Error() string { return "missing " + string(e) }

func TestMoeForwardRejectsMalformedMLXRouter(t *testing.T) {
	cfg := LlamaConfig{NumExperts: 2, NumExpertsPerTok: 1, MoEIntermediate: 2}
	layer := &LlamaLayer{RouterW: &mlx.QuantWeight{OutDim: 2, InDim: 8, Bits: 4, GroupSize: 8, Groups: 1, Weight: []uint32{0}, Scales: []float32{1, 1}, Biases: []float32{0, 0}}}
	if got := moeForward(make([]float32, 8), layer, cfg); got != nil {
		t.Fatalf("malformed router output=%v, want nil", got)
	}
}

func TestLoadSwitchMLXExpertsSupportsF32ScalesAndBiases(t *testing.T) {
	const (
		base       = "model.layers.0.block_sparse_moe.switch_mlp.gate_proj"
		experts    = 2
		outDim     = 1
		inDim      = 8
		groupSize  = 4
		bits       = 4
		packedCols = 1
		groups     = 2
	)
	weightRaw := make([]byte, experts*outDim*packedCols*4)
	binary.LittleEndian.PutUint32(weightRaw[0:], 0x76543210)
	binary.LittleEndian.PutUint32(weightRaw[4:], 0xfedcba98)
	scalesRaw := f32Raw([]float32{1.25, 2.5, 3.75, 4.5})
	biasesRaw := f32Raw([]float32{-1, -2, -3, -4})

	src := switchMLXRawSource{
		base + ".weight": {raw: weightRaw, dtype: "U32", shape: []int{experts, outDim, packedCols}},
		base + ".scales": {raw: scalesRaw, dtype: "F32", shape: []int{experts, outDim, groups}},
		base + ".biases": {raw: biasesRaw, dtype: "F32", shape: []int{experts, outDim, groups}},
	}
	expertWeights, err := LoadSwitchMLXExperts(src, base, experts, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("LoadSwitchMLXExperts returned error: %v", err)
	}
	if len(expertWeights) != experts {
		t.Fatalf("len=%d, want %d", len(expertWeights), experts)
	}
	if got := expertWeights[1].Scales; len(got) != groups || got[0] != 3.75 || got[1] != 4.5 {
		t.Fatalf("expert 1 scales=%v", got)
	}
	if got := expertWeights[0].Biases; len(got) != groups || got[0] != -1 || got[1] != -2 {
		t.Fatalf("expert 0 biases=%v", got)
	}
}

func TestLoadSwitchMLXExpertsRejectsUnsupportedDtypes(t *testing.T) {
	const base = "moe"
	src := switchMLXRawSource{
		base + ".weight": {raw: make([]byte, 4), dtype: "F16", shape: []int{1, 1, 1}},
		base + ".scales": {raw: make([]byte, 2), dtype: "BF16", shape: []int{1, 1, 1}},
		base + ".biases": {raw: make([]byte, 2), dtype: "BF16", shape: []int{1, 1, 1}},
	}
	if _, err := LoadSwitchMLXExperts(src, base, 1, 1, 8, 8, 4); err == nil {
		t.Fatalf("LoadSwitchMLXExperts accepted F16 weight dtype")
	}

	src[base+".weight"] = struct {
		raw   []byte
		dtype string
		shape []int
	}{raw: make([]byte, 4), dtype: "U32", shape: []int{1, 1, 1}}
	src[base+".scales"] = struct {
		raw   []byte
		dtype string
		shape []int
	}{raw: make([]byte, 2), dtype: "I8", shape: []int{1, 1, 1}}
	if _, err := LoadSwitchMLXExperts(src, base, 1, 1, 8, 8, 4); err == nil {
		t.Fatalf("LoadSwitchMLXExperts accepted I8 scale dtype")
	}
}

func f32Raw(vals []float32) []byte {
	raw := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return raw
}
