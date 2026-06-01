package model

import (
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGGUFParseConfigQwenNextHybridMetadata(t *testing.T) {
	g := &gguf.GGUF{Meta: map[string]any{
		"general.architecture":              "qwen35moe",
		"qwen35moe.embedding_length":        uint32(2048),
		"qwen35moe.block_count":             uint32(40),
		"qwen35moe.attention.head_count":    uint32(16),
		"qwen35moe.attention.head_count_kv": uint32(2),
		"qwen35moe.feed_forward_length":     uint32(512),
		"qwen35moe.context_length":          uint32(262144),
		"qwen35moe.attention.key_length":    uint32(256),
		"qwen35moe.attention.value_length":  uint32(256),
		"qwen35moe.full_attention_interval": uint32(4),
		"qwen35moe.ssm.conv_kernel":         uint32(4),
		"qwen35moe.ssm.group_count":         uint32(16),
		"qwen35moe.ssm.inner_size":          uint32(4096),
		"qwen35moe.ssm.state_size":          uint32(128),
		"qwen35moe.ssm.time_step_rank":      uint32(32),
	}}
	cfg, err := ggufParseConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.IsQwenNextHybridGGUF() || cfg.SSMInnerSize != 4096 || cfg.SSMConvKernel != 4 || cfg.AttentionKeyLength != 256 || cfg.FullAttentionInterval != 4 {
		t.Fatalf("unexpected hybrid cfg: %+v", cfg)
	}
	if err := cfg.ValidateQwenNextHybridMetadata(); err != nil {
		t.Fatal(err)
	}
}

func TestGGUFQwenNextHybridMetadataRequiresSSM(t *testing.T) {
	cfg := GGUFLlamaConfig{HiddenSize: 2048, AttentionKeyLength: 256, AttentionValueLength: 256, FullAttentionInterval: 4}
	if err := cfg.ValidateQwenNextHybridMetadata(); err == nil {
		t.Fatal("expected incomplete SSM metadata error")
	}
}

func TestGGUFQwenNextLinearShapesMatchLocalREAP(t *testing.T) {
	cfg := GGUFLlamaConfig{HiddenSize: 2048, SSMInnerSize: 4096, SSMStateSize: 128, SSMConvKernel: 4, SSMTimeStepRank: 32, SSMGroupCount: 16}
	shapes, err := cfg.QwenNextLinearShapes()
	if err != nil {
		t.Fatal(err)
	}
	if shapes.KeyDim != 2048 || shapes.ValueDim != 4096 || shapes.ConvDim != 8192 || shapes.QKV[0] != 2048 || shapes.QKV[1] != 8192 {
		t.Fatalf("unexpected shapes: %+v", shapes)
	}
}

func TestGGUFQwenNextStateSizes(t *testing.T) {
	cfg := GGUFLlamaConfig{HiddenSize: 2048, SSMInnerSize: 4096, SSMStateSize: 128, SSMConvKernel: 4, SSMTimeStepRank: 32, SSMGroupCount: 16}
	st, err := cfg.NewQwenNextState()
	if err != nil {
		t.Fatal(err)
	}
	if len(st.Conv) != 8192*4 || len(st.SSM) != 32*4096*128 {
		t.Fatalf("state lens conv=%d ssm=%d", len(st.Conv), len(st.SSM))
	}
}

func TestSplitGGUFQwenNextFusedQKV(t *testing.T) {
	shapes := loaderconfig.Qwen35LinearAttentionShapes{KeyDim: 2, ValueDim: 3, ConvDim: 7}
	q, k, v, err := splitGGUFQwenNextFusedQKV([]float32{1, 2, 3, 4, 5, 6, 7}, shapes)
	if err != nil {
		t.Fatal(err)
	}
	if len(q) != 2 || q[0] != 1 || q[1] != 2 || len(k) != 2 || k[0] != 3 || k[1] != 4 || len(v) != 3 || v[0] != 5 || v[2] != 7 {
		t.Fatalf("q=%v k=%v v=%v", q, k, v)
	}
}

func TestGGUFQwenNextConvStateAndDepthwise(t *testing.T) {
	state := make([]float32, 4) // convDim=2, kernel=2
	if err := updateGGUFQwenNextConvStateInPlace(state, []float32{1}, []float32{2}, nil, 2); err != nil {
		t.Fatal(err)
	}
	if err := updateGGUFQwenNextConvStateInPlace(state, []float32{3}, []float32{4}, nil, 2); err != nil {
		t.Fatal(err)
	}
	if got := state; got[0] != 1 || got[1] != 2 || got[2] != 3 || got[3] != 4 {
		t.Fatalf("state=%v", got)
	}
	out, err := applyGGUFQwenNextDepthwiseConv(state, []float32{10, 20, 100, 200}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if out[0] != 310 || out[1] != 840 {
		t.Fatalf("out=%v", out)
	}
}

func TestGGUFQwenNextDeltaUpdate(t *testing.T) {
	cfg := GGUFLlamaConfig{HiddenSize: 2, SSMInnerSize: 2, SSMStateSize: 2, SSMConvKernel: 1, SSMTimeStepRank: 1, SSMGroupCount: 1}
	ssm := make([]float32, 4)
	out, err := applyGGUFQwenNextDeltaUpdateInPlace(ssm, []float32{1, 0}, []float32{1, 0}, []float32{2, 4}, []float32{1}, []float32{1}, []float32{0}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0] == 0 || out[1] == 0 || ssm[0] == 0 || ssm[2] == 0 {
		t.Fatalf("out=%v ssm=%v", out, ssm)
	}
}

func TestGGUFQwenNextGatedRMSNorm(t *testing.T) {
	x := []float32{3, 4}
	if err := applyGGUFQwenNextGatedRMSNormValueHeads(x, []float32{1, 1}, []float32{1, 1}, 1, 2, 0); err != nil {
		t.Fatal(err)
	}
	if x[0] == 3 || x[1] == 4 {
		t.Fatalf("not normalized: %v", x)
	}
}
