package ptx

import (
	"strings"
	"testing"
)

func assertPTXContains(t *testing.T, name, src string, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(src, want) {
			t.Fatalf("%s missing %q", name, want)
		}
	}
}

func TestConv1DPTXContracts(t *testing.T) {
	assertPTXContains(t, "Conv1DK3S1PTX", Conv1DK3S1PTX,
		".visible .entry conv1d_k3_s1",
		".param .u64 out_ptr",
		".param .u64 in_ptr",
		".param .u64 wt_ptr",
		".param .u64 bias_ptr",
		".param .u32 in_channels",
		".param .u32 out_channels",
		".param .u32 in_length",
	)
	assertPTXContains(t, "Conv1DK3S2PTX", Conv1DK3S2PTX,
		".visible .entry conv1d_k3_s2",
		".param .u64 out_ptr",
		".param .u64 in_ptr",
		".param .u64 wt_ptr",
		".param .u64 bias_ptr",
		"fma.rn.f32",
		"st.global.f32",
	)
	if strings.Contains(Conv1DK3S2PTX, "TODO") {
		t.Fatalf("Conv1DK3S2PTX still advertises TODO-only body")
	}
}

func TestAttentionPTXContracts(t *testing.T) {
	assertPTXContains(t, "AttentionFullPTX", AttentionFullPTX,
		".visible .entry attention_full",
		".param .u64 out_ptr",
		".param .u64 q_ptr",
		".param .u64 k_ptr",
		".param .u64 v_ptr",
		".param .u32 seq_q",
		".param .u32 seq_kv",
		".param .u32 head_dim",
		"fma.rn.f32",
		"ex2.approx.ftz.f32",
		"st.global.f32",
	)
	assertPTXContains(t, "CrossAttentionPTX", CrossAttentionPTX,
		".visible .entry cross_attention",
		".param .u64 out_ptr",
		".param .u64 q_ptr",
		".param .u64 k_ptr",
		".param .u64 v_ptr",
		"fma.rn.f32",
		"ex2.approx.ftz.f32",
		"st.global.f32",
	)
	if strings.Contains(AttentionFullPTX, "TODO") || strings.Contains(CrossAttentionPTX, "TODO") {
		t.Fatalf("attention PTX still advertises TODO-only body")
	}
}

func TestAttentivePoolPTXContract(t *testing.T) {
	assertPTXContains(t, "AttentivePoolPTX", AttentivePoolPTX,
		".visible .entry attentive_stat_pool",
		".param .u64 out_ptr",
		".param .u64 h_ptr",
		".param .u64 attn_w_ptr",
		".param .u64 attn_b_ptr",
		".param .u32 channels",
		".param .u32 length",
	)
}
