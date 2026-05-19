package vulkan

import (
	"strings"
	"testing"
)

func TestVulkanWrapperStubsRejectInvalidInputs(t *testing.T) {
	buf := &VkBuf{size: 4}
	cases := []struct {
		name string
		fn   func() error
		want string
	}{
		{"rmsnorm", func() error { return VkRMSNormF32(buf, buf, 2, 1e-6) }, "invalid"},
		{"rmsnorm no scale", func() error { return VkRMSNormNoScaleF32(buf, 2, 1e-6) }, "invalid"},
		{"gemv", func() error { return VkGemvF32(buf, buf, buf, 2, 2) }, "invalid"},
		{"silu", func() error { return VkSiLUMulF32(buf, buf, buf, 2) }, "invalid"},
		{"gelu", func() error { return VkGELUTanhMulF32(buf, buf, 2) }, "invalid"},
		{"rope", func() error { return VkRoPEPartialF32(buf, buf, 0, 1, 4, 2) }, "invalid"},
		{"attention", func() error { return VkAttentionScoresF32(buf, buf, buf, 2, 2, 1, 2, 1) }, "invalid"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.fn()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestVulkanWrapperStubsReportPendingWhenBuffersAreValid(t *testing.T) {
	buf := &VkBuf{size: 4096}
	cases := []struct {
		name string
		fn   func() error
	}{
		{"rmsnorm", func() error { return VkRMSNormF32(buf, buf, 2, 1e-6) }},
		{"rmsnorm no scale", func() error { return VkRMSNormNoScaleF32(buf, 2, 1e-6) }},
		{"gemv", func() error { return VkGemvF32(buf, buf, buf, 2, 2) }},
		{"silu", func() error { return VkSiLUMulF32(buf, buf, buf, 2) }},
		{"gelu", func() error { return VkGELUTanhMulF32(buf, buf, 2) }},
		{"rope", func() error { return VkRoPEPartialF32(buf, buf, 0, 1, 4, 2) }},
		{"attention", func() error { return VkAttentionScoresF32(buf, buf, buf, 2, 2, 1, 2, 1) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.fn()
			if err == nil || !strings.Contains(err.Error(), "not available") {
				t.Fatalf("err=%v, want pending/not available", err)
			}
		})
	}
}
