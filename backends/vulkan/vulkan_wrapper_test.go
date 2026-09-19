package vulkan

import (
	"math"
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
		{"vec add f32", func() error { return VkVecAddF32(buf, buf, buf, 2) }, "invalid"},
		{"vec add bf16 odd", func() error { return VkVecAddBF16(&VkBuf{size: 4}, &VkBuf{size: 4}, &VkBuf{size: 4}, 3) }, "invalid"},
		{"vec add bf16 short", func() error { return VkVecAddBF16(buf, buf, buf, 4) }, "invalid"},
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

// TestVulkanWrapperStubsValidBuffers checks behavior with valid-sized buffers.
//
// Two cases depending on the host:
//   - Vulkan NOT ready: operations fail closed before dispatch
//   - Vulkan ready: allocate real device buffers → ops should succeed (no error)
func TestVulkanWrapperStubsValidBuffers(t *testing.T) {
	const n = 2
	const floatsPerBuf = 4096 // well over any n*sizeof(float32) needed

	if !VulkanReady() {
		// Offline / CI path: use bare VkBuf and require a fail-closed state.
		// Whether initialization or optional cache construction is reached first is
		// intentionally not part of this wrapper contract.
		bare := &VkBuf{size: floatsPerBuf * 4}
		cases := []struct {
			name string
			fn   func() error
		}{
			{"vec add f32", func() error { return VkVecAddF32(bare, bare, bare, n) }},
			{"vec add bf16", func() error { return VkVecAddBF16(bare, bare, bare, n) }},
			{"rmsnorm", func() error { return VkRMSNormF32(bare, bare, n, 1e-6) }},
			{"rmsnorm no scale", func() error { return VkRMSNormNoScaleF32(bare, n, 1e-6) }},
			{"gemv", func() error { return VkGemvF32(bare, bare, bare, n, n) }},
			{"silu", func() error { return VkSiLUMulF32(bare, bare, bare, n) }},
			{"gelu", func() error { return VkGELUTanhMulF32(bare, bare, n) }},
			{"rope", func() error { return VkRoPEPartialF32(bare, &VkBuf{size: floatsPerBuf * 4}, 0, 1, 4, 2) }},
			{"attention", func() error { return VkAttentionScoresF32(bare, bare, bare, n, n, 1, n, 1) }},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				err := tc.fn()
				if err == nil || !strings.Contains(err.Error(), "not initialized") && !strings.Contains(err.Error(), "not available") {
					t.Fatalf("err=%v, want unavailable Vulkan state", err)
				}
			})
		}
		return
	}

	// Live Vulkan path: allocate real buffers and verify ops complete without error.
	alloc := func(t *testing.T, n int) *VkBuf {
		t.Helper()
		b, err := VkBufAlloc(n * 4)
		if err != nil {
			t.Fatalf("VkBufAlloc(%d): %v", n, err)
		}
		t.Cleanup(func() { b.Free() })
		return b
	}

	t.Run("vec add f32", func(t *testing.T) {
		dst, a, b := alloc(t, floatsPerBuf), alloc(t, floatsPerBuf), alloc(t, floatsPerBuf)
		if err := VkVecAddF32(dst, a, b, n); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("rmsnorm", func(t *testing.T) {
		x, w := alloc(t, floatsPerBuf), alloc(t, floatsPerBuf)
		if err := VkRMSNormF32(x, w, n, 1e-6); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("rmsnorm no scale", func(t *testing.T) {
		x := alloc(t, floatsPerBuf)
		if err := VkRMSNormNoScaleF32(x, n, 1e-6); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("gemv", func(t *testing.T) {
		out, x, w := alloc(t, floatsPerBuf), alloc(t, floatsPerBuf), alloc(t, floatsPerBuf*floatsPerBuf)
		if err := VkGemvF32(out, x, w, n, n); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("silu", func(t *testing.T) {
		dst, gate, up := alloc(t, floatsPerBuf), alloc(t, floatsPerBuf), alloc(t, floatsPerBuf)
		if err := VkSiLUMulF32(dst, gate, up, n); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("gelu", func(t *testing.T) {
		gate, up := alloc(t, floatsPerBuf), alloc(t, floatsPerBuf)
		if err := VkGELUTanhMulF32(gate, up, n); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("rope", func(t *testing.T) {
		x, freqs := alloc(t, floatsPerBuf), alloc(t, floatsPerBuf)
		if err := VkRoPEPartialF32(x, freqs, 0, 1, 4, 2); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("attention", func(t *testing.T) {
		out, q, k := alloc(t, floatsPerBuf), alloc(t, floatsPerBuf), alloc(t, floatsPerBuf)
		if err := VkAttentionScoresF32(out, q, k, n, n, 1, n, 1); err != nil {
			t.Fatal(err)
		}
		for _, bad := range []struct {
			heads, kvHeads int
			scale          float32
		}{{4, 3, 1}, {2, 3, 1}, {2, 1, float32(math.NaN())}, {2, 1, float32(math.Inf(1))}} {
			if err := VkAttentionScoresF32(out, q, k, 1, bad.heads, bad.kvHeads, 1, bad.scale); err == nil {
				t.Fatal("invalid attention geometry accepted", bad)
			}
		}
	})
}

// Geometry is validated before backend admission, so these synthetic buffers
// never reach a driver even on an online host. Shader indices are uint32.
func TestVulkanWrappersRejectShaderIndexOverflow(t *testing.T) {
	if ^uint(0)>>32 == 0 {
		t.Skip("requires 64-bit int")
	}
	huge := uint64(1) << 32
	n := int(huge)
	b := &VkBuf{size: ^uint64(0)}
	for name, fn := range map[string]func() error{
		"vec":               func() error { return VkVecAddF32(b, b, b, n) },
		"bf16":              func() error { return VkVecAddBF16(b, b, b, n) },
		"rms":               func() error { return VkRMSNormF32(b, b, n, 1e-6) },
		"gemv_product":      func() error { return VkGemvF32(b, b, b, 65536, 65536) },
		"silu":              func() error { return VkSiLUMulF32(b, b, b, n) },
		"gelu":              func() error { return VkGELUTanhMulF32(b, b, n) },
		"attention_product": func() error { return VkAttentionScoresF32(b, b, b, 65536, 65536, 1, 1, 1) },
	} {
		t.Run(name, func(t *testing.T) {
			if err := fn(); err == nil || !strings.Contains(err.Error(), "invalid") {
				t.Fatalf("err=%v, want invalid geometry", err)
			}
		})
	}
}
