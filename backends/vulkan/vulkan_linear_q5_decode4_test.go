package vulkan

import (
	"context"
	"errors"
	"testing"
)

func TestVulkanOfflineQ5Decode4Admission(t *testing.T) {
	c, e := InspectVulkanShader(spirv_linear_q5_decode4_f32)
	if e != nil {
		t.Fatal(e)
	}
	want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 16384, StorageBindings: 15, PushBytes: 12}
	if c != want {
		t.Fatal(c, want)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if o, e := NewVkLinearQ5Decode4SetStream(ctx, []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) { t.Fatal("cancelled reader"); return nil, nil }); o != nil || !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if o, e := NewVkLinearQ5Decode4SetStream(context.Background(), []VkLinearQ5Shape{{1, 32}}, nil); o != nil || e == nil {
		t.Fatal("nil reader")
	}
	offlineVK(t)
	mockVK(t, &vkLimits, offlineLimits())
	vkLimits.SharedMemoryBytes = 16383
	if _, e := NewVkLinearQ5Decode4SetStream(context.Background(), []VkLinearQ5Shape{{1, 32}}, func(context.Context, int) ([]byte, error) { t.Fatal("before device limits"); return nil, nil }); !errors.Is(e, ErrVulkanLimit) {
		t.Fatal(e)
	}
}
