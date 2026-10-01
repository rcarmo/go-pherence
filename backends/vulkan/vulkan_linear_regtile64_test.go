package vulkan

import (
	"context"
	"math"
	"os"
	"testing"
	"unsafe"
)

func TestVulkanOfflineLinearRegTile64Admission(t *testing.T) {
	newLifetimeMock(t)
	contract, err := InspectVulkanShader(spirv_linear_f32_regtile64)
	want := VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 16384, StorageBindings: 15, PushBytes: 12}
	if err != nil || contract != want {
		t.Fatal(contract, err)
	}
	vkLimits = offlineLimits()
	vkLimits.SharedMemoryBytes = 16383
	if _, err := NewVkLinearRegTile64F32(context.Background()); err == nil {
		t.Fatal("shared bound")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewVkLinearRegTile64F32(ctx); err != context.Canceled {
		t.Fatal("cancelled constructor", err)
	}
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 4
	kernel.pushSize = 12
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {
		t.Fatal("stage must not dispatch")
	})
	op := &VkLinearF32{kernel: kernel, outputTile: 64}
	vkLimits.StorageBufferRange = math.MaxUint32
	for _, dims := range [][3]int{{1, 1, 1}, {63, 17, 65}, {64, 64, 64}, {65, 31, 127}, {1500, 1280, 5120}} {
		m, k, n := dims[0], dims[1], dims[2]
		stage, err := op.Stage(context.Background(), linearMetadataTensor(1, m, n), linearMetadataTensor(2, m, k), linearMetadataTensor(3, n, k), linearMetadataTensor(4, n))
		if err != nil || stage.Groups != ([3]uint32{uint32((n + 63) / 64), uint32((m + 63) / 64), 1}) {
			t.Fatal(dims, stage, err)
		}
	}
}

func TestVulkanNativeLinearRegTile64(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_LINEAR_REGTILE64") != "1" {
		t.Skip("explicit candidate window")
	}
	t.Setenv("GO_PHERENCE_TEST_VULKAN_LINEAR_REGTILE", "1")
	testVulkanNativeLinearRegTile(t, NewVkLinearRegTileF32, NewVkLinearRegTile64F32)
}
