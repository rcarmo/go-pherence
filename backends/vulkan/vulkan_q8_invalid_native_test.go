package vulkan

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestVulkanNativeQ8InvalidBlocks(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_INTDOT_INVALID") != "1" {
		t.Skip("explicit invalid quantiser blocks")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	if !VulkanInitIntegerDot() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical device")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	bad := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 1001, -1001, 1e-35}
	x := make([]float32, len(bad)*32)
	for i, v := range bad {
		x[i*32] = v
	}
	a := nativeArena(t)
	guard := nativeGuard(t, a)
	tx := nativeTensor(t, a, x, len(x))
	q := nativeTensor(t, a, nil, len(bad)*9)
	right := nativeGuard(t, a)
	k, e := VkKernelCreateIntegerDot(spirv_quant_q8_integer_dot, 2, 4)
	if e != nil {
		t.Fatal(e)
	}
	nativeClose(t, k)
	p, e := NewVkF32Plan(context.Background(), []VkF32Stage{{Kernel: k, Groups: [3]uint32{1, 1, 1}, Tensors: []*VkTensorF32{tx, q}, PushWords: []uint32{uint32(len(bad))}}})
	if e != nil {
		t.Fatal(e)
	}
	nativeClose(t, p)
	nativeRun(t, p.Run)
	got := nativeDownload(t, q)
	for b := range bad {
		if math.Float32bits(got[b*9]) != 0x7e007e00 {
			t.Fatal("invalid marker", b)
		}
		for j := 1; j < 9; j++ {
			if math.Float32bits(got[b*9+j]) != 0 {
				t.Fatal("invalid data", b, j)
			}
		}
	}
	guard()
	right()
	t.Log("invalid NaN/Inf/extreme/underflow rejected via NaN marker")
}
