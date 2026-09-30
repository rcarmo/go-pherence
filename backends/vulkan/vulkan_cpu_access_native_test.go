package vulkan

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Opt-in gate only: ordinary host tests must not initialise/probe a GPU.
// Independent analytic X*I+B, requiring coherent cached native storage.
func TestVulkanNativeCPUAccessLinear(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_CPU_ACCESS") != "1" {
		t.Skip("set GO_PHERENCE_TEST_VULKAN_CPU_ACCESS=1 with separately authorised device access")
	}
	device := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if device == "" {
		t.Fatal("require GO_PHERENCE_VULKAN_DEVICE to identify the authorised device")
	}
	if !VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	if !strings.Contains(strings.ToLower(VulkanDeviceName()), strings.ToLower(device)) {
		t.Fatalf("unexpected device %q", VulkanDeviceName())
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	op, err := NewVkLinearF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()
	arena, err := NewVkSharedTensorArena(ctx, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer arena.Close()
	alloc := func(shape ...int) *VkTensorF32 {
		v, err := arena.AllocF32(ctx, shape...)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	x, w, b, y, guard := alloc(4, 8), alloc(8, 8), alloc(8), alloc(4, 8), alloc(2)
	weight := make([]float32, 64)
	bias := make([]float32, 8)
	for i := 0; i < 8; i++ {
		weight[i*8+i] = 1
		bias[i] = float32(i) * 0.125
	}
	if err := w.Upload(ctx, weight); err != nil {
		t.Fatal(err)
	}
	if err := b.Upload(ctx, bias); err != nil {
		t.Fatal(err)
	}
	if err := guard.Upload(ctx, []float32{71, 83}); err != nil {
		t.Fatal(err)
	}
	copied := make([]float32, 32)
	for run := 0; run < 3; run++ {
		if err := x.WithCPUWrite(ctx, func(v []float32) error {
			for i := range v {
				v[i] = float32(i-run) * 0.03125
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		if err := op.Forward(ctx, y, x, w, b); err != nil {
			t.Fatal(err)
		}
		if err := y.WithCPURead(ctx, func(v []float32) error {
			for i, value := range v {
				want := float32(i-run)*0.03125 + bias[i%8]
				if math.Abs(float64(value-want)) > 2e-5 {
					t.Fatalf("analytic result[%d] %g/%g", i, value, want)
				}
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		if err := y.Download(ctx, copied); err != nil {
			t.Fatal(err)
		}
		if err := y.WithCPURead(ctx, func(v []float32) error {
			for i, value := range v {
				if value != copied[i] {
					t.Fatal("copied/mapped read differs")
				}
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := guard.WithCPURead(ctx, func(v []float32) error {
		if v[0] != 71 || v[1] != 83 {
			t.Fatal("guard overwritten")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
