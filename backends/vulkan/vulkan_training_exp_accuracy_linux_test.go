//go:build linux

package vulkan

import (
	"context"
	"math"
	"os"
	"testing"
	"time"
)

func TestVulkanNativeTrainingExpAccuracy(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_EXP_ACCURACY_NATIVE") != "1" {
		t.Skip("explicit exp range qualification")
	}
	code, e := os.ReadFile(os.Getenv("GO_PHERENCE_VULKAN_EXP_ACCURACY_SPIRV"))
	if e != nil {
		t.Fatal(e)
	}
	captureInit(t)
	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	if e = VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 8 << 20, MaxAllocations: 64}); e != nil {
		t.Fatal(e)
	}
	k, e := VkKernelCreate(code, 5, 24)
	if e != nil {
		t.Fatal(e)
	}
	const n = 65537
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(-30 * float64(i) / float64(n-1))
	}
	a, e := NewVkTensorArena(ctx, 8*n+1024)
	if e != nil {
		t.Fatal(e)
	}
	in, e := a.AllocF32(ctx, n)
	if e != nil {
		t.Fatal(e)
	}
	out, e := a.AllocF32(ctx, n)
	if e != nil {
		t.Fatal(e)
	}
	z, e := a.AllocF32(ctx, 1)
	if e != nil {
		t.Fatal(e)
	}
	if e = in.Upload(ctx, x); e != nil {
		t.Fatal(e)
	}
	s := VkF32Stage{Kernel: k, Tensors: []*VkTensorF32{z, z, z, in, out}, PushWords: []uint32{n, 1, 1, 0, 0, 0}, Groups: [3]uint32{(n + 255) / 256, 1, 1}}
	p, e := NewVkF32Plan(ctx, []VkF32Stage{s})
	if e != nil {
		t.Fatal(e)
	}
	if e = p.Run(ctx); e != nil {
		t.Fatal(e)
	}
	got := make([]float32, n)
	if e = out.Download(ctx, got); e != nil {
		t.Fatal(e)
	}
	if e = p.Close(); e != nil {
		t.Fatal(e)
	}
	var max float64
	changed := 0
	for i, g := range got {
		want := float32(math.Exp(float64(x[i])))
		d := math.Abs(float64(g) - float64(want))
		max = math.Max(max, d)
		if d != 0 {
			changed++
			if changed < 12 {
				t.Logf("exp x=%g got=%g want=%g delta=%g", x[i], g, want, d)
			}
		}
		if d > 6e-8 {
			t.Fatal("exp gate", x[i], d)
		}
	}
	if os.Getenv("GO_PHERENCE_VULKAN_EXP_INTERMEDIATES") == "1" {
		for mode := uint32(1); mode < 3; mode++ {
			s.PushWords[4] = mode
			p, e := NewVkF32Plan(ctx, []VkF32Stage{s})
			if e != nil {
				t.Fatal(e)
			}
			if e = p.Run(ctx); e != nil {
				t.Fatal(e)
			}
			v := make([]float32, n)
			if e = out.Download(ctx, v); e != nil {
				t.Fatal(e)
			}
			if e = p.Close(); e != nil {
				t.Fatal(e)
			}
			for i, g := range got {
				if g != float32(math.Exp(float64(x[i]))) {
					t.Logf("intermediate mode%d x=%g value=%.17g", mode, x[i], v[i])
				}
			}
		}
	}
	if e = a.Close(); e != nil {
		t.Fatal(e)
	}
	if e = k.Close(); e != nil {
		t.Fatal(e)
	}
	t.Logf("exp values=%d changed=%d max=%g", n, changed, max)
}
