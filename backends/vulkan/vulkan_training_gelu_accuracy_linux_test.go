//go:build linux

package vulkan

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"testing"
	"time"
)

func TestVulkanNativeTrainingGELUAccuracy(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_GELU_ACCURACY_NATIVE") != "1" {
		t.Skip("explicit native activation accuracy")
	}
	code := spirvTrainingGELU
	if p := os.Getenv("GO_PHERENCE_VULKAN_GELU_ACCURACY_SPIRV"); p != "" {
		var e error
		code, e = os.ReadFile(p)
		if e != nil {
			t.Fatal(e)
		}
		s := sha256.Sum256(code)
		if hex.EncodeToString(s[:]) != os.Getenv("GO_PHERENCE_VULKAN_GELU_ACCURACY_SHA256") {
			t.Fatal("activation asset pin")
		}
	}
	c, e := InspectVulkanShader(code)
	if e != nil || c.StorageBindings != 7 || c.PushBytes != 8 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, e)
	}
	captureInit(t)
	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	if e = VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 8 << 20, MaxAllocations: 64}); e != nil {
		t.Fatal(e)
	}
	kernel, e := VkKernelCreate(code, 3, 8)
	if e != nil {
		t.Fatal(e)
	}
	op := &VkTrainingF32{gelu: kernel}
	const n = 65537
	x, seed := make([]float32, n), make([]float32, n)
	for i := range x {
		x[i] = float32(-8 + 16*float64(i)/float64(n-1))
		seed[i] = 1
	}
	a, e := NewVkTensorArena(ctx, 4*3*n+512)
	if e != nil {
		t.Fatal(e)
	}
	tensor := func(v []float32) *VkTensorF32 {
		z, e := a.AllocF32(ctx, len(v))
		if e != nil {
			t.Fatal(e)
		}
		if e = z.Upload(ctx, v); e != nil {
			t.Fatal(e)
		}
		return z
	}
	xv, gv, out := tensor(x), tensor(seed), tensor(make([]float32, n))
	var maxForward, maxBackward float64
	for mode := uint32(0); mode < 2; mode++ {
		stage, e := op.GELUStage(ctx, out, xv, gv, mode)
		if e != nil {
			t.Fatal(e)
		}
		p, e := NewVkF32Plan(ctx, []VkF32Stage{stage})
		if e != nil {
			t.Fatal(e)
		}
		if e = p.Run(ctx); e != nil {
			t.Fatal("native failure no retry", e)
		}
		got := make([]float32, n)
		if e = out.Download(ctx, got); e != nil {
			t.Fatal(e)
		}
		if e = p.Close(); e != nil {
			t.Fatal(e)
		}
		for i, g := range got {
			v := float64(x[i])
			k := math.Sqrt(2 / math.Pi)
			th := math.Tanh(k * (v + .044715*v*v*v))
			want := float32(.5 * v * (1 + th))
			gate := float64(1e-6)
			if mode == 1 {
				want = float32(.5*(1+th) + .5*v*(1-th*th)*k*(1+3*.044715*v*v))
				gate = 2e-7
			}
			d := math.Abs(float64(g) - float64(want))
			if mode == 0 {
				maxForward = math.Max(maxForward, d)
			} else {
				maxBackward = math.Max(maxBackward, d)
			}
			if math.IsNaN(float64(g)) || d > gate {
				t.Fatal("activation independent gate", mode, x[i], g, want, d)
			}
		}
	}
	if e = a.Close(); e != nil {
		t.Fatal(e)
	}
	if e = kernel.Close(); e != nil {
		t.Fatal(e)
	}
	u := VulkanMemoryStats()
	if u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
		t.Fatal("closure", u)
	}
	t.Logf("GELU %d values forwardMax=%g derivativeMax=%g", n, maxForward, maxBackward)
}
