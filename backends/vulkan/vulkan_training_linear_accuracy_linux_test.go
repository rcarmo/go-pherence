//go:build linux

package vulkan

import (
	"context"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"os"
	"testing"
	"time"
)

// Uses independent CPU SIMD and F64 dot products, including cancellation and
// awkward tails. F64 bound is relative to sum(abs(products)), not output size.
func TestVulkanNativeTrainingLinearAccuracy(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_LINEAR_ACCURACY_NATIVE") != "1" {
		t.Skip("explicit native linear accuracy qualification")
	}
	captureInit(t)
	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	if e := VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 8 << 20, MaxAllocations: 64}); e != nil {
		t.Fatal(e)
	}
	op, e := NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	var maxCPU, maxRelative float64
	for _, shape := range [][3]int{{1, 1, 1}, {7, 5, 1}, {8, 4, 1}, {15, 4, 1}, {16, 8, 1}, {17, 8, 1}, {257, 12, 1}, {1024, 256, 1}, {4096, 32, 1}, {64, 64, 16}, {129, 64, 17}, {1024, 256, 29}, {64, 64, 577}, {129, 65, 950}, {1024, 64, 950}} {
		k, n, rows := shape[0], shape[1], shape[2]
		x, w, b := make([]float32, rows*k), make([]float32, k*n), make([]float32, n)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*.37)) * .3
		}
		for i := range w {
			w[i] = float32(math.Cos(float64(i)*.23)) * .04
		}
		for i := range b {
			b[i] = float32(i%5-2) * .01
		}
		a, e := NewVkTensorArena(ctx, 4*(2*rows*k+k*n+n+rows*n)+512)
		if e != nil {
			t.Fatal(e)
		}
		tensor := func(v []float32, shape ...int) *VkTensorF32 {
			t.Helper()
			z, e := a.AllocF32(ctx, shape...)
			if e != nil {
				t.Fatal(e)
			}
			if e = z.Upload(ctx, v); e != nil {
				t.Fatal(e)
			}
			return z
		}
		xv, xt, wv, bv, out := tensor(x, rows, k), tensor(x, rows, k), tensor(w, n, k), tensor(b, n), tensor(make([]float32, rows*n), rows, n)
		cpu := make([]float32, rows*n)
		if rows >= 16 {
			if !simd.DenseNTTo(cpu, x, w, rows, n, k, 1, k, k, n) {
				t.Fatal("CPU dense shape")
			}
		} else if !simd.GemvRows(cpu, x, w, n, k) {
			t.Fatal("CPU reference shape")
		}
		for mode := uint32(0); mode < 2; mode++ {
			s, e := op.LinearDualForwardStage(ctx, out, xv, xt, wv, bv, mode)
			if e != nil {
				t.Fatal(e)
			}
			p, e := NewVkF32Plan(ctx, []VkF32Stage{s})
			if e != nil {
				t.Fatal(e)
			}
			if e = p.Run(ctx); e != nil {
				t.Fatal("native failure no retry", e)
			}
			got := make([]float32, rows*n)
			if e = out.Download(ctx, got); e != nil {
				t.Fatal(e)
			}
			if e = p.Close(); e != nil {
				t.Fatal(e)
			}
			for j, g := range got {
				want := cpu[j]
				var ref, absSum float64
				row, col := j/n, j%n
				for i, v := range x[row*k : (row+1)*k] {
					prod := float64(v) * float64(w[col*k+i])
					ref += prod
					absSum += math.Abs(prod)
				}
				if mode == 0 {
					want += b[col]
					ref += float64(b[col])
					absSum += math.Abs(float64(b[col]))
				}
				d := math.Abs(float64(g) - float64(want))
				maxCPU = math.Max(maxCPU, d)
				rel := math.Abs(float64(g)-ref) / math.Max(absSum, 1e-30)
				maxRelative = math.Max(maxRelative, rel)
				if (math.IsNaN(float64(g)) || math.IsInf(float64(g), 0)) || d > 2e-6 || rel > 2e-6 {
					t.Fatal("linear accuracy", shape, mode, j, d, rel)
				}
			}
		}
		if e = a.Close(); e != nil {
			t.Fatal(e)
		}
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	u := VulkanMemoryStats()
	if u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
		t.Fatal("closure", u)
	}
	t.Logf("linear CPUmax=%g F64 error/sumabs=%g", maxCPU, maxRelative)
}
