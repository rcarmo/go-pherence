//go:build linux

package vulkan

import (
	"context"
	"math"
	"os"
	"testing"
	"time"
)

func TestVulkanNativeTrainingNormAccuracy(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_NORM_ACCURACY_NATIVE") != "1" {
		t.Skip("explicit norm rounding qualification")
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
	var maxInv, maxValue float64
	for _, n := range []int{2, 7, 32, 257, 1024, 4096} {
		x, xt, w, b := make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*.17)) * .4
			xt[i] = float32(math.Cos(float64(i)*.11)) * .02
			w[i] = 1
			b[i] = .01
		}
		a, e := NewVkTensorArena(ctx, 4*(8*n+10)+512)
		if e != nil {
			t.Fatal(e)
		}
		tensor := func(v []float32, sh ...int) *VkTensorF32 {
			z, e := a.AllocF32(ctx, sh...)
			if e != nil {
				t.Fatal(e)
			}
			if e = z.Upload(ctx, v); e != nil {
				t.Fatal(e)
			}
			return z
		}
		xv, tv, wv, bv, stats, scratch, out := tensor(x, 1, n), tensor(xt, 1, n), tensor(w, n), tensor(b, n), tensor(make([]float32, 5), 1, 5), tensor(make([]float32, 5), 1, 5), tensor(make([]float32, n), 1, n)
		var mean, variance float32
		for _, v := range x {
			mean += v
		}
		mean /= float32(n)
		for _, v := range x {
			d := v - mean
			variance += d * d
		}
		for kind := uint32(0); kind < 2; kind++ {
			den := float32(n)
			if kind == 1 {
				den--
			}
			v := variance / den
			inv := float32(1 / math.Sqrt(float64(v+float32(1e-5))))
			for mode := uint32(0); mode < 2; mode++ {
				dst, st := out, stats
				if mode == 0 {
					dst, st = stats, scratch
				}
				s, e := op.NormDualForwardStage(ctx, dst, xv, tv, wv, bv, st, kind, mode, 1e-5)
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
				size := n
				if mode == 0 {
					size = 5
				}
				got := make([]float32, size)
				if e = dst.Download(ctx, got); e != nil {
					t.Fatal(e)
				}
				if e = p.Close(); e != nil {
					t.Fatal(e)
				}
				if mode == 0 {
					d := math.Abs(float64(got[2]) - float64(inv))
					maxInv = math.Max(maxInv, d)
					if d > float64(inv)*1.2e-7 {
						t.Fatal("norm inverse rounding", n, kind, got[2], inv)
					}
				} else {
					for i, g := range got {
						want := x[i] * inv
						if kind == 0 {
							want = (x[i]-mean)*inv + b[i]
						}
						d := math.Abs(float64(g) - float64(want))
						maxValue = math.Max(maxValue, d)
						if d > 2e-6 {
							t.Fatal("norm value", n, kind, d)
						}
					}
				}
			}
		}
		// Primal-only LayerNorm VJP against the independent CPU association.
		zero, seed := make([]float32, n), make([]float32, n)
		for i := range seed {
			seed[i] = float32(math.Cos(float64(i)*.23)) * .03
		}
		zv, gv := tensor(zero, 1, n), tensor(seed, 1, n)
		if e = tv.Upload(ctx, zero); e != nil {
			t.Fatal(e)
		}
		forward, e := op.NormDualForwardStage(ctx, stats, xv, tv, wv, bv, scratch, 0, 0, 1e-5)
		if e != nil {
			t.Fatal(e)
		}
		plan, e := NewVkF32Plan(ctx, []VkF32Stage{forward})
		if e != nil {
			t.Fatal(e)
		}
		if e = plan.Run(ctx); e != nil {
			t.Fatal(e)
		}
		if e = plan.Close(); e != nil {
			t.Fatal(e)
		}
		inv := float32(1 / math.Sqrt(float64(variance/float32(n)+float32(1e-5))))
		var sg, dg float32
		for i, g := range seed {
			base := (x[i] - mean) * inv
			sg += g
			dg += g * base
		}
		sg /= float32(n)
		dg /= float32(n)
		for mode := uint32(0); mode < 2; mode++ {
			stage, e := op.NormDualBackwardStage(ctx, out, xv, zv, wv, gv, zv, stats, 0, mode)
			if e != nil {
				t.Fatal(e)
			}
			p, e := NewVkF32Plan(ctx, []VkF32Stage{stage})
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
			for i, g := range got {
				want := float32(0)
				if mode == 0 {
					base := (x[i] - mean) * inv
					want = inv * (seed[i] - sg - base*dg)
				}
				if math.Abs(float64(g-want)) > 2e-6 {
					t.Fatal("primal norm VJP", n, mode, i, g, want)
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
	t.Logf("norm inverse max=%g value max=%g", maxInv, maxValue)
}
