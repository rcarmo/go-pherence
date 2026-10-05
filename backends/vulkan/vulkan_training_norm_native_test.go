package vulkan

import (
	"context"
	"math"
	"os"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func TestVulkanNativeNormUnitMaximum(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_NATIVE_NORM") != "1" {
		t.Skip("set GO_PHERENCE_VULKAN_NATIVE_NORM=1 for explicit native norm regression")
	}
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" {
		t.Fatal("Vulkan disabled")
	}
	if !VulkanInit() {
		t.Fatal("Vulkan unavailable; no fallback")
	}
	ctx := context.Background()
	op, e := NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	d := &VulkanResidentDriver{Operator: op}
	var owned []train.ResidentBuffer
	alloc := func(n int) train.ResidentBuffer {
		b, e := d.NewBuffer(ctx, n)
		if e != nil {
			t.Fatal(e)
		}
		owned = append(owned, b)
		return b
	}
	a, b, clip := alloc(4), alloc(4), alloc(4)
	// The first value reproduced 0.9999999 for native x/x before the fix.
	for _, n := range []int{1, 3, 255, 256, 257, 513} {
		for _, sign := range []float32{1, -1} {
			values := make([]float32, n)
			values[n-1] = sign * .022259595
			x := alloc(n)
			count, _ := train.NormStatsShape(n)
			stats := alloc(count)
			if e = x.Upload(ctx, values); e != nil {
				t.Fatal(e)
			}
			if e = d.NormStats(ctx, stats, x, n, false); e != nil {
				t.Fatal(e)
			}
			summaries := make([]float32, count)
			if e = stats.Download(ctx, summaries); e != nil {
				t.Fatal(e)
			}
			for i := 0; i < count; i += 3 {
				if summaries[i+2] != 0 || summaries[i] < 0 || summaries[i+1] < 0 {
					t.Fatal("invalid norm stats", n, sign, summaries)
				}
				if summaries[i] > 0 && summaries[i+1] != 1 {
					t.Fatal("single maximum must contribute exactly one", n, sign, summaries)
				}
			}
			if e = a.Upload(ctx, []float32{0, 0, 0, 0}); e != nil {
				t.Fatal(e)
			}
			if e = d.NormMerge(ctx, b, a, stats, n, 0, 0); e != nil {
				t.Fatal(e)
			}
			if e = d.NormMerge(ctx, a, b, stats, n, 0, 2); e != nil {
				t.Fatal(e)
			}
			if e = d.NormClip(ctx, clip, a, int64(n), .01); e != nil {
				t.Fatal(e)
			}
			result := make([]float32, 4)
			if e = clip.Download(ctx, result); e != nil {
				t.Fatal(e)
			}
			norm, scale, e := train.DecodeGlobalNorm(result, int64(n))
			if e != nil {
				t.Fatal(n, sign, result, e)
			}
			want := math.Abs(float64(values[n-1]))
			rel := math.Abs(norm-want) / want
			if rel > 3e-7 || math.Abs(float64(scale)-.01/want) > 3e-7 {
				t.Fatal("independent norm/clip mismatch", n, sign, result, norm, rel)
			}
			t.Logf("n=%d sign=%g native norm=%g relativeError=%g clip=%g", n, sign, norm, rel, scale)
		}
	}
	// Multichunk, mixed magnitudes: independent F64 reduction and device clip.
	n := 513
	values := make([]float32, n)
	var square float64
	for i := range values {
		values[i] = float32((i%37)-18) * .00073
		square += float64(values[i]) * float64(values[i])
	}
	x := alloc(n)
	count, _ := train.NormStatsShape(n)
	stats := alloc(count)
	if e = x.Upload(ctx, values); e != nil {
		t.Fatal(e)
	}
	if e = d.NormStats(ctx, stats, x, n, false); e != nil {
		t.Fatal(e)
	}
	if e = a.Upload(ctx, []float32{0, 0, 0, 0}); e != nil {
		t.Fatal(e)
	}
	if e = d.NormMerge(ctx, b, a, stats, n, 0, 0); e != nil {
		t.Fatal(e)
	}
	if e = d.NormMerge(ctx, a, b, stats, n, 0, 2); e != nil {
		t.Fatal(e)
	}
	if e = d.NormClip(ctx, clip, a, int64(n), .01); e != nil {
		t.Fatal(e)
	}
	result := make([]float32, 4)
	if e = clip.Download(ctx, result); e != nil {
		t.Fatal(e)
	}
	norm, scale, e := train.DecodeGlobalNorm(result, int64(n))
	if e != nil {
		t.Fatal(result, e)
	}
	want := math.Sqrt(square)
	rel := math.Abs(norm-want) / want
	if rel > 3e-7 || math.Abs(float64(scale)-.01/want) > 3e-7 {
		t.Fatal("mixed independent norm/clip mismatch", result, norm, rel)
	}
	t.Logf("mixed n=%d norm=%g reference=%g relativeError=%g", n, norm, want, rel)
	for i := len(owned) - 1; i >= 0; i-- {
		if e = owned[i].Close(); e != nil {
			t.Fatal("close", e)
		}
	}
	if e = d.Close(); e != nil {
		t.Fatal(e)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
}
