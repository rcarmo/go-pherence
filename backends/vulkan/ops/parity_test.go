package ops

import (
	"math"
	"strings"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func requireVulkanKernel(t *testing.T, name string, ready func() bool) {
	t.Helper()
	if !VulkanInit() {
		t.Skip("vulkan runtime not available")
	}
	initVkKernels()
	if !ready() {
		t.Skipf("vulkan %s pipeline not available", name)
	}
}

func uploadVkF32(t *testing.T, data []float32) *VkBuf {
	t.Helper()
	b, err := VkBufAlloc(len(data) * 4)
	if err != nil {
		t.Fatalf("VkBufAlloc: %v", err)
	}
	b.Upload(data)
	t.Cleanup(b.Free)
	return b
}

func downloadVkF32(t *testing.T, b *VkBuf, n int) []float32 {
	t.Helper()
	out := make([]float32, n)
	b.Download(out)
	return out
}

func assertCloseF32(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d len(want)=%d", len(got), len(want))
	}
	for i := range got {
		if d := float32(math.Abs(float64(got[i] - want[i]))); d > tol {
			t.Fatalf("[%d] got %g want %g diff %g tol %g", i, got[i], want[i], d, tol)
		}
	}
}

func TestVulkanVecAddF32Parity(t *testing.T) {
	requireVulkanKernel(t, "vec_add_f32", func() bool { return vkVecAddF32 != nil })
	a := []float32{1, -2, 3.5, 4, -5, 6.25, 7, -8.5, 9}
	b := []float32{0.5, 2, -3, 4, 5.5, -6, 7.25, 8, -9}
	want := make([]float32, len(a))
	for i := range want {
		want[i] = a[i] + b[i]
	}
	ab := uploadVkF32(t, a)
	bb := uploadVkF32(t, b)
	out := uploadVkF32(t, make([]float32, len(a)))
	if err := VkVecAddF32(out, ab, bb, len(a)); err != nil {
		if strings.Contains(err.Error(), "not available") {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	assertCloseF32(t, downloadVkF32(t, out, len(a)), want, 1e-6)
}

func TestVulkanSiLUMulF32Parity(t *testing.T) {
	requireVulkanKernel(t, "silu_mul_f32", func() bool { return vkSiLUMulF32 != nil })
	gate := []float32{-4, -1, -0.25, 0, 0.5, 2, 6}
	up := []float32{1.5, -2, 3, 4, -5, 6, -7}
	want := make([]float32, len(gate))
	for i := range want {
		g := gate[i]
		want[i] = g / (1 + float32(math.Exp(float64(-g)))) * up[i]
	}
	gb := uploadVkF32(t, gate)
	ub := uploadVkF32(t, up)
	out := uploadVkF32(t, make([]float32, len(gate)))
	if err := VkSiLUMulF32(out, gb, ub, len(gate)); err != nil {
		if strings.Contains(err.Error(), "not available") {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	assertCloseF32(t, downloadVkF32(t, out, len(gate)), want, 1e-5)
}

func TestVulkanGemvF32Parity(t *testing.T) {
	requireVulkanKernel(t, "gemv_f32", func() bool { return vkGemvF32 != nil })
	inDim, outDim := 5, 3
	x := []float32{1, -2, 3, 0.5, -1.5}
	w := []float32{
		1, 2, 3, 4, 5,
		-1, 0.5, 2, -3, 1,
		0.25, -0.5, 0.75, -1, 1.25,
	}
	want := make([]float32, outDim)
	if !simd.GemvRows(want, x, w, outDim, inDim) {
		t.Fatal("GemvRows returned false")
	}
	xb := uploadVkF32(t, x)
	wb := uploadVkF32(t, w)
	out := uploadVkF32(t, make([]float32, outDim))
	if err := VkGemvF32(out, xb, wb, inDim, outDim); err != nil {
		if strings.Contains(err.Error(), "not available") {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	assertCloseF32(t, downloadVkF32(t, out, outDim), want, 1e-4)
}
