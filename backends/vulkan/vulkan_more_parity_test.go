package vulkan

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/backends/simd/kernels"
)

func TestVulkanRMSNormF32Parity(t *testing.T) {
	requireVulkanKernel(t, "rms_norm_f32", func() bool { return vkRMSNormF32 != nil })
	x := []float32{1, -2, 3, -4, 0.5, -0.25, 8}
	w := []float32{0.5, 1, -1.5, 2, 0.25, -0.75, 1.25}
	want := append([]float32(nil), x...)
	ss := float32(0)
	for _, v := range want {
		ss += v * v
	}
	inv := float32(1 / math.Sqrt(float64(ss/float32(len(want))+1e-6)))
	for i := range want {
		want[i] = w[i] * want[i] * inv
	}
	xb := uploadVkF32(t, x)
	wb := uploadVkF32(t, w)
	if err := VkRMSNormF32(xb, wb, len(x), 1e-6); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, downloadVkF32(t, xb, len(x)), want, 1e-5)
}

func TestVulkanRMSNormNoScaleF32Parity(t *testing.T) {
	requireVulkanKernel(t, "rms_norm_no_scale_f32", func() bool { return vkRMSNormNoScaleF32 != nil })
	x := []float32{1, -2, 3, -4, 0.5, -0.25, 8}
	want := append([]float32(nil), x...)
	ss := float32(0)
	for _, v := range want {
		ss += v * v
	}
	inv := float32(1 / math.Sqrt(float64(ss/float32(len(want))+1e-6)))
	for i := range want {
		want[i] *= inv
	}
	xb := uploadVkF32(t, x)
	if err := VkRMSNormNoScaleF32(xb, len(x), 1e-6); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, downloadVkF32(t, xb, len(x)), want, 1e-5)
}

func TestVulkanGELUTanhMulF32Parity(t *testing.T) {
	requireVulkanKernel(t, "gelu_tanh_mul_f32", func() bool { return vkGELUTanhMulF32 != nil })
	gate := []float32{-4, -1, -0.25, 0, 0.5, 2, 6}
	up := []float32{1.5, -2, 3, 4, -5, 6, -7}
	want := make([]float32, len(gate))
	kernels.GELUTanhMul(want, gate, up)
	gb := uploadVkF32(t, gate)
	ub := uploadVkF32(t, up)
	if err := VkGELUTanhMulF32(gb, ub, len(gate)); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, downloadVkF32(t, gb, len(gate)), want, 1e-4)
}
