package ops

import (
	"math"
	"testing"
	"unsafe"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func packBF16Pair(lo, hi uint16) uint32 {
	return uint32(lo) | (uint32(hi) << 16)
}

func uploadVkU32(t *testing.T, data []uint32) *VkBuf {
	t.Helper()
	b, err := VkBufAlloc(len(data) * 4)
	if err != nil {
		t.Fatalf("VkBufAlloc: %v", err)
	}
	t.Cleanup(b.Free)
	if len(data) > 0 {
		src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
		dst := unsafe.Slice((*byte)(b.mapped), len(data)*4)
		copy(dst, src)
	}
	return b
}

func downloadVkU32(t *testing.T, b *VkBuf, n int) []uint32 {
	t.Helper()
	out := make([]uint32, n)
	if n > 0 {
		src := unsafe.Slice((*byte)(b.mapped), n*4)
		dst := unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), n*4)
		copy(dst, src)
	}
	return out
}

func TestVulkanVecAddBF16Parity(t *testing.T) {
	requireVulkanKernel(t, "vec_add_bf16", func() bool { return vkVecAddBF16 != nil })
	aF32 := []float32{1, -2, 3.5, 4}
	bF32 := []float32{0.5, 2, -3, -4}
	aBF16 := simd.BF16FromF32Slice(aF32)
	bBF16 := simd.BF16FromF32Slice(bF32)
	wantBF16 := make([]uint16, len(aBF16))
	simd.BF16VecAdd(wantBF16, aBF16, bBF16)
	aPacked := []uint32{packBF16Pair(aBF16[0], aBF16[1]), packBF16Pair(aBF16[2], aBF16[3])}
	bPacked := []uint32{packBF16Pair(bBF16[0], bBF16[1]), packBF16Pair(bBF16[2], bBF16[3])}
	outPacked := make([]uint32, len(aPacked))
	ab := uploadVkU32(t, aPacked)
	bb := uploadVkU32(t, bPacked)
	out := uploadVkU32(t, outPacked)
	if err := VkVecAddBF16(out, ab, bb, len(aBF16)); err != nil {
		t.Fatal(err)
	}
	gotPacked := downloadVkU32(t, out, len(outPacked))
	got := []uint16{uint16(gotPacked[0] & 0xffff), uint16(gotPacked[0] >> 16), uint16(gotPacked[1] & 0xffff), uint16(gotPacked[1] >> 16)}
	for i := range got {
		gf := simd.BF16ToF32(got[i])
		wf := simd.BF16ToF32(wantBF16[i])
		if math.Abs(float64(gf-wf)) > 1e-3 {
			t.Fatalf("got[%d]=%v want %v (packed=%#x)", i, gf, wf, gotPacked)
		}
	}
}
