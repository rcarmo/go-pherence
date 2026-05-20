package vulkan

// Vulkan compute operations for inference.
// Each operation has a GLSL source (for documentation/regeneration)
// and pre-compiled SPIR-V binary.
//
// GLSL sources are included as comments. To regenerate SPIR-V:
//   glslangValidator -V -S comp shader.glsl -o shader.spv
//
// All shaders use:
//   - layout(local_size_x = 256) for workgroup size
//   - Storage buffers (binding 0, 1, ...) for data
//   - Push constants for dimensions/parameters
//   - BF16 emulated via uint16 bitshift (no extensions needed)

import (
	"fmt"
	"sync"
	"unsafe"
)

// Vulkan kernel cache
var (
	vkKernelOnce         sync.Once
	vkVecAddF32          *VkComputeKernel
	vkVecAddBF16         *VkComputeKernel
	vkRMSNormF32         *VkComputeKernel
	vkRMSNormBF16        *VkComputeKernel
	vkRMSNormNoScaleF32  *VkComputeKernel
	vkGemvF32            *VkComputeKernel
	vkGemvBF16Mixed      *VkComputeKernel
	vkSiLUMulF32         *VkComputeKernel
	vkGELUTanhMulF32     *VkComputeKernel
	vkRoPEPartialF32     *VkComputeKernel
	vkAttentionScoresF32 *VkComputeKernel
)

// initVkKernels compiles embedded Vulkan compute shaders into optional kernels.
func initVkKernels() {
	vkKernelOnce.Do(func() {
		if !vkReady {
			return
		}
		create := func(name string, spirv []byte, numBuffers, pushConstantSize int) *VkComputeKernel {
			if len(spirv) == 0 || len(spirv)%4 != 0 {
				debugf("[vulkan] %s SPIR-V has invalid length %d\n", name, len(spirv))
				return nil
			}
			k, err := VkKernelCreate(spirv, numBuffers, pushConstantSize)
			if err != nil {
				debugf("[vulkan] %s pipeline unavailable: %v\n", name, err)
				return nil
			}
			debugf("[vulkan] %s pipeline ready\n", name)
			return k
		}

		vkVecAddF32 = create("vec_add_f32", spirv_vec_add_f32, 3, 4)
		vkVecAddBF16 = create("vec_add_bf16", spirv_vec_add_bf16, 3, 4)
		vkRMSNormF32 = create("rms_norm_f32", spirv_rms_norm_f32, 2, 8)
		vkRMSNormBF16 = create("rms_norm_bf16", spirv_rms_norm_bf16, 2, 8)
		vkRMSNormNoScaleF32 = create("rms_norm_no_scale_f32", spirv_rms_norm_no_scale_f32, 1, 8)
		vkGemvF32 = create("gemv_f32", spirv_gemv_f32, 3, 8)
		vkGemvBF16Mixed = create("gemv_bf16_mixed", spirv_gemv_bf16_mixed, 3, 8)
		vkSiLUMulF32 = create("silu_mul_f32", spirv_silu_mul_f32, 3, 4)
		vkGELUTanhMulF32 = create("gelu_tanh_mul_f32", spirv_gelu_tanh_mul_f32, 2, 4)
		vkRoPEPartialF32 = create("rope_partial_f32", spirv_rope_partial_f32, 2, 16)
		vkAttentionScoresF32 = create("attention_score", spirv_attention_score, 3, 20)
	})
}

// VkVecAddF32 dispatches c[i] = a[i] + b[i] on Vulkan.
func VkVecAddF32(dst, a, b *VkBuf, n int) error {
	initVkKernels()
	if n <= 0 || !vkBufHasFloat32s(dst, n) || !vkBufHasFloat32s(a, n) || !vkBufHasFloat32s(b, n) {
		return fmt.Errorf("invalid vulkan vec_add_f32 buffers n=%d", n)
	}
	if vkVecAddF32 == nil {
		return fmt.Errorf("vulkan vec_add_f32 not available (SPIR-V needs glslangValidator)")
	}
	nn := uint32(n)
	groups := (nn + 255) / 256
	return vkVecAddF32.Dispatch(groups, 1, 1, []*VkBuf{a, b, dst}, unsafe.Pointer(&nn))
}

func vkBufHasBytes(b *VkBuf, n int) bool {
	return b != nil && n >= 0 && b.size >= uint64(n)
}

// VkVecAddBF16 dispatches c[i] = BF16(F32(a[i]) + F32(b[i])) on Vulkan.
func VkVecAddBF16(dst, a, b *VkBuf, n int) error {
	initVkKernels()
	if n <= 0 || n%2 != 0 {
		return fmt.Errorf("invalid vulkan vec_add_bf16 element count n=%d", n)
	}
	packedBytes := (n / 2) * 4
	if !vkBufHasBytes(dst, packedBytes) || !vkBufHasBytes(a, packedBytes) || !vkBufHasBytes(b, packedBytes) {
		return fmt.Errorf("invalid vulkan vec_add_bf16 buffers n=%d", n)
	}
	if vkVecAddBF16 == nil {
		return fmt.Errorf("vulkan vec_add_bf16 not available")
	}
	// BF16 packed: 2 elements per uint32, so dispatch n/2 threads
	nn := uint32(n / 2)
	groups := (nn + 255) / 256
	return vkVecAddBF16.Dispatch(groups, 1, 1, []*VkBuf{a, b, dst}, unsafe.Pointer(&nn))
}

// SPIR-V for BF16 vec_add (packed: 2× BF16 per uint32)
// GLSL source:
//
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer A { uint a[]; };
// layout(set=0, binding=1) buffer B { uint b[]; };
// layout(set=0, binding=2) buffer C { uint c[]; };
// layout(push_constant) uniform P { uint n; };  // n = number of uint32 pairs
//
//	void main() {
//	    uint i = gl_GlobalInvocationID.x;
//	    if (i >= n) return;
//	    uint pa = a[i], pb = b[i];
//	    // Unpack 2× BF16, widen to F32
//	    float a0 = uintBitsToFloat(pa << 16);       // lower BF16
//	    float a1 = uintBitsToFloat(pa & 0xFFFF0000); // upper BF16
//	    float b0 = uintBitsToFloat(pb << 16);
//	    float b1 = uintBitsToFloat(pb & 0xFFFF0000);
//	    // Add in F32
//	    float c0 = a0 + b0;
//	    float c1 = a1 + b1;
//	    // Pack back: narrow F32→BF16
//	    c[i] = (floatBitsToUint(c0) >> 16) | (floatBitsToUint(c1) & 0xFFFF0000);
//	}
var spirvBF16VecAdd = buildSPIRVBF16VecAdd()

func buildSPIRVBF16VecAdd() []byte {
	// For now, use the same F32 vec_add SPIR-V as placeholder.
	// The BF16 packing logic needs proper SPIR-V encoding which is complex
	// to hand-assemble. In production, use glslangValidator.
	return buildSPIRVVecAdd()
}

// ---- GLSL sources for all kernels (for documentation/regeneration) ----

// GLSL: F32 RMSNorm
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer X { float x[]; };
// layout(set=0, binding=1) buffer W { float w[]; };
// layout(push_constant) uniform P { uint n; float eps; };
// shared float sdata[256];
// void main() {
//     uint tid = gl_LocalInvocationID.x;
//     uint gid = gl_GlobalInvocationID.x;
//     // Phase 1: partial sum of squares
//     float ss = 0.0;
//     for (uint i = tid; i < n; i += 256) ss += x[i] * x[i];
//     sdata[tid] = ss;
//     barrier();
//     // Tree reduce
//     for (uint s = 128; s > 0; s >>= 1) {
//         if (tid < s) sdata[tid] += sdata[tid + s];
//         barrier();
//     }
//     float invRMS = inversesqrt(sdata[0] / float(n) + eps);
//     barrier();
//     // Phase 2: apply
//     for (uint i = tid; i < n; i += 256) x[i] = w[i] * x[i] * invRMS;
// }

// GLSL: BF16 RMSNorm
// Same as F32 but x/w are uint[] with BF16 packed as lower 16 bits of each uint.
// Widen: uintBitsToFloat(x[i] << 16)
// Narrow: floatBitsToUint(result) >> 16

// GLSL: F32 GEMV (matrix-vector multiply)
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer X { float x[]; };     // [inDim]
// layout(set=0, binding=1) buffer W { float w[]; };     // [outDim * inDim] row-major
// layout(set=0, binding=2) buffer OUT { float out[]; }; // [outDim]
// layout(push_constant) uniform P { uint inDim; uint outDim; };
// shared float sdata[256];
// void main() {
//     uint row = gl_WorkGroupID.x;
//     uint tid = gl_LocalInvocationID.x;
//     if (row >= outDim) return;
//     float sum = 0.0;
//     for (uint i = tid; i < inDim; i += 256)
//         sum += w[row * inDim + i] * x[i];
//     sdata[tid] = sum;
//     barrier();
//     for (uint s = 128; s > 0; s >>= 1) {
//         if (tid < s) sdata[tid] += sdata[tid + s];
//         barrier();
//     }
//     if (tid == 0) out[row] = sdata[0];
// }

// GLSL: BF16 GEMV
// Same structure but x is uint[] (BF16), w is float[] (F32 weights).
// Mixed precision: BF16 activations × F32 weights → BF16 output.
// Output: out is uint[] with BF16 in lower 16 bits.

func vkCheckedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}

func vkBufHasFloat32s(b *VkBuf, n int) bool {
	if b == nil || n < 0 {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	if n > maxInt/4 {
		return false
	}
	return b.size >= uint64(n*4)
}

func vkUnavailable(name string) error {
	return fmt.Errorf("vulkan %s not available (SPIR-V pipeline wiring pending)", name)
}

// VkRMSNormF32 dispatches x[i] = w[i] * x[i] / rms(x) on Vulkan.
func VkRMSNormF32(x, w *VkBuf, n int, eps float32) error {
	initVkKernels()
	if n <= 0 || !vkBufHasFloat32s(x, n) || !vkBufHasFloat32s(w, n) {
		return fmt.Errorf("invalid vulkan rms_norm_f32 buffers n=%d", n)
	}
	if vkRMSNormF32 == nil {
		return vkUnavailable("rms_norm_f32")
	}
	push := struct {
		N   uint32
		Eps float32
	}{uint32(n), eps}
	return vkRMSNormF32.Dispatch(1, 1, 1, []*VkBuf{x, w}, unsafe.Pointer(&push))
}

// VkRMSNormNoScaleF32 dispatches x[i] = x[i] / rms(x) on Vulkan.
func VkRMSNormNoScaleF32(x *VkBuf, n int, eps float32) error {
	initVkKernels()
	if n <= 0 || !vkBufHasFloat32s(x, n) {
		return fmt.Errorf("invalid vulkan rms_norm_no_scale_f32 buffer n=%d", n)
	}
	if vkRMSNormNoScaleF32 == nil {
		return vkUnavailable("rms_norm_no_scale_f32")
	}
	push := struct {
		N   uint32
		Eps float32
	}{uint32(n), eps}
	return vkRMSNormNoScaleF32.Dispatch(1, 1, 1, []*VkBuf{x}, unsafe.Pointer(&push))
}

// VkGemvF32 dispatches out[outDim] = W[outDim,inDim] · x[inDim] on Vulkan.
func VkGemvF32(out, x, w *VkBuf, inDim, outDim int) error {
	initVkKernels()
	weightLen, ok := vkCheckedMulInt(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || !vkBufHasFloat32s(out, outDim) || !vkBufHasFloat32s(x, inDim) || !vkBufHasFloat32s(w, weightLen) {
		return fmt.Errorf("invalid vulkan gemv_f32 dims in=%d out=%d", inDim, outDim)
	}
	if vkGemvF32 == nil {
		return vkUnavailable("gemv_f32")
	}
	push := struct{ InDim, OutDim uint32 }{uint32(inDim), uint32(outDim)}
	return vkGemvF32.Dispatch(uint32(outDim), 1, 1, []*VkBuf{x, w, out}, unsafe.Pointer(&push))
}

// VkSiLUMulF32 dispatches dst[i] = silu(gate[i]) * up[i] on Vulkan.
func VkSiLUMulF32(dst, gate, up *VkBuf, n int) error {
	initVkKernels()
	if n <= 0 || !vkBufHasFloat32s(dst, n) || !vkBufHasFloat32s(gate, n) || !vkBufHasFloat32s(up, n) {
		return fmt.Errorf("invalid vulkan silu_mul_f32 buffers n=%d", n)
	}
	if vkSiLUMulF32 == nil {
		return vkUnavailable("silu_mul_f32")
	}
	nn := uint32(n)
	groups := (nn + 255) / 256
	return vkSiLUMulF32.Dispatch(groups, 1, 1, []*VkBuf{gate, up, dst}, unsafe.Pointer(&nn))
}

// VkGELUTanhMulF32 dispatches gate[i] = gelu_tanh(gate[i]) * up[i] on Vulkan.
func VkGELUTanhMulF32(gate, up *VkBuf, n int) error {
	initVkKernels()
	if n <= 0 || !vkBufHasFloat32s(gate, n) || !vkBufHasFloat32s(up, n) {
		return fmt.Errorf("invalid vulkan gelu_tanh_mul_f32 buffers n=%d", n)
	}
	if vkGELUTanhMulF32 == nil {
		return vkUnavailable("gelu_tanh_mul_f32")
	}
	nn := uint32(n)
	groups := (nn + 255) / 256
	return vkGELUTanhMulF32.Dispatch(groups, 1, 1, []*VkBuf{gate, up}, unsafe.Pointer(&nn))
}

// VkRoPEPartialF32 dispatches partial rotary embedding on Vulkan.
func VkRoPEPartialF32(x, freqs *VkBuf, pos, nHeads, headDim, rotHalf int) error {
	initVkKernels()
	total, okTotal := vkCheckedMulInt(nHeads, headDim)
	pairs, okPairs := vkCheckedMulInt(nHeads, rotHalf)
	posPairs, okPos := vkCheckedMulInt(pos+1, rotHalf)
	freqNeed, okFreq := vkCheckedMulInt(posPairs, 2)
	if pos < 0 || nHeads <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf > headDim/2 || !okTotal || !okPairs || !okPos || !okFreq || !vkBufHasFloat32s(x, total) || !vkBufHasFloat32s(freqs, freqNeed) {
		return fmt.Errorf("invalid vulkan rope_partial_f32 dims pos=%d heads=%d headDim=%d rotHalf=%d", pos, nHeads, headDim, rotHalf)
	}
	if vkRoPEPartialF32 == nil {
		return vkUnavailable("rope_partial_f32")
	}
	push := struct{ Pos, Heads, HeadDim, RotHalf uint32 }{uint32(pos), uint32(nHeads), uint32(headDim), uint32(rotHalf)}
	groups := uint32((pairs + 255) / 256)
	return vkRoPEPartialF32.Dispatch(groups, 1, 1, []*VkBuf{x, freqs}, unsafe.Pointer(&push))
}

// VkAttentionScoresF32 dispatches GQA attention scores on Vulkan.
func VkAttentionScoresF32(out, q, kCache *VkBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) error {
	initVkKernels()
	qLen, okQ := vkCheckedMulInt(nHeads, headDim)
	kvDim, okKV := vkCheckedMulInt(nKVHeads, headDim)
	cacheLen, okCache := vkCheckedMulInt(seqLen, kvDim)
	scoreLen, okScore := vkCheckedMulInt(nHeads, seqLen)
	if seqLen <= 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 || !okQ || !okKV || !okCache || !okScore || !vkBufHasFloat32s(out, scoreLen) || !vkBufHasFloat32s(q, qLen) || !vkBufHasFloat32s(kCache, cacheLen) {
		return fmt.Errorf("invalid vulkan attention_score dims seq=%d heads=%d kvHeads=%d headDim=%d", seqLen, nHeads, nKVHeads, headDim)
	}
	if vkAttentionScoresF32 == nil {
		return vkUnavailable("attention_score_f32")
	}
	push := struct {
		Heads, KVHeads, HeadDim, SeqLen uint32
		Scale                           float32
	}{uint32(nHeads), uint32(nKVHeads), uint32(headDim), uint32(seqLen), scale}
	return vkAttentionScoresF32.Dispatch(uint32(nHeads), 1, 1, []*VkBuf{q, kCache, out}, unsafe.Pointer(&push))
}
