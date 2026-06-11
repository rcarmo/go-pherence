//go:build riscv64

package ideogram4

import simdruntime "github.com/rcarmo/go-pherence/backends/simd/runtime"

func k3RotateHalf(vec, cos, sin []float32) bool {
	half := len(cos)
	if half <= 0 || len(sin) < half || len(vec) < 2*half {
		return false
	}
	x1 := append([]float32(nil), vec[:half]...)
	x2 := append([]float32(nil), vec[half:2*half]...)
	// first half: x1*cos - x2*sin
	simdruntime.VecMul(vec[:half], x1, cos[:half])
	tmp := make([]float32, half)
	simdruntime.VecMul(tmp, x2, sin[:half])
	simdruntime.VecScaleAdd(vec[:half], vec[:half], tmp, -1)
	// second half: x2*cos + x1*sin
	simdruntime.VecMul(vec[half:2*half], x2, cos[:half])
	simdruntime.VecMul(tmp, x1, sin[:half])
	simdruntime.VecScaleAdd(vec[half:2*half], vec[half:2*half], tmp, 1)
	return true
}

func k3MRoPEToQK(q, k []float32, rope *MRoPE, tokens, heads, headDim int) bool {
	if !k3Enabled() || rope == nil || tokens <= 0 || heads <= 0 || headDim <= 0 || len(q) < tokens*heads*headDim || len(k) < tokens*heads*headDim {
		return false
	}
	// Compose existing RVV vector primitives for rotate-half. A future dedicated
	// K3 assembly kernel can fuse the copy/mul/add operations.
	for t := 0; t < tokens; t++ {
		cos := rope.cos[t*rope.half : (t+1)*rope.half]
		sin := rope.sin[t*rope.half : (t+1)*rope.half]
		for h := 0; h < heads; h++ {
			off := t*heads*headDim + h*headDim
			k3RotateHalf(q[off:off+headDim], cos, sin)
			k3RotateHalf(k[off:off+headDim], cos, sin)
		}
	}
	return true
}

func k3QwenRoPE(vec []float32, rt ropeTable, t int) bool {
	if !k3Enabled() || len(vec) < rt.headDim || t < 0 || t >= len(rt.cos)/rt.half {
		return false
	}
	base := t * rt.half
	return k3RotateHalf(vec[:rt.headDim], rt.cos[base:base+rt.half], rt.sin[base:base+rt.half])
}
