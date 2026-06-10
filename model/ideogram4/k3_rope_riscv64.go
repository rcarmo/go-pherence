//go:build riscv64

package ideogram4

func k3MRoPEToQK(q, k []float32, rope *MRoPE, tokens, heads, headDim int) bool {
	if !k3Enabled() || rope == nil || tokens <= 0 || heads <= 0 || headDim <= 0 || len(q) < tokens*heads*headDim || len(k) < tokens*heads*headDim {
		return false
	}
	// K3 runtime seam for DiT MRoPE. Current body preserves scalar semantics;
	// replace with RVV rotate-half kernel using k3_isa.h macros.
	for t := 0; t < tokens; t++ {
		for h := 0; h < heads; h++ {
			off := t*heads*headDim + h*headDim
			rope.applyToHead(q[off:off+headDim], t)
			rope.applyToHead(k[off:off+headDim], t)
		}
	}
	return true
}

func k3QwenRoPE(vec []float32, rt ropeTable, t int) bool {
	if !k3Enabled() || len(vec) < rt.headDim || t < 0 || t >= len(rt.cos)/rt.half {
		return false
	}
	base := t * rt.half
	for i := 0; i < rt.half; i++ {
		x1 := vec[i]
		x2 := vec[rt.half+i]
		c := rt.cos[base+i]
		s := rt.sin[base+i]
		vec[i] = x1*c - x2*s
		vec[rt.half+i] = x2*c + x1*s
	}
	return true
}
